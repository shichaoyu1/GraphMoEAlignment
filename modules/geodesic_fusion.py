"""Metric-geodesic modality graph fusion for Paper 4.

This is a compact discriminative adaptation of Metric Flow Matching. It learns
endpoint-preserving curved paths under a data-induced diagonal metric; it does
not implement a generative flow model.
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _masked_mean(values, mask, dim):
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp(min=1.0)


class GeodesicModalityGraphFusion(nn.Module):
    """Fuse a modality matrix through metric-geodesic graph edges."""

    def __init__(
        self,
        shared_dim: int,
        num_modalities: int = 4,
        fusion_mode: str = "geodesic",
        metric_support: str = "case_and_anchors",
        path_steps: int = 5,
        gamma: float = 0.5,
        rho: float = 1e-3,
        metric_alpha: float = 1.0,
        graph_temperature: float = 1.0,
        bend_init: float = 0.1,
        use_graph: bool = True,
    ):
        super().__init__()
        if fusion_mode not in {"concat", "euclidean", "geodesic"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if metric_support not in {"case_and_anchors", "case_only"}:
            raise ValueError(f"Unsupported metric support: {metric_support}")
        if path_steps < 3:
            raise ValueError("Geodesic paths require at least three samples")
        if gamma <= 0 or rho <= 0 or graph_temperature <= 0:
            raise ValueError("gamma, rho, and graph_temperature must be positive")

        self.shared_dim = int(shared_dim)
        self.num_modalities = int(num_modalities)
        self.fusion_mode = fusion_mode
        self.metric_support = metric_support
        self.path_steps = int(path_steps)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.metric_alpha = float(metric_alpha)
        self.graph_temperature = float(graph_temperature)
        self.use_graph = bool(use_graph)

        pairs = list(itertools.combinations(range(self.num_modalities), 2))
        self.register_buffer("pair_indices", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.register_buffer("path_times", torch.linspace(0.0, 1.0, self.path_steps), persistent=False)

        hidden_dim = max(self.shared_dim * 2, 32)
        self.geopath_net = nn.Sequential(
            nn.Linear(self.shared_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shared_dim),
        )
        self.message_net = nn.Sequential(
            nn.Linear(self.shared_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shared_dim),
        )
        self.graph_norm = nn.LayerNorm(self.shared_dim)
        self.concat_projection = nn.Sequential(
            nn.Linear(self.num_modalities * self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.SiLU(),
        )
        self.bend_logit = nn.Parameter(torch.tensor(_logit(bend_init), dtype=torch.float32))

    @property
    def num_pairs(self):
        return int(self.pair_indices.shape[0])

    def _anchor_context(self, modality_nodes, anchor_prototypes, modality_mask):
        pooled = _masked_mean(modality_nodes, modality_mask[:, None, :], dim=2).mean(dim=1)
        pooled = F.normalize(pooled, dim=-1)
        prototypes = F.normalize(anchor_prototypes, dim=-1)
        weights = torch.softmax(pooled @ prototypes.T / math.sqrt(self.shared_dim), dim=-1)
        return F.normalize(weights @ prototypes, dim=-1)

    def pair_path(self, start, end, context, mode=None):
        """Return endpoint-preserving paths for matching leading dimensions."""
        mode = self.fusion_mode if mode is None else mode
        times = self.path_times.to(dtype=start.dtype, device=start.device)
        view_shape = (1,) * (start.ndim - 1) + (self.path_steps, 1)
        t = times.view(view_shape)
        linear = (1.0 - t) * start.unsqueeze(-2) + t * end.unsqueeze(-2)
        if mode != "geodesic":
            return linear, linear

        descriptor = torch.cat([start + end, (start - end).abs(), context], dim=-1)
        direction = F.normalize(self.geopath_net(descriptor), dim=-1, eps=1e-8)
        endpoint_distance = (end - start).norm(dim=-1, keepdim=True).unsqueeze(-2)
        envelope = 4.0 * t * (1.0 - t)
        path = (
            linear
            + envelope
            * torch.sigmoid(self.bend_logit)
            * endpoint_distance
            * direction.unsqueeze(-2)
        )
        return path, linear

    def land_metric_diagonal(self, points, supports, support_mask=None):
        """Evaluate the detached-support LAND diagonal metric."""
        supports = supports.detach()
        prefix = [supports.shape[0]] + [1] * (points.ndim - 2) + [supports.shape[1], supports.shape[2]]
        differences = points.unsqueeze(-2) - supports.view(prefix)
        squared = differences.square()
        weights = torch.exp(-squared.sum(dim=-1) / (2.0 * self.gamma**2))
        if support_mask is not None:
            mask_shape = [support_mask.shape[0]] + [1] * (points.ndim - 2) + [support_mask.shape[1]]
            weights = weights * support_mask.to(weights.dtype).view(mask_shape)
        local_variance = (weights.unsqueeze(-1) * squared).sum(dim=-2) + self.rho
        return local_variance.reciprocal().pow(self.metric_alpha)

    def path_energy(self, paths, supports, support_mask=None):
        dt = 1.0 / float(self.path_steps - 1)
        velocities = (paths[..., 1:, :] - paths[..., :-1, :]) / dt
        midpoints = 0.5 * (paths[..., 1:, :] + paths[..., :-1, :])
        metric = self.land_metric_diagonal(midpoints, supports, support_mask)
        # Scaling by D preserves the minimizing path and avoids exp(-E)
        # underflow when the latent dimension is large.
        segment_energy = (velocities.square() * metric).mean(dim=-1)
        return segment_energy.sum(dim=-1) * dt

    def _case_supports(self, modality_nodes, anchor_prototypes, modality_mask):
        batch, regions = modality_nodes.shape[:2]
        case_nodes = modality_nodes.reshape(batch, -1, self.shared_dim).detach()
        case_mask = modality_mask[:, None, :].expand(-1, regions, -1).reshape(batch, -1)
        if self.metric_support == "case_only":
            return case_nodes, case_mask
        prototypes = F.normalize(anchor_prototypes, dim=-1).detach()
        prototype_mask = torch.ones(
            batch, prototypes.shape[0], device=modality_nodes.device, dtype=torch.bool
        )
        supports = torch.cat([case_nodes, prototypes.unsqueeze(0).expand(batch, -1, -1)], dim=1)
        return supports, torch.cat([case_mask, prototype_mask], dim=1)

    def _edge_matrices(self, pair_values, pair_vectors, pair_valid):
        batch, regions = pair_values.shape[:2]
        matrix = pair_values.new_zeros(batch, regions, self.num_modalities, self.num_modalities)
        vectors = pair_vectors.new_zeros(
            batch, regions, self.num_modalities, self.num_modalities, self.shared_dim
        )
        for pair_idx, (source, target) in enumerate(self.pair_indices.tolist()):
            valid = pair_valid[:, None, pair_idx].to(pair_values.dtype)
            value = pair_values[:, :, pair_idx] * valid
            vector = pair_vectors[:, :, pair_idx] * valid.unsqueeze(-1)
            matrix[:, :, source, target] = value
            matrix[:, :, target, source] = value
            vectors[:, :, source, target] = vector
            vectors[:, :, target, source] = vector
        return matrix, vectors

    def _graph_fuse(self, modality_nodes, context, pair_summary, energy, pair_valid, modality_mask):
        energy_matrix, path_matrix = self._edge_matrices(energy, pair_summary, pair_valid)
        edge_mask, _ = self._edge_matrices(torch.ones_like(energy), pair_summary, pair_valid)
        kernel = torch.exp(-energy_matrix.clamp(max=50.0) / self.graph_temperature) * edge_mask
        valid = modality_mask[:, None, :].to(kernel.dtype)
        identity = torch.eye(self.num_modalities, device=kernel.device, dtype=kernel.dtype)
        kernel = kernel + identity[None, None] * valid.unsqueeze(-1)
        kernel = kernel * valid.unsqueeze(-1) * valid.unsqueeze(-2)
        adjacency = kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if not self.use_graph:
            pair_weight = torch.exp(-energy.clamp(max=50.0) / self.graph_temperature)
            pair_weight = pair_weight * pair_valid[:, None].to(pair_weight.dtype)
            denominator = pair_weight.sum(dim=2, keepdim=True)
            fused = (pair_summary * pair_weight.unsqueeze(-1)).sum(dim=2)
            fused = fused / denominator.clamp(min=1e-8)
            fallback = _masked_mean(modality_nodes, modality_mask[:, None, :], dim=2)
            fused = torch.where(denominator > 0, fused, fallback)
            return F.normalize(fused, dim=-1), adjacency

        source_nodes = modality_nodes.unsqueeze(2).expand(-1, -1, self.num_modalities, -1, -1)
        graph_context = context[:, None, None, None, :].expand(
            -1, modality_nodes.shape[1], self.num_modalities, self.num_modalities, -1
        )
        messages = self.message_net(torch.cat([source_nodes, path_matrix, graph_context], dim=-1))
        aggregated = (adjacency.unsqueeze(-1) * messages).sum(dim=3)
        propagated = self.graph_norm(modality_nodes + aggregated)
        fused = _masked_mean(propagated, modality_mask[:, None, :], dim=2)
        return F.normalize(fused, dim=-1), adjacency

    def forward(self, modality_nodes, anchor_prototypes, modality_mask=None, return_paths=True):
        if modality_nodes.ndim != 4:
            raise ValueError("modality_nodes must have shape [B, R, M, D]")
        batch, _, modalities, dim = modality_nodes.shape
        if modalities != self.num_modalities or dim != self.shared_dim:
            raise ValueError(
                f"Expected M={self.num_modalities}, D={self.shared_dim}; got M={modalities}, D={dim}"
            )
        if anchor_prototypes is None or anchor_prototypes.ndim != 2:
            raise ValueError("anchor_prototypes must have shape [K, D]")
        if modality_mask is None:
            modality_mask = torch.ones(batch, modalities, device=modality_nodes.device, dtype=torch.bool)
        else:
            modality_mask = modality_mask.to(device=modality_nodes.device).bool()
        if (modality_mask.sum(dim=-1) == 0).any():
            raise ValueError("At least one modality must be available for every case")

        modality_nodes = F.normalize(modality_nodes, dim=-1)
        context = self._anchor_context(modality_nodes, anchor_prototypes, modality_mask)
        pair_start = modality_nodes[:, :, self.pair_indices[:, 0]]
        pair_end = modality_nodes[:, :, self.pair_indices[:, 1]]
        pair_context = context[:, None, None, :].expand_as(pair_start)
        pair_valid = modality_mask[:, self.pair_indices[:, 0]] & modality_mask[:, self.pair_indices[:, 1]]

        path_mode = "euclidean" if self.fusion_mode == "concat" else self.fusion_mode
        paths, linear_paths = self.pair_path(pair_start, pair_end, pair_context, mode=path_mode)
        supports, support_mask = self._case_supports(modality_nodes, anchor_prototypes, modality_mask)
        energy = self.path_energy(paths, supports, support_mask)
        linear_energy = self.path_energy(linear_paths, supports, support_mask)
        pair_summary = paths[..., 1:-1, :].mean(dim=-2)
        path_deviation = (paths - linear_paths).norm(dim=-1).mean(dim=-1)

        if self.fusion_mode == "concat":
            masked_nodes = modality_nodes * modality_mask[:, None, :, None].to(modality_nodes.dtype)
            fused = F.normalize(self.concat_projection(masked_nodes.flatten(start_dim=2)), dim=-1)
            adjacency = modality_nodes.new_zeros(batch, modality_nodes.shape[1], modalities, modalities)
        else:
            fused, adjacency = self._graph_fuse(
                modality_nodes, context, pair_summary, energy, pair_valid, modality_mask
            )

        if self.fusion_mode == "geodesic":
            loss_paths, loss_linear = self.pair_path(
                pair_start.detach(), pair_end.detach(), pair_context.detach(), mode="geodesic"
            )
            loss_supports, loss_support_mask = self._case_supports(
                modality_nodes.detach(), anchor_prototypes.detach(), modality_mask
            )
            loss_energy = self.path_energy(loss_paths, loss_supports, loss_support_mask)
            loss_linear_energy = self.path_energy(loss_linear, loss_supports, loss_support_mask)
            valid_weight = pair_valid[:, None].to(loss_energy.dtype).expand_as(loss_energy)
            ratio_loss = loss_energy / loss_linear_energy.detach().clamp(min=1e-8)
            geo_energy_loss = (ratio_loss * valid_weight).sum() / valid_weight.sum().clamp(min=1.0)
        else:
            geo_energy_loss = fused.sum() * 0.0

        valid_weight = pair_valid[:, None].to(energy.dtype).expand_as(energy)
        energy_ratio = energy / linear_energy.clamp(min=1e-8)
        diagnostics = {
            "geodesic_energy": (energy * valid_weight).sum() / valid_weight.sum().clamp(min=1.0),
            "linear_energy": (linear_energy * valid_weight).sum() / valid_weight.sum().clamp(min=1.0),
            "energy_ratio": (energy_ratio * valid_weight).sum() / valid_weight.sum().clamp(min=1.0),
            "path_deviation": (path_deviation * valid_weight).sum() / valid_weight.sum().clamp(min=1.0),
            "bend_scale": torch.sigmoid(self.bend_logit),
        }
        include_paths = return_paths and self.fusion_mode != "concat"
        return {
            "fused_nodes": fused,
            "modality_nodes": modality_nodes,
            "paths": paths if include_paths else None,
            "interior_paths": paths[..., 1:-1, :] if include_paths else None,
            "pair_indices": self.pair_indices,
            "pair_valid": pair_valid,
            "adjacency": adjacency,
            "geodesic_energy": energy,
            "linear_energy": linear_energy,
            "energy_ratio": energy_ratio,
            "path_deviation": path_deviation,
            "geo_energy_loss": geo_energy_loss,
            "diagnostics": diagnostics,
        }


__all__ = ["GeodesicModalityGraphFusion"]
