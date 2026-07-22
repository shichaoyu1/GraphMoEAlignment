"""Hierarchical Log-Euclidean SPD graph fusion for Paper 4."""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _spectral_map(matrix, transform, eigenvalue_min=1e-4):
    values, vectors = torch.linalg.eigh(_symmetrize(matrix))
    values = transform(values.clamp(min=eigenvalue_min))
    return _symmetrize((vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2))


def project_spd(matrix, eigenvalue_min=1e-4):
    return _spectral_map(matrix, lambda values: values, eigenvalue_min)


def spd_logm(matrix, eigenvalue_min=1e-4):
    return _spectral_map(matrix, torch.log, eigenvalue_min)


def spd_expm(matrix):
    values, vectors = torch.linalg.eigh(_symmetrize(matrix))
    values = torch.exp(values.clamp(min=-12.0, max=12.0))
    return _symmetrize((vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2))


def trace_normalize(matrix, eps=1e-8):
    dim = matrix.shape[-1]
    trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1)
    normalized = matrix / (trace / float(dim)).clamp(min=eps).unsqueeze(-1).unsqueeze(-1)
    return normalized, trace


def symmetric_vectorize(matrix):
    """Isometric half-vectorization for symmetric matrices."""
    dim = matrix.shape[-1]
    row, col = torch.triu_indices(dim, dim, device=matrix.device)
    values = matrix[..., row, col]
    scale = torch.where(row == col, 1.0, math.sqrt(2.0)).to(values.dtype)
    return values * scale


def spd_geodesic(start, end, times, eigenvalue_min=1e-4):
    """Closed-form Log-Euclidean geodesic sampled at ``times``."""
    start_log = spd_logm(start, eigenvalue_min)
    end_log = spd_logm(end, eigenvalue_min)
    times = torch.as_tensor(times, device=start.device, dtype=start.dtype)
    shape = (1,) * (start.ndim - 2) + (times.numel(), 1, 1)
    tangent_path = (
        (1.0 - times.view(shape)) * start_log.unsqueeze(-3)
        + times.view(shape) * end_log.unsqueeze(-3)
    )
    return spd_expm(tangent_path)


def _masked_adjacency(representations, bias, temperature, node_mask):
    delta = representations.unsqueeze(-3) - representations.unsqueeze(-4)
    squared_distance = delta.square().sum(dim=(-1, -2))
    distance = torch.sqrt(squared_distance + 1e-12)
    symmetric_bias = _symmetrize(bias)
    logits = symmetric_bias - squared_distance / max(float(temperature), 1e-8)
    valid_edges = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
    logits = logits.masked_fill(~valid_edges, -1e9)
    adjacency = torch.softmax(logits, dim=-1) * valid_edges.to(logits.dtype)
    adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return adjacency, distance


class HierarchicalSPDGraphFusion(nn.Module):
    """Fuse 12 region-modality nodes and global semantic-family supports."""

    def __init__(
        self,
        token_dim,
        shared_dim,
        family_ids,
        family_names,
        family_prior=None,
        spd_dim=16,
        num_regions=3,
        num_modalities=4,
        geometry="spd",
        use_upper_graph=True,
        use_anchor_families=True,
        path_steps=5,
        eigenvalue_min=1e-4,
        local_temperature=1.0,
        upper_temperature=1.0,
    ):
        super().__init__()
        if geometry not in {"spd", "euclidean"}:
            raise ValueError(f"Unsupported manifold geometry: {geometry}")
        if path_steps < 3:
            raise ValueError("SPD paths require at least three samples")
        self.shared_dim = int(shared_dim)
        self.spd_dim = int(spd_dim)
        self.num_regions = int(num_regions)
        self.num_modalities = int(num_modalities)
        self.geometry = geometry
        self.use_upper_graph = bool(use_upper_graph)
        self.use_anchor_families = bool(use_anchor_families)
        self.path_steps = int(path_steps)
        self.eigenvalue_min = float(eigenvalue_min)
        self.local_temperature = float(local_temperature)
        self.upper_temperature = float(upper_temperature)
        self.family_names = list(family_names)
        self.num_families = len(self.family_names)
        self.num_anchor_families = max(self.num_families - 1, 0)

        self.token_adapters = nn.ModuleList(
            [nn.Linear(token_dim, self.spd_dim) for _ in range(self.num_modalities)]
        )
        self.prototype_adapters = nn.ModuleList(
            [nn.Linear(shared_dim, self.spd_dim) for _ in range(self.num_anchor_families)]
        )
        self.residual_factor = nn.Parameter(torch.zeros(self.spd_dim, self.spd_dim))
        self.local_relation_bias = nn.Parameter(torch.zeros(self.num_modalities, self.num_modalities))
        upper_nodes = self.num_regions + self.num_families
        self.upper_relation_bias = nn.Parameter(torch.zeros(upper_nodes, upper_nodes))
        self.local_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.upper_mix_logit = nn.Parameter(torch.tensor(0.0))
        vector_dim = self.spd_dim * (self.spd_dim + 1) // 2
        self.readout = nn.Sequential(
            nn.Linear(vector_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )
        pairs = list(itertools.combinations(range(self.num_modalities), 2))
        self.register_buffer("pair_indices", torch.tensor(pairs, dtype=torch.long), persistent=False)
        self.register_buffer("path_times", torch.linspace(0.0, 1.0, self.path_steps), persistent=False)
        self.register_buffer("family_ids", torch.as_tensor(family_ids, dtype=torch.long))
        if family_prior is None:
            family_prior = torch.eye(max(self.num_families, 1), dtype=torch.float32)
        self.register_buffer("family_prior", torch.as_tensor(family_prior, dtype=torch.float32))

    def _token_spd(self, tokens):
        matrices = []
        raw_scales = []
        raw_traces = []
        for modality, adapter in enumerate(self.token_adapters):
            values = tokens[:, :, modality]
            raw_scales.append(values.square().mean(dim=(-1, -2)).sqrt())
            projected = adapter(values)
            centered = projected - projected.mean(dim=-2, keepdim=True)
            denominator = max(projected.shape[-2] - 1, 1)
            covariance = centered.transpose(-1, -2) @ centered / float(denominator)
            jitter = torch.diag(
                torch.linspace(1.0, 2.0, self.spd_dim, device=tokens.device, dtype=tokens.dtype)
            )
            covariance = _symmetrize(covariance + self.eigenvalue_min * jitter)
            covariance, trace = trace_normalize(covariance)
            matrices.append(covariance)
            raw_traces.append(trace)
        return (
            torch.stack(matrices, dim=2),
            torch.stack(raw_scales, dim=2),
            torch.stack(raw_traces, dim=2),
        )

    def _family_spd(self, prototypes):
        matrices = []
        for family, adapter in enumerate(self.prototype_adapters):
            indices = torch.where(self.family_ids == family)[0]
            if indices.numel() == 0:
                continue
            values = adapter(prototypes[indices])
            second_moment = values.transpose(0, 1) @ values / float(max(values.shape[0], 1))
            jitter = torch.diag(
                torch.linspace(1.0, 2.0, self.spd_dim, device=prototypes.device, dtype=prototypes.dtype)
            )
            matrix = _symmetrize(second_moment + self.eigenvalue_min * jitter)
            matrices.append(trace_normalize(matrix)[0])
        factor = torch.tril(self.residual_factor)
        diagonal = F.softplus(torch.diagonal(factor)) + 0.1
        factor = factor - torch.diag(torch.diagonal(factor)) + torch.diag(diagonal)
        jitter = torch.diag(
            torch.linspace(1.0, 2.0, self.spd_dim, device=prototypes.device, dtype=prototypes.dtype)
        )
        residual = factor @ factor.t() + self.eigenvalue_min * jitter
        matrices.append(trace_normalize(residual)[0])
        return torch.stack(matrices, dim=0)

    def _representation(self, matrices):
        return spd_logm(matrices, self.eigenvalue_min) if self.geometry == "spd" else matrices

    def _readout(self, representation):
        return F.normalize(self.readout(symmetric_vectorize(representation)), dim=-1)

    def _local_paths(self, local_representation, pair_valid):
        start = local_representation[:, :, self.pair_indices[:, 0]]
        end = local_representation[:, :, self.pair_indices[:, 1]]
        times = self.path_times.to(dtype=start.dtype, device=start.device)
        path = (
            (1.0 - times.view(1, 1, 1, -1, 1, 1)) * start.unsqueeze(-3)
            + times.view(1, 1, 1, -1, 1, 1) * end.unsqueeze(-3)
        )
        interior = path[..., 1:-1, :, :]
        embeddings = self._readout(interior)
        return embeddings, pair_valid

    def _upper_prior_logits(self, node_count, device, dtype):
        logits = torch.zeros(node_count, node_count, device=device, dtype=dtype)
        if not self.use_anchor_families or self.num_families == 0 or node_count <= self.num_regions:
            return logits
        start = self.num_regions
        prior = self.family_prior[: self.num_families, : self.num_families].to(device=device, dtype=dtype)
        logits[start:, start:] = torch.log(prior + 1e-4)
        return logits

    def forward(self, tokens, anchor_prototypes, modality_mask=None):
        if tokens.ndim != 5:
            raise ValueError("tokens must have shape [B, R, M, T, C]")
        batch, regions, modalities = tokens.shape[:3]
        if regions != self.num_regions or modalities != self.num_modalities:
            raise ValueError("Unexpected region or modality count")
        if modality_mask is None:
            modality_mask = torch.ones(batch, modalities, device=tokens.device, dtype=torch.bool)
        else:
            modality_mask = modality_mask.to(device=tokens.device).bool()
        if (modality_mask.sum(dim=-1) == 0).any():
            raise ValueError("At least one modality must be available for every case")

        matrices, raw_scales, raw_traces = self._token_spd(tokens)
        eigenvalues = torch.linalg.eigvalsh(matrices)
        condition_numbers = eigenvalues[..., -1] / eigenvalues[..., 0].clamp(min=self.eigenvalue_min)
        local_representation = self._representation(matrices)
        local_mask = modality_mask[:, None, :].expand(-1, regions, -1)
        local_bias = self.local_relation_bias[None, None]
        local_adjacency, local_distances = _masked_adjacency(
            local_representation,
            local_bias,
            self.local_temperature,
            local_mask,
        )
        local_mix = torch.sigmoid(self.local_mix_logit)
        local_messages = torch.einsum("brmn,brnij->brmij", local_adjacency, local_representation)
        local_updated = (1.0 - local_mix) * local_representation + local_mix * local_messages
        centrality = local_adjacency.sum(dim=-2) * local_mask.to(local_adjacency.dtype)
        centrality = centrality / centrality.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        region_representation = torch.einsum("brm,brmij->brij", centrality, local_updated)

        upper_nodes = region_representation
        upper_names = ["Necrotic/Core", "Edema", "Enhancing"]
        active_anchor_families = self.use_anchor_families and self.use_upper_graph
        if active_anchor_families:
            family_matrices = self._family_spd(anchor_prototypes)
            family_representation = self._representation(family_matrices)
            upper_nodes = torch.cat(
                [upper_nodes, family_representation[None].expand(batch, -1, -1, -1)], dim=1
            )
            upper_names.extend(self.family_names)
        node_count = upper_nodes.shape[1]
        upper_mask = torch.ones(batch, node_count, device=tokens.device, dtype=torch.bool)
        upper_bias = self.upper_relation_bias[:node_count, :node_count]
        upper_bias = upper_bias + self._upper_prior_logits(node_count, tokens.device, tokens.dtype)
        upper_adjacency, upper_distances = _masked_adjacency(
            upper_nodes,
            upper_bias[None],
            self.upper_temperature,
            upper_mask,
        )
        if self.use_upper_graph:
            upper_mix = torch.sigmoid(self.upper_mix_logit)
            upper_messages = torch.einsum("bmn,bnij->bmij", upper_adjacency, upper_nodes)
            upper_nodes = (1.0 - upper_mix) * upper_nodes + upper_mix * upper_messages
        else:
            upper_adjacency = torch.eye(node_count, device=tokens.device, dtype=tokens.dtype)[None].expand(batch, -1, -1)
        final_regions = upper_nodes[:, : self.num_regions]
        fused = self._readout(final_regions)

        pair_valid = modality_mask[:, self.pair_indices[:, 0]] & modality_mask[:, self.pair_indices[:, 1]]
        path_embeddings, pair_valid = self._local_paths(local_representation, pair_valid)
        log_condition = torch.log(condition_numbers.clamp(min=1.0))
        condition_loss = F.relu(log_condition - math.log(1e3)).square().mean()
        topology_loss = fused.sum() * 0.0
        if active_anchor_families and self.num_families > 1:
            start = self.num_regions
            learned = upper_adjacency[:, start:, start:].mean(dim=0)
            learned = learned / learned.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            prior = self.family_prior[: self.num_families, : self.num_families]
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            topology_loss = (learned - prior).square().mean()

        diagnostics = {
            "raw_scale": raw_scales.mean(),
            "raw_spd_trace": raw_traces.mean(),
            "normalized_spd_trace": torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1).mean(),
            "condition_number": condition_numbers.mean(),
            "local_distance": local_distances.mean(),
            "upper_distance": upper_distances.mean(),
            "local_edge_entropy": -(local_adjacency * torch.log(local_adjacency.clamp(min=1e-8))).sum(dim=-1).mean(),
            "upper_edge_entropy": -(upper_adjacency * torch.log(upper_adjacency.clamp(min=1e-8))).sum(dim=-1).mean(),
        }
        return {
            "fused_nodes": fused,
            "local_adjacency": local_adjacency,
            "upper_adjacency": upper_adjacency,
            "local_distances": local_distances,
            "upper_distances": upper_distances,
            "raw_scales": raw_scales,
            "raw_spd_traces": raw_traces,
            "spd_eigenvalues": eigenvalues,
            "condition_numbers": condition_numbers,
            "interior_path_embeddings": path_embeddings,
            "pair_indices": self.pair_indices,
            "pair_valid": pair_valid,
            "upper_node_names": upper_names,
            "condition_loss": condition_loss,
            "topology_loss": topology_loss,
            "diagnostics": diagnostics,
        }


__all__ = [
    "HierarchicalSPDGraphFusion",
    "project_spd",
    "spd_logm",
    "spd_expm",
    "trace_normalize",
    "symmetric_vectorize",
    "spd_geodesic",
]
