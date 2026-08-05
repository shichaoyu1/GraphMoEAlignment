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


def token_spd_matrices(tokens, adapters, eigenvalue_min=1e-4, token_mask=None):
    """Build trace-normalized covariance descriptors from padded token sequences."""
    if tokens.ndim != 5:
        raise ValueError("tokens must have shape [B, G, M, T, C]")
    if len(adapters) != tokens.shape[2]:
        raise ValueError("Adapter count must match the modality dimension")
    if token_mask is None:
        token_mask = torch.ones(tokens.shape[:-1], device=tokens.device, dtype=torch.bool)
    else:
        if tuple(token_mask.shape) != tuple(tokens.shape[:-1]):
            raise ValueError("token_mask must have shape [B, G, M, T]")
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)

    matrices = []
    raw_scales = []
    raw_traces = []
    for modality, adapter in enumerate(adapters):
        values = tokens[:, :, modality]
        valid = token_mask[:, :, modality].unsqueeze(-1).to(values.dtype)
        count = valid.sum(dim=-2).clamp(min=1.0)
        scale_denominator = (count * values.shape[-1]).clamp(min=1.0)
        raw_scales.append(
            ((values.square() * valid).sum(dim=(-1, -2)) / scale_denominator.squeeze(-1)).sqrt()
        )
        projected = adapter(values)
        mean = (projected * valid).sum(dim=-2, keepdim=True) / count.unsqueeze(-2)
        centered = (projected - mean) * valid
        denominator = (count - 1.0).clamp(min=1.0).unsqueeze(-1)
        covariance = centered.transpose(-1, -2) @ centered / denominator
        jitter = torch.diag(
            torch.linspace(
                1.0,
                2.0,
                covariance.shape[-1],
                device=tokens.device,
                dtype=tokens.dtype,
            )
        )
        covariance = _symmetrize(covariance + float(eigenvalue_min) * jitter)
        covariance, trace = trace_normalize(covariance)
        matrices.append(covariance)
        raw_traces.append(trace)
    return (
        torch.stack(matrices, dim=2),
        torch.stack(raw_scales, dim=2),
        torch.stack(raw_traces, dim=2),
    )


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
    """Legacy adjacency retained for checkpoint-compatible reproduction."""
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


def _identity_adjacency(node_mask, dtype):
    count = node_mask.shape[-1]
    identity = torch.eye(count, device=node_mask.device, dtype=dtype)
    prefix = (1,) * (node_mask.ndim - 1)
    adjacency = identity.view(*prefix, count, count) * node_mask.unsqueeze(-1).to(dtype)
    return adjacency


def _normalize_group(logits, mask, uniform=False):
    if uniform:
        weights = mask.to(logits.dtype)
    else:
        weights = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1) * mask.to(logits.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _rotate_valid_weights(weights, mask):
    """Deterministically move each valid edge weight to the next valid target."""
    shuffled = torch.flip(weights, dims=(-1,)) * mask.to(weights.dtype)
    total = shuffled.sum(dim=-1, keepdim=True)
    normalized = shuffled / total.clamp(min=1e-8)
    return torch.where(total > 0, normalized, weights)


def _cross_budget_adjacency(
    representations,
    bias,
    temperature,
    node_mask,
    cross_mass,
    uniform=False,
    shuffle=False,
):
    delta = representations.unsqueeze(-3) - representations.unsqueeze(-4)
    squared_distance = delta.square().sum(dim=(-1, -2))
    distance = torch.sqrt(squared_distance + 1e-12)
    symmetric_bias = _symmetrize(bias)
    logits = symmetric_bias - squared_distance / max(float(temperature), 1e-8)
    count = logits.shape[-1]
    identity_mask = torch.eye(count, device=logits.device, dtype=torch.bool)
    prefix = (1,) * (node_mask.ndim - 1)
    identity_mask = identity_mask.view(*prefix, count, count)
    valid_edges = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
    cross_mask = valid_edges & ~identity_mask
    cross = _normalize_group(logits, cross_mask, uniform=uniform)
    if shuffle:
        cross = _rotate_valid_weights(cross, cross_mask)
    has_cross = cross_mask.any(dim=-1, keepdim=True)
    allocated = torch.where(
        has_cross,
        torch.as_tensor(float(cross_mass), device=logits.device, dtype=logits.dtype),
        torch.zeros((), device=logits.device, dtype=logits.dtype),
    )
    identity = _identity_adjacency(node_mask, logits.dtype)
    adjacency = (1.0 - allocated) * identity + allocated * cross
    adjacency = adjacency * valid_edges.to(logits.dtype)
    adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return adjacency, distance


def _typed_cross_budget_adjacency(
    representations,
    bias,
    temperature,
    node_mask,
    num_regions,
    cross_mass,
    region_family_fraction,
    uniform=False,
    shuffle=False,
    remove_cross_type=False,
):
    delta = representations.unsqueeze(-3) - representations.unsqueeze(-4)
    squared_distance = delta.square().sum(dim=(-1, -2))
    distance = torch.sqrt(squared_distance + 1e-12)
    logits = _symmetrize(bias) - squared_distance / max(float(temperature), 1e-8)
    count = logits.shape[-1]
    identity_mask = torch.eye(count, device=logits.device, dtype=torch.bool)
    identity_mask = identity_mask.view(*((1,) * (node_mask.ndim - 1)), count, count)
    valid = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
    source_region = torch.arange(count, device=logits.device) < int(num_regions)
    target_region = source_region.view(*((1,) * (node_mask.ndim - 1)), 1, count)
    source_region = source_region.view(*((1,) * (node_mask.ndim - 1)), count, 1)
    same_type = source_region == target_region
    same_mask = valid & ~identity_mask & same_type
    cross_type_mask = valid & ~identity_mask & ~same_type
    if remove_cross_type:
        cross_type_mask = torch.zeros_like(cross_type_mask)

    same = _normalize_group(logits, same_mask, uniform=uniform)
    cross_type = _normalize_group(logits, cross_type_mask, uniform=uniform)
    if shuffle:
        same = _rotate_valid_weights(same, same_mask)
        cross_type = _rotate_valid_weights(cross_type, cross_type_mask)

    region_fraction = float(region_family_fraction)
    cross_type_fraction = torch.where(
        source_region,
        torch.as_tensor(region_fraction, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(region_fraction, device=logits.device, dtype=logits.dtype),
    )
    same_available = same_mask.any(dim=-1, keepdim=True)
    cross_available = cross_type_mask.any(dim=-1, keepdim=True)
    same_share = (1.0 - cross_type_fraction) * same_available.to(logits.dtype)
    cross_share = cross_type_fraction * cross_available.to(logits.dtype)
    share_total = same_share + cross_share
    same_share = torch.where(share_total > 0, same_share / share_total.clamp(min=1e-8), same_share)
    cross_share = torch.where(share_total > 0, cross_share / share_total.clamp(min=1e-8), cross_share)
    allocated = float(cross_mass) * (share_total > 0).to(logits.dtype)
    identity = _identity_adjacency(node_mask, logits.dtype)
    adjacency = (1.0 - allocated) * identity + allocated * (same_share * same + cross_share * cross_type)
    adjacency = adjacency * valid.to(logits.dtype)
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
        graph_policy="cross_budget",
        local_cross_mass=0.35,
        upper_cross_mass=0.40,
        region_family_fraction=0.625,
        local_topology="learned",
        upper_topology="learned",
        graph_intervention="none",
    ):
        super().__init__()
        if geometry not in {"spd", "euclidean"}:
            raise ValueError(f"Unsupported manifold geometry: {geometry}")
        if path_steps < 3:
            raise ValueError("SPD paths require at least three samples")
        if graph_policy not in {"legacy_softmax", "cross_budget"}:
            raise ValueError(f"Unsupported SPD graph policy: {graph_policy}")
        for name, value in {
            "local_topology": local_topology,
            "upper_topology": upper_topology,
        }.items():
            if value not in {"learned", "identity", "uniform"}:
                raise ValueError(f"Unsupported {name}: {value}")
        if graph_intervention not in {
            "none", "identity", "uniform", "shuffle", "no_local", "no_upper", "no_region_family"
        }:
            raise ValueError(f"Unsupported Paper 4 graph intervention: {graph_intervention}")
        for name, value in {
            "local_cross_mass": local_cross_mass,
            "upper_cross_mass": upper_cross_mass,
            "region_family_fraction": region_family_fraction,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
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
        self.graph_policy = graph_policy
        self.local_cross_mass = float(local_cross_mass)
        self.upper_cross_mass = float(upper_cross_mass)
        self.region_family_fraction = float(region_family_fraction)
        self.local_topology = local_topology
        self.upper_topology = upper_topology
        self.graph_intervention = graph_intervention
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

    def _token_spd(self, tokens, token_mask=None):
        return token_spd_matrices(
            tokens,
            self.token_adapters,
            eigenvalue_min=self.eigenvalue_min,
            token_mask=token_mask,
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

    def forward(
        self,
        tokens,
        anchor_prototypes,
        modality_mask=None,
        token_mask=None,
        graph_intervention=None,
    ):
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

        intervention = graph_intervention
        if intervention is None:
            intervention = self.graph_intervention if not self.training else "none"
        if intervention not in {
            "none", "identity", "uniform", "shuffle", "no_local", "no_upper", "no_region_family"
        }:
            raise ValueError(f"Unsupported Paper 4 graph intervention: {intervention}")

        matrices, raw_scales, raw_traces = self._token_spd(tokens, token_mask=token_mask)
        eigenvalues = torch.linalg.eigvalsh(matrices)
        condition_numbers = eigenvalues[..., -1] / eigenvalues[..., 0].clamp(min=self.eigenvalue_min)
        local_representation = self._representation(matrices)
        local_mask = modality_mask[:, None, :].expand(-1, regions, -1)
        if token_mask is not None:
            local_mask = local_mask & token_mask.to(device=tokens.device, dtype=torch.bool).any(dim=-1)
        if (local_mask.sum(dim=-1) == 0).any():
            raise ValueError("Every group must contain at least one modality with a valid token")
        local_bias = self.local_relation_bias[None, None]
        local_topology = self.local_topology
        if intervention in {"identity", "no_local"}:
            local_topology = "identity"
        elif intervention == "uniform":
            local_topology = "uniform"
        local_identity = local_topology == "identity"
        if self.graph_policy == "legacy_softmax" and local_topology == "learned":
            local_adjacency, local_distances = _masked_adjacency(
                local_representation, local_bias, self.local_temperature, local_mask
            )
        elif local_identity:
            local_adjacency = _identity_adjacency(local_mask, local_representation.dtype)
            delta = local_representation.unsqueeze(-3) - local_representation.unsqueeze(-4)
            local_distances = torch.sqrt(delta.square().sum(dim=(-1, -2)) + 1e-12)
        else:
            local_adjacency, local_distances = _cross_budget_adjacency(
                local_representation,
                local_bias,
                self.local_temperature,
                local_mask,
                self.local_cross_mass,
                uniform=local_topology == "uniform",
                shuffle=intervention == "shuffle",
            )
        local_messages = torch.einsum("brmn,brnij->brmij", local_adjacency, local_representation)
        if self.graph_policy == "legacy_softmax":
            local_mix = torch.sigmoid(self.local_mix_logit)
            local_updated = (1.0 - local_mix) * local_representation + local_mix * local_messages
        else:
            local_updated = local_messages
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
        upper_topology = self.upper_topology
        if intervention in {"identity", "no_upper"}:
            upper_topology = "identity"
        elif intervention == "uniform":
            upper_topology = "uniform"
        upper_identity = upper_topology == "identity" or not self.use_upper_graph
        if self.graph_policy == "legacy_softmax" and upper_topology == "learned" and not upper_identity:
            upper_adjacency, upper_distances = _masked_adjacency(
                upper_nodes, upper_bias[None], self.upper_temperature, upper_mask
            )
        elif upper_identity:
            upper_adjacency = _identity_adjacency(upper_mask, upper_nodes.dtype)
            delta = upper_nodes.unsqueeze(-3) - upper_nodes.unsqueeze(-4)
            upper_distances = torch.sqrt(delta.square().sum(dim=(-1, -2)) + 1e-12)
        else:
            upper_adjacency, upper_distances = _typed_cross_budget_adjacency(
                upper_nodes,
                upper_bias[None],
                self.upper_temperature,
                upper_mask,
                self.num_regions,
                self.upper_cross_mass,
                self.region_family_fraction,
                uniform=upper_topology == "uniform",
                shuffle=intervention == "shuffle",
                remove_cross_type=intervention == "no_region_family",
            )
        upper_before = upper_nodes
        if self.use_upper_graph and not upper_identity:
            upper_messages = torch.einsum("bmn,bnij->bmij", upper_adjacency, upper_nodes)
            if self.graph_policy == "legacy_softmax":
                upper_mix = torch.sigmoid(self.upper_mix_logit)
                upper_nodes = (1.0 - upper_mix) * upper_nodes + upper_mix * upper_messages
            else:
                upper_nodes = upper_messages
        final_regions = upper_nodes[:, : self.num_regions]
        fused = self._readout(final_regions)

        identity_weight = local_mask.to(local_representation.dtype)
        identity_region = (local_representation * identity_weight[..., None, None]).sum(dim=2)
        identity_region = identity_region / identity_weight.sum(dim=2).clamp(min=1.0)[..., None, None]
        identity_fused = self._readout(identity_region)
        local_update_ratio = (local_updated - local_representation).flatten(-2).norm(dim=-1)
        local_update_ratio = local_update_ratio / local_representation.flatten(-2).norm(dim=-1).clamp(min=1e-8)
        upper_update_ratio = (upper_nodes - upper_before).flatten(-2).norm(dim=-1)
        upper_update_ratio = upper_update_ratio / upper_before.flatten(-2).norm(dim=-1).clamp(min=1e-8)
        fused_l2_shift = (fused - identity_fused).norm(dim=-1)
        fused_cosine_shift = 1.0 - F.cosine_similarity(fused, identity_fused, dim=-1)

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
            "local_offdiagonal_mass": (
                local_adjacency * (1.0 - torch.eye(modalities, device=tokens.device, dtype=tokens.dtype))
            ).sum(dim=-1).mean(),
            "upper_offdiagonal_mass": (
                upper_adjacency * (1.0 - torch.eye(node_count, device=tokens.device, dtype=tokens.dtype))
            ).sum(dim=-1).mean(),
            "region_family_mass": (
                upper_adjacency[:, : self.num_regions, self.num_regions :].sum(dim=-1).mean()
                if node_count > self.num_regions
                else fused.sum() * 0.0
            ),
            "local_update_ratio": local_update_ratio.mean(),
            "upper_update_ratio": upper_update_ratio.mean(),
            "identity_l2_shift": fused_l2_shift.mean(),
            "identity_cosine_shift": fused_cosine_shift.mean(),
        }
        return {
            "fused_nodes": fused,
            "identity_fused_nodes": identity_fused,
            "local_adjacency": local_adjacency,
            "upper_adjacency": upper_adjacency,
            "local_distances": local_distances,
            "upper_distances": upper_distances,
            "raw_scales": raw_scales,
            "raw_spd_traces": raw_traces,
            "spd_eigenvalues": eigenvalues,
            "condition_numbers": condition_numbers,
            "local_update_ratio": local_update_ratio,
            "upper_update_ratio": upper_update_ratio,
            "identity_l2_shift": fused_l2_shift,
            "identity_cosine_shift": fused_cosine_shift,
            "interior_path_embeddings": path_embeddings,
            "pair_indices": self.pair_indices,
            "pair_valid": pair_valid,
            "upper_node_names": upper_names,
            "condition_loss": condition_loss,
            "topology_loss": topology_loss,
            "diagnostics": diagnostics,
            "graph_intervention": intervention,
            "graph_policy": self.graph_policy,
            "local_topology": local_topology,
            "upper_topology": upper_topology,
        }


class SPDMomentGraphFusion(nn.Module):
    """Task-agnostic SPD fusion for tokenized multimodal benchmarks."""

    def __init__(
        self,
        token_dim,
        shared_dim,
        num_modalities,
        spd_dim=16,
        num_groups=1,
        num_supports=0,
        geometry="spd",
        local_topology="learned",
        upper_topology="identity",
        local_cross_mass=0.35,
        upper_cross_mass=0.40,
        region_family_fraction=0.625,
        eigenvalue_min=1e-4,
        local_temperature=1.0,
        upper_temperature=1.0,
    ):
        super().__init__()
        if geometry not in {"spd", "euclidean"}:
            raise ValueError(f"Unsupported manifold geometry: {geometry}")
        for name, value in {
            "local_topology": local_topology,
            "upper_topology": upper_topology,
        }.items():
            if value not in {"learned", "identity", "uniform"}:
                raise ValueError(f"Unsupported {name}: {value}")
        self.num_modalities = int(num_modalities)
        self.num_groups = int(num_groups)
        self.num_supports = int(num_supports)
        self.spd_dim = int(spd_dim)
        self.geometry = geometry
        self.local_topology = local_topology
        self.upper_topology = upper_topology
        self.local_cross_mass = float(local_cross_mass)
        self.upper_cross_mass = float(upper_cross_mass)
        self.region_family_fraction = float(region_family_fraction)
        self.eigenvalue_min = float(eigenvalue_min)
        self.local_temperature = float(local_temperature)
        self.upper_temperature = float(upper_temperature)
        self.token_adapters = nn.ModuleList(
            [nn.Linear(token_dim, self.spd_dim) for _ in range(self.num_modalities)]
        )
        self.local_relation_bias = nn.Parameter(
            torch.zeros(self.num_modalities, self.num_modalities)
        )
        upper_count = self.num_groups + self.num_supports
        self.upper_relation_bias = nn.Parameter(torch.zeros(upper_count, upper_count))
        vector_dim = self.spd_dim * (self.spd_dim + 1) // 2
        self.readout = nn.Sequential(
            nn.Linear(vector_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.SiLU(),
        )

    def _representation(self, matrices):
        if self.geometry == "spd":
            return spd_logm(matrices, self.eigenvalue_min)
        return matrices

    @staticmethod
    def _resolve_topology(base, override):
        return base if override is None else override

    def _local_adjacency(self, representation, mask, topology):
        if topology == "identity":
            adjacency = _identity_adjacency(mask, representation.dtype)
            delta = representation.unsqueeze(-3) - representation.unsqueeze(-4)
            distances = torch.sqrt(delta.square().sum(dim=(-1, -2)) + 1e-12)
            return adjacency, distances
        return _cross_budget_adjacency(
            representation,
            self.local_relation_bias[None, None],
            self.local_temperature,
            mask,
            self.local_cross_mass,
            uniform=topology == "uniform",
        )

    def forward(
        self,
        tokens,
        modality_mask=None,
        token_mask=None,
        support_nodes=None,
        local_topology_override=None,
        upper_topology_override=None,
    ):
        if tokens.ndim != 5:
            raise ValueError("tokens must have shape [B, G, M, T, C]")
        batch, groups, modalities = tokens.shape[:3]
        if groups != self.num_groups or modalities != self.num_modalities:
            raise ValueError("Unexpected group or modality count")
        if modality_mask is None:
            modality_mask = torch.ones(batch, modalities, device=tokens.device, dtype=torch.bool)
        else:
            modality_mask = modality_mask.to(device=tokens.device, dtype=torch.bool)
        matrices, raw_scales, raw_traces = token_spd_matrices(
            tokens,
            self.token_adapters,
            eigenvalue_min=self.eigenvalue_min,
            token_mask=token_mask,
        )
        representation = self._representation(matrices)
        local_mask = modality_mask[:, None, :].expand(-1, groups, -1)
        if token_mask is not None:
            local_mask = local_mask & token_mask.to(device=tokens.device, dtype=torch.bool).any(dim=-1)
        if (local_mask.sum(dim=-1) == 0).any():
            raise ValueError("Every group must contain at least one available modality")

        local_topology = self._resolve_topology(self.local_topology, local_topology_override)
        if local_topology not in {"learned", "identity", "uniform"}:
            raise ValueError(f"Unsupported local topology override: {local_topology}")
        local_adjacency, local_distances = self._local_adjacency(
            representation, local_mask, local_topology
        )
        local_messages = torch.einsum("bgmn,bgnij->bgmij", local_adjacency, representation)
        centrality = local_adjacency.sum(dim=-2) * local_mask.to(local_adjacency.dtype)
        centrality = centrality / centrality.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        group_representation = torch.einsum("bgm,bgmij->bgij", centrality, local_messages)

        upper_topology = self._resolve_topology(self.upper_topology, upper_topology_override)
        if upper_topology not in {"learned", "identity", "uniform"}:
            raise ValueError(f"Unsupported upper topology override: {upper_topology}")
        upper_nodes = group_representation
        if self.num_supports:
            if support_nodes is None:
                raise ValueError("support_nodes are required when num_supports is non-zero")
            if support_nodes.ndim == 3:
                support_nodes = support_nodes.unsqueeze(0).expand(batch, -1, -1, -1)
            expected = (batch, self.num_supports, self.spd_dim, self.spd_dim)
            if tuple(support_nodes.shape) != expected:
                raise ValueError(f"support_nodes must have shape {expected}")
            upper_nodes = torch.cat([upper_nodes, self._representation(support_nodes)], dim=1)
        upper_mask = torch.ones(batch, upper_nodes.shape[1], device=tokens.device, dtype=torch.bool)
        if upper_topology == "identity" or upper_nodes.shape[1] == 1:
            upper_adjacency = _identity_adjacency(upper_mask, upper_nodes.dtype)
            delta = upper_nodes.unsqueeze(-3) - upper_nodes.unsqueeze(-4)
            upper_distances = torch.sqrt(delta.square().sum(dim=(-1, -2)) + 1e-12)
            upper_updated = upper_nodes
        else:
            upper_adjacency, upper_distances = _typed_cross_budget_adjacency(
                upper_nodes,
                self.upper_relation_bias[None],
                self.upper_temperature,
                upper_mask,
                self.num_groups,
                self.upper_cross_mass,
                self.region_family_fraction,
                uniform=upper_topology == "uniform",
            )
            upper_updated = torch.einsum("bmn,bnij->bmij", upper_adjacency, upper_nodes)
        final_groups = upper_updated[:, : self.num_groups]
        fused = F.normalize(self.readout(symmetric_vectorize(final_groups)), dim=-1)
        return {
            "fused_nodes": fused,
            "local_adjacency": local_adjacency,
            "upper_adjacency": upper_adjacency,
            "local_distances": local_distances,
            "upper_distances": upper_distances,
            "spd_matrices": matrices,
            "local_representation": representation,
            "group_representation": group_representation,
            "raw_scales": raw_scales,
            "raw_spd_traces": raw_traces,
            "local_topology": local_topology,
            "upper_topology": upper_topology,
            "diagnostics": {
                "local_edge_entropy": -(
                    local_adjacency * torch.log(local_adjacency.clamp(min=1e-8))
                ).sum(dim=-1).mean(),
                "local_offdiagonal_mass": (
                    local_adjacency
                    * (1.0 - torch.eye(modalities, device=tokens.device, dtype=tokens.dtype))
                ).sum(dim=-1).mean(),
            },
        }


__all__ = [
    "HierarchicalSPDGraphFusion",
    "SPDMomentGraphFusion",
    "project_spd",
    "spd_logm",
    "spd_expm",
    "trace_normalize",
    "symmetric_vectorize",
    "token_spd_matrices",
    "spd_geodesic",
]
