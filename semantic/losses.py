"""Semantic-alignment loss functions."""

from glioma.objectives import (
    SemanticPrototypeBank,
    dcca_alignment_loss,
    masked_multi_positive_nll,
    medclip_multi_positive_loss,
    multi_positive_contrastive_loss,
)
import torch
import torch.nn.functional as F


def anchor_center_loss(queries, target_ids, prototypes):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    losses = []
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        center = prototypes[ids].mean(dim=0, keepdim=True)
        center = F.normalize(center, dim=-1).squeeze(0)
        losses.append(1.0 - torch.sum(queries[row] * center))
    if not losses:
        return queries.sum() * 0
    return torch.stack(losses).mean()


def topomoe_family_balanced_losses(
    routing_weights,
    family_log_probs,
    target_ids,
    anchor_family_ids,
    residual_index,
):
    """Equal-family router supervision plus within-family multi-positive NLL."""
    if routing_weights is None or family_log_probs is None:
        raise ValueError("TopoMoE v2 losses require routing weights and family log probabilities")
    routing = routing_weights.reshape(-1, routing_weights.shape[-1])
    conditional = family_log_probs.reshape(-1, family_log_probs.shape[-1])
    family_ids = torch.as_tensor(anchor_family_ids, device=routing.device, dtype=torch.long)
    if len(target_ids) != routing.shape[0]:
        raise ValueError(f"Expected {routing.shape[0]} target rows, got {len(target_ids)}")

    family_route_losses = []
    within_anchor_losses = []
    for row, positive_ids in enumerate(target_ids):
        positive_ids = [int(anchor_id) for anchor_id in positive_ids if 0 <= int(anchor_id) < len(family_ids)]
        if not positive_ids:
            continue
        positive_families = sorted(
            {
                int(family_ids[anchor_id].item())
                for anchor_id in positive_ids
                if int(family_ids[anchor_id].item()) < int(residual_index)
            }
        )
        if not positive_families:
            continue

        family_target = torch.zeros_like(routing[row])
        family_target[positive_families] = 1.0 / len(positive_families)
        family_route_losses.append(
            -(family_target * torch.log(routing[row].clamp(min=1e-8))).sum()
        )

        per_family = []
        for family in positive_families:
            family_positives = [
                anchor_id
                for anchor_id in positive_ids
                if int(family_ids[anchor_id].item()) == family
            ]
            per_family.append(-torch.logsumexp(conditional[row, family_positives], dim=0))
        within_anchor_losses.append(torch.stack(per_family).mean())

    zero = routing.sum() * 0
    family_route = torch.stack(family_route_losses).mean() if family_route_losses else zero
    within_anchor = torch.stack(within_anchor_losses).mean() if within_anchor_losses else zero
    return family_route, within_anchor


def geodesic_path_semantic_loss(interior_paths, pair_valid, target_ids, prototypes, temperature=0.07):
    """Align every valid interior path point with its region-level targets."""
    if interior_paths is None or interior_paths.numel() == 0:
        return prototypes.sum() * 0
    batch, regions, pairs, steps, dim = interior_paths.shape
    if len(target_ids) != batch * regions:
        raise ValueError(f"Expected {batch * regions} target rows, got {len(target_ids)}")
    queries = []
    path_targets = []
    for batch_idx in range(batch):
        for region_idx in range(regions):
            targets = target_ids[batch_idx * regions + region_idx]
            for pair_idx in range(pairs):
                if not bool(pair_valid[batch_idx, pair_idx]):
                    continue
                queries.append(interior_paths[batch_idx, region_idx, pair_idx])
                path_targets.extend([targets] * steps)
    if not queries:
        return interior_paths.sum() * 0
    query_tensor = torch.cat(queries, dim=0).reshape(-1, dim)
    return multi_positive_contrastive_loss(
        query_tensor,
        path_targets,
        prototypes,
        temperature=temperature,
    )

__all__ = [
    "SemanticPrototypeBank",
    "multi_positive_contrastive_loss",
    "masked_multi_positive_nll",
    "medclip_multi_positive_loss",
    "dcca_alignment_loss",
    "anchor_center_loss",
    "topomoe_family_balanced_losses",
    "geodesic_path_semantic_loss",
]
