"""Explicit candidate masks and published proxy-loss adaptations."""

import torch
import torch.nn.functional as F


def target_masks(targets, n_anchors, device, candidate_policy="full_gallery"):
    if candidate_policy not in {"full_gallery", "observed_fields"}:
        raise ValueError("Unsupported candidate policy")
    positives = torch.zeros(len(targets), n_anchors, device=device, dtype=torch.bool)
    candidates = torch.ones_like(positives) if candidate_policy == "full_gallery" else torch.zeros_like(positives)
    for row, target in enumerate(targets):
        positives[row, list(target.positives)] = True
        if candidate_policy == "observed_fields":
            candidates[row, list(target.observed_candidates)] = True
    if (positives & ~candidates).any():
        raise ValueError("Positive anchors must belong to the candidate set")
    return positives, candidates


def retrieval_loss(logits, positive_mask, candidate_mask, mean_positive=False):
    valid = positive_mask.any(-1)
    if not valid.any():
        return logits.sum() * 0
    logits = logits[valid]
    positives, candidates = positive_mask[valid], candidate_mask[valid]
    normalizer = torch.logsumexp(logits.masked_fill(~candidates, -torch.inf), dim=-1)
    if mean_positive:
        # SupCon's mean-positive log-probability, adapted to label proxies.
        positive_term = logits.masked_fill(~positives, 0).sum(-1) / positives.sum(-1)
    else:
        positive_term = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=-1)
    return (normalizer - positive_term).mean()


def proxy_anchor_loss(cosine, positive_mask, candidate_mask, alpha=32.0, margin=0.1):
    """Kim et al., CVPR2020, Eq.2; multi-hot labels and candidate masks are adaptations."""
    valid = positive_mask.any(-1)
    if not valid.any():
        return cosine.sum() * 0
    cosine = cosine[valid]
    positives = positive_mask[valid]
    negatives = candidate_mask[valid] & ~positives
    zero = torch.zeros(1, cosine.shape[1], device=cosine.device, dtype=cosine.dtype)
    pos = (-alpha * (cosine - margin)).masked_fill(~positives, -torch.inf)
    neg = (alpha * (cosine + margin)).masked_fill(~negatives, -torch.inf)
    pos_term = torch.logsumexp(torch.cat([zero, pos], dim=0), dim=0)
    neg_term = torch.logsumexp(torch.cat([zero, neg], dim=0), dim=0)
    return pos_term[positives.any(0)].mean() + neg_term.mean()


def observed_field_classification_loss(logits, targets, vocab):
    """Multi-task categorical CE: unknown or rule-excluded fields have no loss."""
    positives, _ = target_masks(targets, len(vocab), logits.device)
    losses = []
    for field in sorted({a["field"] for a in vocab}):
        ids = [i for i, a in enumerate(vocab) if a["field"] == field]
        relevant = positives[:, ids]
        valid = relevant.any(-1)
        if valid.any():
            losses.append(F.cross_entropy(logits[valid][:, ids], relevant[valid].long().argmax(-1)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def family_supervision_loss(gates, positives, family_ids):
    present = torch.stack([positives[:, family_ids == family].any(-1)
                           for family in range(gates.shape[-1])], dim=-1)
    valid = present.any(-1)
    if not valid.any():
        return gates.sum() * 0
    target = present[valid].to(gates.dtype)
    target = target / target.sum(-1, keepdim=True)
    return -(target * gates[valid].clamp_min(1e-8).log()).sum(-1).mean()
