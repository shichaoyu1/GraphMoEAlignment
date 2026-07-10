"""Objective functions and prototype bank for semantic alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPrototypeBank(nn.Module):
    def __init__(self, num_anchors, dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)

    def forward(self):
        return self.prototypes


def masked_multi_positive_nll(logits, target_ids):
    """Multi-positive InfoNCE NLL over precomputed logits (rows x anchors)."""
    if logits.numel() == 0:
        return logits.sum() * 0

    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, ids in enumerate(target_ids):
        if ids:
            mask[row, ids] = True
    if not mask.any():
        return logits.sum() * 0

    valid_rows = mask.any(dim=-1)
    if not valid_rows.any():
        return logits.sum() * 0

    masked_logits = logits.masked_fill(~mask, -1e9)
    per_row_loss = -(torch.logsumexp(masked_logits, dim=-1) - torch.logsumexp(logits, dim=-1))
    return per_row_loss[valid_rows].mean()


def multi_positive_contrastive_loss(queries, target_ids, prototypes, temperature=0.07):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    logits = queries @ prototypes.t() / temperature
    return masked_multi_positive_nll(logits, target_ids)


def medclip_multi_positive_loss(queries, target_ids, prototypes, ignore_ids_by_anchor, temperature=0.07):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    logits = queries @ prototypes.t() / temperature

    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        positive_mask[row, ids] = True
        for anchor_id in ids:
            for ignore_id in ignore_ids_by_anchor.get(anchor_id, ()):
                valid_mask[row, ignore_id] = False
        valid_mask[row, ids] = True
    if not positive_mask.any():
        return logits.sum() * 0

    valid_rows = positive_mask.any(dim=-1)
    if not valid_rows.any():
        return logits.sum() * 0

    masked_pos_logits = logits.masked_fill(~positive_mask, -1e9)
    masked_all_logits = logits.masked_fill(~valid_mask, -1e9)
    per_row_loss = -(torch.logsumexp(masked_pos_logits, dim=-1) - torch.logsumexp(masked_all_logits, dim=-1))
    return per_row_loss[valid_rows].mean()


def dcca_alignment_loss(queries, target_ids, prototypes, reg=1e-3):
    if queries.numel() == 0:
        return prototypes.sum() * 0
    queries = F.normalize(queries, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)

    x_rows = []
    y_rows = []
    for row, ids in enumerate(target_ids):
        if not ids:
            continue
        x_rows.append(queries[row])
        y_rows.append(prototypes[ids].mean(dim=0))
    if len(x_rows) < 2:
        return queries.sum() * 0

    x = torch.stack(x_rows, dim=0)
    y = torch.stack(y_rows, dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    n = x.shape[0]
    dim_x = x.shape[1]
    dim_y = y.shape[1]
    eye_x = torch.eye(dim_x, device=x.device, dtype=x.dtype)
    eye_y = torch.eye(dim_y, device=y.device, dtype=y.dtype)

    c_xx = (x.T @ x) / max(n - 1, 1) + reg * eye_x
    c_yy = (y.T @ y) / max(n - 1, 1) + reg * eye_y
    c_xx = 0.5 * (c_xx + c_xx.T)
    c_yy = 0.5 * (c_yy + c_yy.T)
    c_xy = (x.T @ y) / max(n - 1, 1)

    eval_x, evec_x = torch.linalg.eigh(c_xx)
    eval_y, evec_y = torch.linalg.eigh(c_yy)
    invsqrt_x = evec_x @ torch.diag(torch.rsqrt(torch.clamp(eval_x, min=1e-6))) @ evec_x.T
    invsqrt_y = evec_y @ torch.diag(torch.rsqrt(torch.clamp(eval_y, min=1e-6))) @ evec_y.T
    t_mat = invsqrt_x @ c_xy @ invsqrt_y
    corr = torch.linalg.svdvals(t_mat).sum()
    return -(corr / float(min(dim_x, dim_y)))


__all__ = [
    "SemanticPrototypeBank",
    "dcca_alignment_loss",
    "masked_multi_positive_nll",
    "medclip_multi_positive_loss",
    "multi_positive_contrastive_loss",
]
