"""Matched published multimodal fusion baselines for Paper 4."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values, mask, dim):
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp(min=1.0)


class PublishedModalityFusion(nn.Module):
    """Fair fusion baselines sharing the same modality encoder outputs."""

    MODES = {"latent_concat", "hemis", "gmu", "mbt_style"}

    def __init__(self, shared_dim, num_modalities=4, mode="latent_concat", bottleneck_tokens=2):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported published fusion baseline: {mode}")
        self.shared_dim = int(shared_dim)
        self.num_modalities = int(num_modalities)
        self.mode = mode
        self.concat_projection = nn.Sequential(
            nn.Linear(self.num_modalities * self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.SiLU(),
        )
        self.hemis_projection = nn.Sequential(
            nn.Linear(2 * self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.SiLU(),
        )
        self.gmu_candidates = nn.ModuleList(
            [nn.Linear(self.shared_dim, self.shared_dim) for _ in range(self.num_modalities)]
        )
        self.gmu_gates = nn.ModuleList(
            [nn.Linear(self.shared_dim, 1) for _ in range(self.num_modalities)]
        )
        heads = 4 if self.shared_dim % 4 == 0 else 1
        self.bottlenecks = nn.Parameter(torch.randn(int(bottleneck_tokens), self.shared_dim) * 0.02)
        self.bottleneck_attention = nn.MultiheadAttention(
            self.shared_dim, heads, batch_first=True
        )
        self.modality_attention = nn.MultiheadAttention(
            self.shared_dim, heads, batch_first=True
        )
        self.bottleneck_norm = nn.LayerNorm(self.shared_dim)
        self.modality_norm = nn.LayerNorm(self.shared_dim)
        self.register_buffer("pair_indices", torch.empty(0, 2, dtype=torch.long), persistent=False)

    def _hemis(self, nodes, mask):
        mean = _masked_mean(nodes, mask, dim=2)
        variance = _masked_mean((nodes - mean.unsqueeze(2)).square(), mask, dim=2)
        return self.hemis_projection(torch.cat([mean, variance], dim=-1))

    def _gmu(self, nodes, mask):
        candidates = torch.stack(
            [torch.tanh(layer(nodes[:, :, index])) for index, layer in enumerate(self.gmu_candidates)],
            dim=2,
        )
        logits = torch.cat(
            [layer(nodes[:, :, index]) for index, layer in enumerate(self.gmu_gates)], dim=-1
        )
        logits = logits.masked_fill(~mask[:, None, :], -1e9)
        gates = torch.softmax(logits, dim=-1)
        return (gates.unsqueeze(-1) * candidates).sum(dim=2)

    def _mbt(self, nodes, mask):
        batch, regions, modalities, dim = nodes.shape
        flat = nodes.reshape(batch * regions, modalities, dim)
        flat_mask = mask[:, None, :].expand(-1, regions, -1).reshape(batch * regions, modalities)
        bottlenecks = self.bottlenecks[None].expand(batch * regions, -1, -1)
        bottleneck_message, _ = self.bottleneck_attention(
            bottlenecks, flat, flat, key_padding_mask=~flat_mask, need_weights=False
        )
        bottlenecks = self.bottleneck_norm(bottlenecks + bottleneck_message)
        modality_message, _ = self.modality_attention(flat, bottlenecks, bottlenecks, need_weights=False)
        updated = self.modality_norm(flat + modality_message).reshape(batch, regions, modalities, dim)
        return _masked_mean(updated, mask[:, None, :], dim=2)

    def forward(self, modality_nodes, anchor_prototypes=None, modality_mask=None, **_):
        del anchor_prototypes
        if modality_nodes.ndim != 4:
            raise ValueError("modality_nodes must have shape [B, R, M, D]")
        batch, regions, modalities, dim = modality_nodes.shape
        if modalities != self.num_modalities or dim != self.shared_dim:
            raise ValueError("Unexpected modality-node shape")
        if modality_mask is None:
            modality_mask = torch.ones(batch, modalities, device=modality_nodes.device, dtype=torch.bool)
        else:
            modality_mask = modality_mask.to(device=modality_nodes.device).bool()
        if (modality_mask.sum(dim=-1) == 0).any():
            raise ValueError("At least one modality must be available for every case")
        nodes = F.normalize(modality_nodes, dim=-1)
        if self.mode == "latent_concat":
            masked = nodes * modality_mask[:, None, :, None].to(nodes.dtype)
            fused = self.concat_projection(masked.reshape(batch, regions, -1))
        elif self.mode == "hemis":
            fused = self._hemis(nodes, modality_mask[:, None, :])
        elif self.mode == "gmu":
            fused = self._gmu(nodes, modality_mask)
        else:
            fused = self._mbt(nodes, modality_mask)
        zero = fused.sum() * 0.0
        return {
            "fused_nodes": F.normalize(fused, dim=-1),
            "modality_nodes": nodes,
            "pair_indices": self.pair_indices,
            "pair_valid": torch.empty(batch, 0, device=nodes.device, dtype=torch.bool),
            "condition_loss": zero,
            "topology_loss": zero,
            "diagnostics": {"published_baseline": self.mode},
        }


__all__ = ["PublishedModalityFusion"]
