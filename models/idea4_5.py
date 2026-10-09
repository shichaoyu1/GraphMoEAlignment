"""Idea4.5 model entry point using the existing Paper4 fusion backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from glioma.models.encoders import ROI2DEncoder
from glioma.models.paper4_benchmark import Paper4BenchmarkModel
from glioma.modules.hierarchical_spd_fusion import _identity_adjacency


class Idea45AvailabilityModel(Paper4BenchmarkModel):
    def __init__(self, *args, **kwargs):
        normalization = kwargs.pop("edge_normalization", "dynamic")
        if normalization not in {"dynamic", "fixed"}:
            raise ValueError("edge_normalization must be dynamic or fixed")
        kwargs.setdefault("moment_mode", "mean_augmented")
        kwargs.setdefault("readout_mode", "nodewise")
        super().__init__(*args, **kwargs)
        self.edge_normalization = normalization
        if normalization == "fixed":
            if self.fusion.local_topology == "learned":
                raise ValueError("Fixed edge normalization requires uniform or identity topology")
            self.fusion._local_adjacency = self._fixed_adjacency

    def _fixed_adjacency(self, representation, mask, topology):
        """Keep each public edge coefficient fixed as observed node count changes."""
        count = mask.shape[-1]
        identity = _identity_adjacency(mask, representation.dtype)
        if topology == "identity":
            adjacency = identity
        elif topology == "uniform":
            eye = torch.eye(count, device=mask.device, dtype=representation.dtype)
            valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            cross_weight = self.fusion.local_cross_mass / max(1, count - 1)
            adjacency = ((1.0 - self.fusion.local_cross_mass) * eye + cross_weight * (1.0 - eye)) * valid
        else:
            raise ValueError("Fixed edge normalization does not implement learned topology")
        delta = representation.unsqueeze(-3) - representation.unsqueeze(-4)
        distances = torch.sqrt(delta.square().sum(dim=(-1, -2)) + 1e-12)
        return adjacency, distances


class Idea45UTSWModel(nn.Module):
    """Encode raw UTSW MRI ROIs into modality tokens before the v3 audit model."""

    def __init__(
        self, z_slices, num_modalities, num_classes, roi_size=96,
        encoder_mode="pooled", token_grid=8, **kwargs,
    ):
        super().__init__()
        if encoder_mode not in {"pooled", "cnn"}:
            raise ValueError("encoder_mode must be pooled or cnn")
        self.num_modalities = int(num_modalities)
        self.encoder_mode = encoder_mode
        self.token_grid = int(token_grid)
        self.token_count = (
            max(1, int(roi_size) // 8) ** 2 if encoder_mode == "cnn" else self.token_grid ** 2
        )
        self.encoders = (
            nn.ModuleList(
                [ROI2DEncoder(in_ch=int(z_slices), feat_dim=256) for _ in range(self.num_modalities)]
            )
            if encoder_mode == "cnn" else nn.ModuleList()
        )
        self.availability = Idea45AvailabilityModel(
            token_dim=192 if encoder_mode == "cnn" else int(z_slices),
            num_modalities=num_modalities, num_classes=num_classes, **kwargs
        )

    def forward(self, images, modality_mask=None, gradient_edge=None, topology_override=None, **_):
        if images.ndim != 5 or images.shape[1] != self.num_modalities:
            raise ValueError("images must have shape [B,M,Z,H,W]")
        if modality_mask is None:
            modality_mask = torch.ones(
                len(images), self.num_modalities, device=images.device, dtype=torch.bool
            )
        modality_mask = modality_mask.to(device=images.device, dtype=torch.bool)
        if self.encoder_mode == "pooled":
            pooled = F.adaptive_avg_pool2d(
                images.reshape(-1, images.shape[2], images.shape[3], images.shape[4]),
                (self.token_grid, self.token_grid),
            )
            token_tensor = pooled.flatten(start_dim=2).transpose(1, 2)
            token_tensor = token_tensor.reshape(
                len(images), self.num_modalities, self.token_count, images.shape[2]
            )
            token_tensor = token_tensor * modality_mask[:, :, None, None].to(token_tensor.dtype)
            token_tensor = token_tensor.unsqueeze(1)
        else:
            tokens = []
            for modality, encoder in enumerate(self.encoders):
                if bool(modality_mask[:, modality].any()):
                    _, encoded = encoder.forward_with_tokens(images[:, modality])
                    encoded = encoded * modality_mask[:, modality, None, None].to(encoded.dtype)
                else:
                    encoded = images.new_zeros((len(images), self.token_count, 192))
                tokens.append(encoded)
            token_tensor = torch.stack(tokens, dim=1).unsqueeze(1)
        token_mask = modality_mask[:, None, :, None].expand(-1, 1, -1, token_tensor.shape[-2])
        return self.availability(
            token_tensor, modality_mask=modality_mask, token_mask=token_mask,
            topology_override=topology_override, gradient_edge=gradient_edge,
        )


__all__ = ["Idea45AvailabilityModel", "Idea45UTSWModel"]
