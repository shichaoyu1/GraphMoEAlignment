"""Idea4.5 model entry point using the existing Paper4 fusion backbone."""

import torch

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


__all__ = ["Idea45AvailabilityModel"]
