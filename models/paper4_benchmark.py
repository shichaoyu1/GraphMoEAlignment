"""Task-agnostic classifier built on the Paper 4 SPD diagnostic fusion core."""

import torch.nn.functional as F
import torch.nn as nn

from glioma.modules.hierarchical_spd_fusion import SPDMomentGraphFusion, symmetric_vectorize


class Paper4BenchmarkModel(nn.Module):
    def __init__(
        self,
        token_dim,
        num_modalities,
        num_classes,
        shared_dim=64,
        spd_dim=8,
        geometry="spd",
        local_topology="learned",
        local_cross_mass=0.35,
        moment_mode="covariance",
        readout_mode="pooled",
    ):
        super().__init__()
        if readout_mode not in {"pooled", "nodewise"}:
            raise ValueError("readout_mode must be pooled or nodewise")
        self.readout_mode = readout_mode
        self.fusion = SPDMomentGraphFusion(
            token_dim=token_dim,
            shared_dim=shared_dim,
            num_modalities=num_modalities,
            spd_dim=spd_dim,
            num_groups=1,
            geometry=geometry,
            local_topology=local_topology,
            upper_topology="identity",
            local_cross_mass=local_cross_mass,
            moment_mode=moment_mode,
        )
        self.classifier = nn.Linear(shared_dim, int(num_classes))

    def forward(
        self,
        tokens,
        modality_mask=None,
        token_mask=None,
        topology_override=None,
        gradient_edge=None,
    ):
        fusion = self.fusion(
            tokens,
            modality_mask=modality_mask,
            token_mask=token_mask,
            local_topology_override=topology_override,
            gradient_edge=gradient_edge,
        )
        if self.readout_mode == "nodewise":
            if self.fusion.num_groups != 1 or self.fusion.num_supports:
                raise ValueError("nodewise readout requires one group and no support nodes")
            nodes = F.normalize(
                self.fusion.readout(symmetric_vectorize(fusion["local_messages"])), dim=-1
            )
            mask = fusion["local_mask"].unsqueeze(-1).to(nodes.dtype)
            pooled = (nodes * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)
        else:
            pooled = fusion["fused_nodes"].mean(dim=1)
        return {"logits": self.classifier(pooled), "fusion": fusion}


__all__ = ["Paper4BenchmarkModel"]
