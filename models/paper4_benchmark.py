"""Task-agnostic classifier built on the Paper 4 SPD diagnostic fusion core."""

import torch.nn as nn

from glioma.modules.hierarchical_spd_fusion import SPDMomentGraphFusion


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
    ):
        super().__init__()
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
        )
        self.classifier = nn.Linear(shared_dim, int(num_classes))

    def forward(
        self,
        tokens,
        modality_mask=None,
        token_mask=None,
        topology_override=None,
    ):
        fusion = self.fusion(
            tokens,
            modality_mask=modality_mask,
            token_mask=token_mask,
            local_topology_override=topology_override,
        )
        pooled = fusion["fused_nodes"].mean(dim=1)
        return {"logits": self.classifier(pooled), "fusion": fusion}


__all__ = ["Paper4BenchmarkModel"]
