"""Information-matched three-class UTSW adaptation; synthetic models remain unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from glioma.data.utsw_authority import FEATURE_DIM, compile_grade
from glioma.models.encoders import ROI2DEncoder
from glioma.models.authority_fusion import GraphUpdate, mean_augmented_spd
from glioma.modules.hierarchical_spd_fusion import spd_logm, symmetric_vectorize
from glioma.modules.authority_projection import project_allowed_simplex


class UTSWAuthorityGrade(nn.Module):
    def __init__(self, variant="acf"):
        super().__init__()
        self.variant = variant
        self.encoders = nn.ModuleList([ROI2DEncoder(in_ch=7) for _ in range(4)])
        self.adapters = nn.ModuleList([nn.Linear(192, 7) for _ in range(4)])
        self.report = nn.Linear(FEATURE_DIM, 7)
        self.support = nn.Linear(8, 7)
        self.register_buffer("templates", torch.eye(8)[:4])
        self.blocks = nn.ModuleList([GraphUpdate(36, 64, .35) for _ in range(2)])
        self.policy = nn.Sequential(nn.Linear(9, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(9 * 36 + 64, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, evidence, bypass=False):
        compiled = compile_grade(evidence, self.variant)
        matrices = []
        for index, (encoder, adapter) in enumerate(zip(self.encoders, self.adapters)):
            _, tokens = encoder.forward_with_tokens(evidence.images[:, index])
            matrices.append(mean_augmented_spd(adapter(tokens)))
        report = self.report(evidence.molecular)[:, None].expand(-1, 2, -1)
        matrices.append(mean_augmented_spd(report))
        supports = mean_augmented_spd(self.support(self.templates)[:, None].expand(-1, 2, -1))
        matrices = torch.cat([torch.stack(matrices, 1), supports[None].expand(len(report), -1, -1, -1)], 1)
        states = symmetric_vectorize(spd_logm(matrices))[:, None] * compiled["available"][..., None]
        for block in self.blocks:
            states, adjacency = block(states, compiled["mask"], compiled["available"], 5,
                                      "learned", None, bypass=bypass)
        policy = torch.cat([evidence.available, compiled["active"][:, None], compiled["allowed"]], 1).float()
        p0 = self.head(torch.cat([states.flatten(1), self.policy(policy)], 1)).softmax(-1)
        p_star = project_allowed_simplex(p0, compiled["allowed"])
        return dict(p0=p0, p_star=p_star, designated_state=states[:, 0, 4], adjacency=adjacency,
                    node_states=states, **compiled)


def grade_loss(output, labels):
    target = F.one_hot(labels, 3).to(output["p_star"])
    brier = (output["p_star"] - target).square().mean()
    penalty = (output["p_star"] - output["p0"]).square().mean()
    return brier + .1 * penalty
