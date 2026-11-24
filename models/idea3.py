"""Compact preference-conditioned MRI encoder and an authority residual baseline."""

import torch
from torch import nn
from torch.nn import functional as F

from glioma.data.utsw_authority import FEATURE_DIM, compile_grade
from glioma.models.encoders import ROI2DEncoder
from glioma.modules.authority_projection import project_allowed_simplex


class AnchorPhenotypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 12, 5, stride=2, padding=2), nn.GroupNorm(3, 12), nn.SiLU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv3d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv3d(32, 32, 4, stride=4), nn.SiLU(), nn.Flatten(),
        )
        self.bottleneck = nn.Sequential(nn.Linear(32 * 8 + 1, 64), nn.LayerNorm(64), nn.SiLU())
        self.anchor_head = nn.Linear(64, 3)
        self.phenotype_head = nn.Linear(64, 1)

    def forward(self, images, weight):
        encoded = self.encoder(images)
        preference = encoded.new_full((len(images), 1), weight)
        embedding = self.bottleneck(torch.cat((encoded, preference), dim=1))
        return self.anchor_head(embedding), self.phenotype_head(embedding).squeeze(-1).sigmoid(), embedding


def task_losses(logits, phenotype, anchors, target):
    observed = anchors >= 0
    counts = observed.sum(dim=0)
    anchor_losses = F.binary_cross_entropy_with_logits(logits, anchors.clamp_min(0), reduction="none")
    anchor = ((anchor_losses * observed).sum(dim=0) / counts.clamp_min(1)).sum()
    anchor = anchor / (counts > 0).sum().clamp_min(1)
    has_phenotype = target.isfinite()
    errors = (phenotype - target.nan_to_num()).abs()
    return anchor, (errors * has_phenotype).sum() / has_phenotype.sum().clamp_min(1)


class ReadOnlyGradeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(ROI2DEncoder() for _ in range(4))
        # The existing graph model uses the pre-pooling encoder trunk and likewise
        # gives only that trunk gradients from the grading objective.
        self.adapters = nn.ModuleList(nn.Linear(192, 7) for _ in range(4))
        self.report = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU())
        self.report_head = nn.Linear(64, 3)
        self.mri_head = nn.Sequential(nn.Linear(4 * 7 + 64, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, evidence):
        return self.predict_from_features(evidence, self.encode_mri(evidence.images))

    def encode_mri(self, images):
        features = []
        for channel, (encoder, adapter) in enumerate(zip(self.encoders, self.adapters)):
            _, tokens = encoder.forward_with_tokens(images[:, channel])
            features.append(adapter(tokens.mean(dim=1)))
        return torch.stack(features, dim=1)

    def predict_from_features(self, evidence, image_features):
        compiled = compile_grade(evidence)
        state = self.report(evidence.molecular) * evidence.available[:, 4:5]
        base_logits = self.report_head(state)
        features = image_features * evidence.available[:, :4, None]
        correction = self.mri_head(torch.cat([features.flatten(1), state], dim=1))
        correction = correction * evidence.available[:, :4].any(dim=1, keepdim=True)
        joint_logits = base_logits + correction
        report_prob = base_logits.softmax(dim=1)
        allowed_ratio = joint_logits[:, :2].softmax(dim=1)
        restricted = torch.cat([report_prob[:, :2].sum(dim=1, keepdim=True) * allowed_ratio,
                                report_prob[:, 2:]], dim=1)
        raw = torch.where(compiled["active"][:, None], restricted, joint_logits.softmax(dim=1))
        projected = project_allowed_simplex(raw, compiled["allowed"])
        return {"p0": raw, "p_star": projected, "designated_state": state, **compiled}
