"""Independent family heads over the unchanged Paper2 ROI encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from glioma.models.glioma_topomoe_net import GliomaTopoMoENet
from glioma.objectives import SemanticPrototypeBank


def conditional_scores(logits, gates, family_ids):
    conditional = torch.empty_like(logits)
    for family in range(gates.shape[-1]):
        selected = family_ids == family
        conditional[..., selected] = F.log_softmax(logits[..., selected], dim=-1)
    return conditional + gates.clamp_min(1e-8).log()[..., family_ids], conditional


class FactorizedHead(nn.Module):
    def __init__(self, dim, family_ids, projection="shared", gate="dynamic", num_nodes=3, initial_gate=None):
        super().__init__()
        if projection not in {"shared", "separate"} or gate not in {"global", "node", "dynamic"}:
            raise ValueError("Unsupported factorized head configuration")
        ids = torch.as_tensor(family_ids, dtype=torch.long)
        if ids.numel() == 0 or set(ids.tolist()) != set(range(int(ids.max()) + 1)):
            raise ValueError("Families must be nonempty and contiguous")
        self.register_buffer("family_ids", ids)
        self.num_families = int(ids.max()) + 1
        self.projection_mode, self.gate_mode = projection, gate
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_families)]) if projection == "separate" else None
        if gate == "dynamic":
            self.router = nn.Sequential(nn.Linear(dim + self.num_families, 64), nn.SiLU(), nn.Linear(64, self.num_families))
            self.register_parameter("gate_logits", None)
        else:
            self.router = None
            shape = (num_nodes if gate == "node" else 1, self.num_families)
            initial = torch.full(shape, 1 / self.num_families) if initial_gate is None else torch.as_tensor(initial_gate, dtype=torch.float32)
            if initial.shape != shape or not torch.isfinite(initial).all() or (initial <= 0).any():
                raise ValueError("Invalid train-only initial gate")
            self.gate_logits = nn.Parameter(initial.log())

    def forward(self, shared, prototypes, temperature, routing_override=None):
        prototypes = F.normalize(prototypes, dim=-1)
        if self.projections is None:
            logits = F.normalize(shared, dim=-1) @ prototypes.t() / temperature
        else:
            logits = torch.empty(*shared.shape[:-1], len(prototypes), device=shared.device, dtype=shared.dtype)
            for family, projection in enumerate(self.projections):
                ids = self.family_ids == family
                logits[..., ids] = F.normalize(projection(shared), dim=-1) @ prototypes[ids].t() / temperature
        if self.router is None:
            gates = self.gate_logits.softmax(-1).expand(shared.shape[0], shared.shape[1], -1)
        else:
            # Same unstructured family context used by the existing control.
            contexts = torch.stack([prototypes[self.family_ids == family].mean(0)
                                    for family in range(self.num_families)])
            context = F.normalize(shared, dim=-1) @ F.normalize(contexts, dim=-1).t()
            gates = self.router(torch.cat([shared, context], dim=-1)).softmax(-1)
        if routing_override is not None:
            if routing_override.shape != gates.shape or (routing_override < 0).any() or not torch.isfinite(routing_override).all() or (routing_override.sum(-1) <= 0).any():
                raise ValueError("Invalid routing override")
            gates = routing_override / routing_override.sum(-1, keepdim=True)
        scores, conditional = conditional_scores(logits, gates, self.family_ids)
        return {"scores": scores, "gates": gates, "conditional": conditional, "logits": logits}


class RevisionModel(nn.Module):
    def __init__(self, vocab, family_ids, method="factorized", projection="shared", gate="dynamic",
                 z_slices=7, feat_dim=256, shared_dim=128, temperature=0.07, initial_gate=None):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.encoder = GliomaTopoMoENet(z_slices=z_slices, feat_dim=feat_dim, shared_dim=shared_dim, routing_enabled=False)
        self.bank = SemanticPrototypeBank(len(vocab), shared_dim)
        self.method, self.temperature = method, temperature
        self.vocab = vocab
        self.head = FactorizedHead(shared_dim, family_ids, projection, gate, initial_gate=initial_gate) if method in {"factorized", "calibration"} else None
        self.classifier = nn.Linear(shared_dim, len(vocab)) if method == "multilabel" else None

    def forward(self, images, region_masks, stage="joint"):
        if self.method == "global":
            # Equal capacity control: each legacy region encoder sees the complete ROI.
            region_masks = torch.ones_like(region_masks)
        _, shared = self.encoder.encode_semantic_units(images, region_masks)
        if self.method == "global":
            shared = shared.mean(1, keepdim=True).expand(-1, 3, -1)
        cosine = F.normalize(shared, dim=-1) @ F.normalize(self.bank(), dim=-1).t()
        output = {"scores": cosine / self.temperature, "cosine": cosine, "shared": shared}
        if self.classifier is not None:
            logits = self.classifier(shared)
            field_scores = torch.empty_like(logits)
            for field in {a["field"] for a in self.vocab}:
                ids = [idx for idx, anchor in enumerate(self.vocab) if anchor["field"] == field]
                field_scores[..., ids] = F.log_softmax(logits[..., ids], dim=-1)
            output.update(scores=field_scores, classification_logits=logits)
        elif self.head is not None and stage != "direct":
            output.update(self.head(shared, self.bank(), self.temperature))
        return output
