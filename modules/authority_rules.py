"""Pure authority compiler. Its public input intentionally contains no target labels."""

from dataclasses import dataclass

import torch

from glioma.data.authority_benchmarks import AuthorityEvidence


@dataclass
class CompiledAuthority:
    allowed_dependencies: torch.Tensor  # [B, relation, receiver, sender]
    node_available: torch.Tensor
    protected_nodes: torch.Tensor
    authorized_sources: torch.Tensor
    relation_active: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    constraint_active: torch.Tensor
    constraint_relation: torch.Tensor
    allowed_classes: torch.Tensor
    directions: torch.Tensor
    status: torch.Tensor  # 0 constrained, 1 unconstrained, 2 conflict, 3 no_input
    task: str
    rule_version: str


def compile_authority(evidence: AuthorityEvidence, num_supports=4):
    if not isinstance(evidence, AuthorityEvidence):
        raise TypeError("compile_authority accepts label-free AuthorityEvidence only")
    e = evidence
    if e.task not in ("s1", "s2") or e.rule_version != "authority-v1":
        raise ValueError("Unsupported task or authority rule version")
    if not bool(((e.verified_validity >= 0) & (e.verified_validity <= 2)).all()):
        raise ValueError("Validity must be invalid=0, valid=1, unknown=2")
    batch, modalities = e.source_available.shape
    relations = 1 if e.task == "s1" else 2
    if e.authority_sources.shape != (batch, relations):
        raise ValueError("Wrong relation-specific authority source shape")
    if not bool(((e.authority_sources >= 0) & (e.authority_sources < modalities)).all()):
        raise ValueError("Authority source index out of range")
    device = e.source_available.device
    row = torch.arange(batch, device=device)
    usable = e.source_available & (e.verified_validity != 0)
    admitted = e.source_available & (e.verified_validity == 1)
    active = admitted.gather(1, e.authority_sources)
    relation_axis = torch.arange(relations, device=device)[None].expand(batch, -1)
    directions = e.statements[row[:, None], e.authority_sources, relation_axis]
    if not bool(((directions == 0) | (directions == 1)).all()):
        raise ValueError("Statements must be binary in the registered synthetic protocols")
    count = modalities + num_supports
    node_available = torch.cat([usable, torch.ones(batch, num_supports, device=device, dtype=torch.bool)], -1)
    node_available = node_available[:, None].expand(-1, relations, -1)
    mask = node_available[..., :, None] & node_available[..., None, :]
    protected = torch.zeros(batch, relations, count, dtype=torch.bool, device=device)
    for r in range(relations):
        owner = e.authority_sources[:, r]
        protected[row, r, owner] = active[:, r]
        # Each protected source reads only its own state plus immutable global supports.
        allowed_row = torch.zeros(batch, count, dtype=torch.bool, device=device)
        allowed_row[row, owner] = True
        allowed_row[:, modalities:] = True
        mask[row, r, owner] = torch.where(active[:, r, None], allowed_row, mask[row, r, owner])
    # Supports cannot relay sample-dependent auxiliary information into a protected row.
    for support in range(modalities, count):
        mask[:, :, support] = False
        mask[:, :, support, support] = True
    mask &= node_available[..., :, None] & node_available[..., None, :]
    classes = 4 if e.task == "s1" else 6
    dtype = torch.float32
    if e.task == "s1":
        directions = directions ^ e.target_id[:, None]
        groups = torch.arange(classes, device=device) // 2
        allowed = (groups[None] == directions[:, :1]) | ~active[:, :1]
        B = -torch.eye(classes, device=device, dtype=dtype)[None].expand(batch, -1, -1).clone()
        b = torch.zeros(batch, classes, device=device)
        ca = ~allowed
        cr = torch.zeros(classes, device=device, dtype=torch.long)
    else:
        allowed = torch.ones(batch, classes, device=device, dtype=torch.bool)
        B = torch.zeros(batch, relations, classes, device=device)
        for r in range(relations):
            sign = directions[:, r].float() * 2 - 1
            B[:, r, 2 * r] = sign
            B[:, r, 2 * r + 1] = -sign
        b = torch.zeros(batch, relations, device=device)
        ca, cr = active.clone(), torch.arange(relations, device=device)
    # A registered inconsistent-protocol stress event: p[0] >= .8 and p[0] <= .2.
    # These are explicit synthetic constraints, not an inferred clinical probability margin.
    extras = torch.zeros(batch, 2, classes, device=device)
    extras[:, 0, 0], extras[:, 1, 0] = 1., -1.
    extra_active = (e.conflict & active.any(-1))[:, None].expand(-1, 2)
    B = torch.cat([B, extras], 1)
    b = torch.cat([b, torch.tensor([.8, -.2], device=device)[None].expand(batch, -1)], 1)
    ca = torch.cat([ca, extra_active], 1)
    cr = torch.cat([cr, torch.arange(relations, relations + 2, device=device)])
    relation_active = torch.cat([active, extra_active], 1)
    status = torch.where(active.any(-1), 0, 1).long()
    status = torch.where(extra_active.any(-1), 2, status)
    status = torch.where(~usable.any(-1), 3, status)
    return CompiledAuthority(mask, node_available, protected, e.authority_sources.clone(), relation_active,
                             B, b, ca, cr, allowed, directions, status, e.task, e.rule_version)


def relation_residuals(probabilities, compiled):
    raw = (compiled.b.to(probabilities) - torch.einsum("brk,bk->br", compiled.B.to(probabilities), probabilities)).clamp_min(0)
    raw = raw * compiled.constraint_active
    return torch.stack([raw[:, compiled.constraint_relation == r].amax(-1)
                        for r in range(compiled.relation_active.shape[1])], -1)
