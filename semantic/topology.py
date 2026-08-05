"""Disease-anchor topology prior and expert-family assignment.

Data-side helpers for the topology-guided mixture-of-anchor experts (TopoMoE).
``build_cooccurrence_prior`` derives a weak structural prior ``A_prior`` from
train-set anchor co-occurrence; ``anchor_family_ids`` maps each anchor to an
evidence-expert family (pathology / molecular / clinical), with an optional
trailing ``residual`` family that owns no anchor.
"""

import torch

from glioma.anchors import semantic_anchors


# Fixed priority order so family indices are stable across runs.
_SOURCE_TO_FAMILY = [
    ("Pathology", "pathology"),
    ("Gene", "molecular"),
    ("Clinical", "clinical"),
]
RESIDUAL_FAMILY = "residual"


def anchor_family_ids(anchor_vocab, include_residual=True):
    """Map each anchor to a stable evidence-family index.

    Returns ``(family_ids, family_names)`` where ``family_ids[i]`` is the family
    index of ``anchor_vocab[i]``. Legacy experiments append a residual family;
    the Neurocomputing profiles disable it because it owns no retrieval anchor.
    """
    present_sources = {anchor.get("source", "") for anchor in anchor_vocab}
    family_names = [name for source, name in _SOURCE_TO_FAMILY if source in present_sources]
    source_to_index = {}
    for source, name in _SOURCE_TO_FAMILY:
        if name in family_names:
            source_to_index[source] = family_names.index(name)
    if include_residual:
        family_names = family_names + [RESIDUAL_FAMILY]

    family_ids = []
    for anchor in anchor_vocab:
        family_ids.append(source_to_index.get(anchor.get("source", ""), 0))
    return family_ids, family_names


def build_cooccurrence_prior(train_cases, anchor_vocab, key_to_id):
    """Symmetric, row-normalized anchor co-occurrence over train metadata.

    ``C[i, j]`` counts train patients whose metadata activates both anchor ``i``
    and anchor ``j`` (diagonal counts single-anchor prevalence). The matrix is
    symmetric by construction and row-normalized into a transition-like prior.
    """
    num_anchors = len(anchor_vocab)
    counts = torch.zeros(num_anchors, num_anchors, dtype=torch.float32)
    for case in train_cases:
        metadata = case.get("metadata", {})
        anchors = semantic_anchors(
            metadata,
            include_pathology=True,
            include_molecular=True,
            include_clinical=True,
        )
        ids = sorted({key_to_id[a["key"]] for a in anchors if a["key"] in key_to_id})
        for i in ids:
            for j in ids:
                counts[i, j] += 1.0

    counts = 0.5 * (counts + counts.t())
    row_sums = counts.sum(dim=1, keepdim=True)
    prior = torch.where(row_sums > 0, counts / row_sums.clamp(min=1e-8), counts)
    return prior


def controlled_topology_prior(num_anchors, policy="uniform", seed=42):
    """Build a deterministic training-time topology control.

    Controls depend only on the train-fold vocabulary size. ``uniform`` removes
    all edge information; ``random`` supplies a symmetric random graph before
    row normalization.
    """
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    if policy == "uniform":
        return torch.full((num_anchors, num_anchors), 1.0 / num_anchors)
    if policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weights = torch.rand(num_anchors, num_anchors, generator=generator)
        weights = 0.5 * (weights + weights.t())
        weights.fill_diagonal_(1.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    raise ValueError(f"Unsupported topology-control policy: {policy}")


def aggregate_family_prior(anchor_prior, family_ids, family_names):
    """Aggregate an anchor transition prior into family-level topology."""
    num_families = len(family_names)
    membership = torch.zeros(num_families, len(family_ids), dtype=anchor_prior.dtype)
    for anchor, family in enumerate(family_ids):
        membership[int(family), anchor] = 1.0
    membership = membership / membership.sum(dim=1, keepdim=True).clamp(min=1.0)
    prior = membership @ anchor_prior @ membership.t()
    if family_names and family_names[-1] == RESIDUAL_FAMILY:
        residual = num_families - 1
        prior[residual] = 1.0
        prior[:, residual] = torch.maximum(
            prior[:, residual], torch.full_like(prior[:, residual], 1e-2)
        )
    return prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)


__all__ = [
    "anchor_family_ids",
    "build_cooccurrence_prior",
    "controlled_topology_prior",
    "aggregate_family_prior",
    "RESIDUAL_FAMILY",
]
