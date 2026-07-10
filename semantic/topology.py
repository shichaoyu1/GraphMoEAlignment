"""Disease-anchor topology prior and expert-family assignment.

Data-side helpers for the topology-guided mixture-of-anchor experts (TopoMoE).
``build_cooccurrence_prior`` derives a weak structural prior ``A_prior`` from
train-set anchor co-occurrence; ``anchor_family_ids`` maps each anchor to an
evidence-expert family (pathology / molecular / clinical) with a trailing
``residual`` family that owns no anchor.
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


def anchor_family_ids(anchor_vocab):
    """Map each anchor to a family index; append a residual family at the end.

    Returns ``(family_ids, family_names)`` where ``family_ids[i]`` is the family
    index of ``anchor_vocab[i]`` and ``family_names`` lists every family
    (anchor-owning families in priority order, then ``residual``).
    """
    present_sources = {anchor.get("source", "") for anchor in anchor_vocab}
    family_names = [name for source, name in _SOURCE_TO_FAMILY if source in present_sources]
    source_to_index = {}
    for source, name in _SOURCE_TO_FAMILY:
        if name in family_names:
            source_to_index[source] = family_names.index(name)
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


__all__ = ["anchor_family_ids", "build_cooccurrence_prior", "RESIDUAL_FAMILY"]
