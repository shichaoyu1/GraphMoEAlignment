"""Semantic anchor vocabulary utilities."""

from collections import defaultdict

from glioma.anchors import semantic_anchors


def build_anchor_vocab(cases, include_pathology=True, include_molecular=True, include_clinical=False):
    anchors = {}
    for case in cases:
        for anchor in semantic_anchors(
            case.get("metadata", {}),
            include_pathology=include_pathology,
            include_molecular=include_molecular,
            include_clinical=include_clinical,
        ):
            anchors[anchor["key"]] = anchor
    ordered = [anchors[key] for key in sorted(anchors)]
    key_to_id = {anchor["key"]: idx for idx, anchor in enumerate(ordered)}
    return ordered, key_to_id


def build_medclip_ignore_ids(anchor_vocab):
    buckets = defaultdict(list)
    for idx, anchor in enumerate(anchor_vocab):
        key = (anchor.get("source", ""), anchor.get("field", ""))
        buckets[key].append(idx)
    ignore_ids = []
    for anchor in anchor_vocab:
        key = (anchor.get("source", ""), anchor.get("field", ""))
        ignore_ids.append(buckets.get(key, []))
    return ignore_ids

__all__ = ["build_anchor_vocab", "build_medclip_ignore_ids"]
