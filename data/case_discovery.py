"""Semantic case discovery utilities."""

from collections import defaultdict

import numpy as np

from glioma.anchors import semantic_anchors
from glioma.data.utsw_dataset import find_utsw_metadata, get_utsw_cases, load_utsw_metadata, parse_utsw_label


def grade_or_fallback_label(metadata):
    for task in ("grade", "idh", "mgmt", "1p19q"):
        try:
            label = parse_utsw_label(metadata, task)
        except ValueError:
            label = None
        if label is not None:
            return int(label)
    return 0


def discover_semantic_cases(root_dir, metadata_tsv=None, max_cases=None, seed=42, include_clinical=False):
    metadata_path = metadata_tsv or find_utsw_metadata(root_dir)
    metadata = load_utsw_metadata(metadata_path) if metadata_path else {}
    cases = []
    for case in get_utsw_cases(root_dir, metadata_tsv=metadata_path):
        info = metadata.get(case["subject_id"], case.get("metadata", {}))
        anchors = semantic_anchors(info, include_clinical=include_clinical)
        if not anchors:
            continue
        item = dict(case)
        item["metadata"] = info
        item["label"] = grade_or_fallback_label(info)
        cases.append(item)

    if max_cases and len(cases) > max_cases:
        rng = np.random.default_rng(seed)
        grouped = defaultdict(list)
        for case in cases:
            grouped[case["label"]].append(case)
        sampled = []
        for group in grouped.values():
            rng.shuffle(group)
        while len(sampled) < max_cases and any(grouped.values()):
            for label in sorted(grouped):
                if grouped[label] and len(sampled) < max_cases:
                    sampled.append(grouped[label].pop())
        cases = sorted(sampled, key=lambda item: item["subject_id"])
    return cases

__all__ = ["discover_semantic_cases", "grade_or_fallback_label"]
