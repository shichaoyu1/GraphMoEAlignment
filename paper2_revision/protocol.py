"""Target provenance and train-only statistical controls."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from glioma.anchors import semantic_anchors, target_anchor_keys


NODES = ("Necrotic/Core", "Edema", "Enhancing")
POLICIES = ("region_rules", "all_patient_anchors")
FIELDS = {
    "Necrotic/Core": ("Tumor Grade", "1p19Q CODEL", "Tumor Type"),
    "Edema": ("IDH", "Tumor Type", "Tumor Grade"),
    "Enhancing": ("Tumor Grade", "MGMT", "Tumor Type"),
}


@dataclass(frozen=True)
class QueryTarget:
    positives: tuple
    observed_candidates: tuple
    oov_keys: tuple
    rule_excluded_known: tuple
    fallback: bool


def query_target(metadata, node, policy, vocab):
    if policy not in POLICIES or node not in NODES:
        raise ValueError("Unsupported target policy or node")
    known = semantic_anchors(metadata)
    by_field = {anchor["field"]: anchor for anchor in known}
    selected_fields = tuple(by_field) if policy == "all_patient_anchors" else FIELDS[node]
    fallback = policy == "region_rules" and not any(field in by_field for field in selected_fields)
    if fallback:
        selected_fields = tuple(by_field)
    keys = target_anchor_keys(metadata, node, policy)
    key_to_id = {anchor["key"]: idx for idx, anchor in enumerate(vocab)}
    observed_fields = {field for field in selected_fields if field in by_field}
    # Supplemental observed-field protocol excludes unknown and unselected fields.
    observed = tuple(idx for idx, anchor in enumerate(vocab) if anchor["field"] in observed_fields)
    return QueryTarget(
        positives=tuple(key_to_id[key] for key in keys if key in key_to_id),
        observed_candidates=observed,
        oov_keys=tuple(key for key in keys if key not in key_to_id),
        rule_excluded_known=tuple(a["key"] for a in known if a["key"] not in keys),
        fallback=fallback,
    )


def targets_for_subjects(subjects, lookup, policy, vocab):
    return [query_target(lookup[str(subject)]["metadata"], node, policy, vocab)
            for subject in subjects for node in NODES]


def validate_splits(splits):
    if set(splits) != {"train", "val", "test"} or any(not values for values in splits.values()):
        raise ValueError("Nonempty train, validation and test splits are required; no fallback to test")
    seen = set()
    for name, cases in splits.items():
        ids = [str(case["subject_id"]) for case in cases]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate patient or patient leakage in " + name)
        seen.update(ids)


def audit_split(cases, policy, vocab):
    positive_sizes, missing_fields, grades = Counter(), Counter(), Counter()
    empty = oov = fallback = excluded = 0
    examples = []
    for case in cases:
        metadata = case["metadata"]
        known_fields = {a["field"] for a in semantic_anchors(metadata)}
        for field in {a["field"] for a in vocab}:
            missing_fields[field] += int(field not in known_fields)
        grades[str(metadata.get("Tumor Grade", "unknown"))] += 1
        for node in NODES:
            target = query_target(metadata, node, policy, vocab)
            positive_sizes[len(target.positives)] += 1
            empty += int(not target.positives)
            oov += len(target.oov_keys)
            fallback += int(target.fallback)
            excluded += len(target.rule_excluded_known)
            if len(examples) < 3:
                examples.append({"subject_id": str(case["subject_id"]), "node_name": node,
                                 **vars(target)})
    return {"n_patients": len(cases), "n_queries": len(cases) * len(NODES),
            "empty_target_queries": empty, "oov_positive_labels": oov,
            "fallback_queries": fallback, "rule_excluded_known_labels": excluded,
            "positive_size_histogram": dict(positive_sizes), "missing_fields": dict(missing_fields),
            "grade_counts": dict(grades), "worked_examples": examples}


def fit_frequency(train_cases, policy, vocab, by_node=False, smoothing=1.0):
    if not train_cases or smoothing <= 0:
        raise ValueError("Frequency controls require training cases and positive smoothing")
    counts = np.full((len(NODES) if by_node else 1, len(vocab)), smoothing, dtype=np.float64)
    for case in train_cases:
        for node_idx, node in enumerate(NODES):
            target = query_target(case["metadata"], node, policy, vocab)
            counts[node_idx if by_node else 0, list(target.positives)] += 1
    return counts / counts.sum(axis=-1, keepdims=True)


def fit_family_prior(train_cases, policy, vocab, family_ids, by_node=False, smoothing=1.0):
    """Equal-present-family target mean, fitted ONLY on train metadata."""
    if not train_cases or smoothing <= 0:
        raise ValueError("Static priors require training cases and positive smoothing")
    num_families = max(family_ids) + 1
    counts = np.full((len(NODES) if by_node else 1, num_families), smoothing, dtype=np.float64)
    for case in train_cases:
        for node_idx, node in enumerate(NODES):
            target = query_target(case["metadata"], node, policy, vocab)
            families = sorted({family_ids[idx] for idx in target.positives})
            if families:
                counts[node_idx if by_node else 0, families] += 1 / len(families)
    return counts / counts.sum(axis=-1, keepdims=True)
