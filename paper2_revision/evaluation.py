"""Patient-level ranking, observed-field diagnostics and explicit gate permutations."""

from collections import defaultdict
from itertools import permutations

import numpy as np

from glioma.paper2_revision.protocol import NODES


def row_metrics(scores, positives, candidates=None):
    candidates = np.arange(len(scores)) if candidates is None else np.asarray(candidates, dtype=int)
    positives = set(positives).intersection(candidates.tolist())
    if not positives:
        return None
    ranking = candidates[np.argsort(-np.asarray(scores)[candidates], kind="stable")]
    hits = [rank for rank, anchor in enumerate(ranking, 1) if int(anchor) in positives]
    return {"map": float(np.mean([i / rank for i, rank in enumerate(hits, 1)])),
            "mrr": 1 / hits[0], "hit@1": float(int(ranking[0]) in positives)}


def summarize(scores, targets, subjects, nodes, grades, family_ids):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or len(scores) != len(targets) or len(subjects) != len(targets) or len(nodes) != len(targets) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores or mismatched query metadata")
    if len(set(zip(subjects, nodes))) != len(targets):
        raise ValueError("Duplicate subject-node queries")
    grouped = {"full_gallery": defaultdict(list), "observed_fields": defaultdict(list)}
    families = defaultdict(list)
    for row, (target, subject) in enumerate(zip(targets, subjects)):
        for name, candidates in (("full_gallery", None), ("observed_fields", target.observed_candidates)):
            metric = row_metrics(scores[row], target.positives, candidates)
            if metric is not None:
                grouped[name][str(subject)].append(metric)
        for family in sorted(set(family_ids)):
            ids = [i for i, value in enumerate(family_ids) if value == family]
            metric = row_metrics(scores[row], target.positives, ids)
            if metric is not None:
                families[family].append(metric["map"])
    result = {}
    for name, patients in grouped.items():
        values = {subject: {key: float(np.mean([m[key] for m in entries])) for key in ("map", "mrr", "hit@1")}
                  for subject, entries in patients.items()}
        grade_values = defaultdict(list)
        for subject, metric in values.items():
            grade_values[str(grades[subject])].append(metric["map"])
        grade_means = {grade: {"n_patients": len(v), "map": float(np.mean(v))} for grade, v in grade_values.items()}
        # Macro over observed recorded grade strings; unknown stays separately reported.
        known_grades = [item["map"] for grade, item in grade_means.items()
                        if grade.strip().lower() not in {"", "na", "n/a", "nan", "none", "unknown"}]
        result[name] = {"n_patients": len(values), "n_valid_queries": sum(map(len, patients.values())),
                        "n_empty_queries": len(targets) - sum(map(len, patients.values())),
                        "metrics": {key: float(np.mean([v[key] for v in values.values()])) if values else None for key in ("map", "mrr", "hit@1")},
                        "by_grade": grade_means, "grade_macro_map": float(np.mean(known_grades)) if known_grades else None,
                        "patient_metrics": values}
    result["within_family"] = {str(f): {"n_valid_queries": len(v), "map_query": float(np.mean(v))} for f, v in families.items()}
    return result


def gate_interventions(gates, conditional, subjects, nodes, family_ids, seed):
    """Gate-position replay, with every nonidentity node permutation recorded."""
    gates, conditional = np.asarray(gates), np.asarray(conditional)
    if len(gates) != len(subjects) or len(conditional) != len(subjects) or len(nodes) != len(subjects):
        raise ValueError("Intervention row mismatch")
    index = {(str(subject), node): row for row, (subject, node) in enumerate(zip(subjects, nodes))}
    patient_ids = sorted(set(map(str, subjects)))
    if len(index) != len(subjects) or set(nodes) != set(NODES) or len(index) != len(patient_ids) * len(NODES):
        raise ValueError("Gate replay requires exactly one query per patient and node")
    family_ids = np.asarray(family_ids, dtype=int)
    mix = lambda weights: np.log(np.clip(weights, 1e-8, None))[:, family_ids] + conditional
    outputs = {"uniform_gate": mix(np.full_like(gates, 1 / gates.shape[-1]))}
    mappings = {}
    for order in permutations(NODES):
        if order == NODES:
            continue
        name = "node_permutation_" + "_".join(str(NODES.index(node)) for node in order)
        mapping = dict(zip(NODES, order))
        rows = [index[(str(subject), mapping[node])] for subject, node in zip(subjects, nodes)]
        outputs[name] = mix(gates[rows])
        mappings[name] = mapping
    if len(patient_ids) > 1:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(patient_ids)
        # Rotate a shuffled order: every patient maps to a DIFFERENT patient.
        mapping = dict(zip(shuffled.tolist(), np.roll(shuffled, 1).tolist()))
        rows = [index[(mapping[str(subject)], node)] for subject, node in zip(subjects, nodes)]
        outputs["patient_shuffled_gate"] = mix(gates[rows])
        mappings["patient_shuffled_gate"] = mapping
    return outputs, {"scope": "frozen-expert gate replay; no retraining or input intervention",
                     "mappings": mappings, "node_permutation_count": 5}
