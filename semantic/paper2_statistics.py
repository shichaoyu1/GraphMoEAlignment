"""Patient-clustered statistics for the Idea2 Neurocomputing experiment matrix."""

from collections import defaultdict
import json
import os

import numpy as np


METRICS = ("map", "mrr", "recall@1")


def find_artifacts(root, filename):
    if os.path.isfile(root):
        return [root] if os.path.basename(root) == filename else []
    matches = []
    for directory, _, files in os.walk(root):
        if filename in files:
            matches.append(os.path.join(directory, filename))
    return sorted(matches)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _row_metrics(scores, positives):
    positives = {int(value) for value in positives if 0 <= int(value) < len(scores)}
    if not positives:
        return None
    ranking = np.argsort(-np.asarray(scores, dtype=np.float64))
    hits = 0
    precisions = []
    first_rank = None
    for rank, anchor_id in enumerate(ranking, start=1):
        if int(anchor_id) in positives:
            hits += 1
            precisions.append(hits / rank)
            if first_rank is None:
                first_rank = rank
    return {
        "map": float(np.mean(precisions)),
        "mrr": float(1.0 / first_rank),
        "recall@1": float(int(int(ranking[0]) in positives)),
    }


def patient_metrics(payload, score_key):
    scores = np.asarray(payload.get(score_key, []), dtype=np.float64)
    targets = payload.get("query_targets", [])
    subject_ids = payload.get("subject_ids", [])
    if scores.ndim != 2 or len(scores) != len(targets):
        raise ValueError(f"{score_key} is missing or does not match query_targets")
    grouped = defaultdict(lambda: defaultdict(list))
    for row, (subject, positives) in enumerate(zip(subject_ids, targets)):
        metrics = _row_metrics(scores[row], positives)
        if metrics is None:
            continue
        for name, value in metrics.items():
            grouped[str(subject)][name].append(value)
    return {
        subject: {name: float(np.mean(values[name])) for name in METRICS}
        for subject, values in grouped.items()
    }


def aggregate_patient_metrics(record_files, score_key):
    repeated = defaultdict(lambda: defaultdict(list))
    for path in record_files:
        current = patient_metrics(load_json(path), score_key)
        for subject, metrics in current.items():
            for name, value in metrics.items():
                repeated[subject][name].append(value)
    return {
        subject: {name: float(np.mean(values[name])) for name in METRICS}
        for subject, values in repeated.items()
    }


def _bootstrap_mean(values, n_bootstrap=10000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_method(patient_values, n_bootstrap=10000, seed=42):
    summary = {"n_patients": int(len(patient_values)), "metrics": {}}
    for offset, metric in enumerate(METRICS):
        mean, ci = _bootstrap_mean(
            [values[metric] for values in patient_values.values()],
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        summary["metrics"][metric] = {"mean": mean, "ci95": ci}
    return summary


def paired_bootstrap(baseline, candidate, n_bootstrap=10000, seed=42):
    subjects = sorted(set(baseline).intersection(candidate))
    result = {"n_patients": int(len(subjects)), "metrics": {}}
    for offset, metric in enumerate(METRICS):
        left = np.asarray([baseline[subject][metric] for subject in subjects], dtype=np.float64)
        right = np.asarray([candidate[subject][metric] for subject in subjects], dtype=np.float64)
        deltas = right - left
        mean, ci = _bootstrap_mean(deltas, n_bootstrap=n_bootstrap, seed=seed + offset)
        result["metrics"][metric] = {
            "baseline_mean": float(left.mean()) if len(left) else float("nan"),
            "candidate_mean": float(right.mean()) if len(right) else float("nan"),
            "mean_delta": mean,
            "ci95": ci,
        }
    return result


def aggregate_interventions(record_files, n_bootstrap=10000, seed=42):
    baseline = aggregate_patient_metrics(record_files, "routed_scores")
    names = set()
    payloads = []
    for path in record_files:
        payload = load_json(path)
        payloads.append((path, payload))
        names.update(payload.get("intervention_scores", {}))
    results = {}
    for offset, name in enumerate(sorted(names)):
        repeated = defaultdict(lambda: defaultdict(list))
        for _, payload in payloads:
            if name not in payload.get("intervention_scores", {}):
                continue
            scenario_payload = dict(payload)
            scenario_payload["scenario_scores"] = payload["intervention_scores"][name]
            current = patient_metrics(scenario_payload, "scenario_scores")
            for subject, metrics in current.items():
                for metric, value in metrics.items():
                    repeated[subject][metric].append(value)
        scenario = {
            subject: {metric: float(np.mean(values[metric])) for metric in METRICS}
            for subject, values in repeated.items()
        }
        results[name] = paired_bootstrap(
            baseline,
            scenario,
            n_bootstrap=n_bootstrap,
            seed=seed + 100 * (offset + 1),
        )
    return results


def routing_contrasts(routing_files, n_bootstrap=10000, seed=42):
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    availability = defaultdict(lambda: defaultdict(list))
    for path in routing_files:
        payload = load_json(path)
        for record in payload.get("records", []):
            subject = str(record["subject_id"])
            node = str(record["node_name"])
            for family, weight in record.get("routing_weights", {}).items():
                values[subject][node][family].append(float(weight))
            for label, present in record.get("label_availability", {}).items():
                availability[subject][label].append(bool(present))

    node_pairs = (("Edema", "Enhancing"), ("Edema", "Necrotic/Core"), ("Enhancing", "Necrotic/Core"))
    families = sorted(
        {
            family
            for subject_values in values.values()
            for node_values in subject_values.values()
            for family in node_values
            if family != "residual"
        }
    )
    strata = {
        "all": lambda subject: True,
        "IDH_available": lambda subject: any(availability[subject].get("IDH", [])),
        "MGMT_available": lambda subject: any(availability[subject].get("MGMT", [])),
        "1p19q_available": lambda subject: any(availability[subject].get("1p19q", [])),
        "molecular_complete": lambda subject: all(
            any(availability[subject].get(label, [])) for label in ("IDH", "MGMT", "1p19q")
        ),
    }
    signatures = {
        tuple(any(availability[subject].get(label, [])) for label in ("IDH", "MGMT", "1p19q"))
        for subject in values
    }
    for signature in sorted(signatures):
        label = "availability_IDH{}_MGMT{}_1p19q{}".format(*[int(value) for value in signature])
        strata[label] = lambda subject, expected=signature: tuple(
            any(availability[subject].get(item, [])) for item in ("IDH", "MGMT", "1p19q")
        ) == expected
    result = {}
    counter = 0
    for stratum, include in strata.items():
        result[stratum] = {}
        for left, right in node_pairs:
            for family in families:
                contrasts = []
                for subject, subject_values in values.items():
                    if not include(subject):
                        continue
                    left_values = subject_values.get(left, {}).get(family, [])
                    right_values = subject_values.get(right, {}).get(family, [])
                    if left_values and right_values:
                        contrasts.append(float(np.mean(left_values) - np.mean(right_values)))
                counter += 1
                mean, ci = _bootstrap_mean(
                    contrasts,
                    n_bootstrap=n_bootstrap,
                    seed=seed + counter,
                )
                result[stratum][f"{left} - {right} | {family}"] = {
                    "n_patients": int(len(contrasts)),
                    "mean_delta": mean,
                    "ci95": ci,
                }
    return result


__all__ = [
    "METRICS",
    "aggregate_interventions",
    "aggregate_patient_metrics",
    "find_artifacts",
    "paired_bootstrap",
    "routing_contrasts",
    "summarize_method",
]
