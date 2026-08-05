"""Aggregate factorial Paper 4 runs without conflating data and model seeds."""

import argparse
import json
import os

import numpy as np

from glioma.eval.paper4_diagnostics import (
    coadaptation_diagnostics,
    edge_stability,
    hierarchical_bootstrap_mean,
)
from glioma.io.artifacts import save_json


def _load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _metric(metrics, preferred=None):
    names = [preferred] if preferred else []
    names.extend(["map", "mAP", "accuracy", "binary_f1", "mrr"])
    for name in names:
        if name and name in metrics and np.isfinite(metrics[name]):
            return float(metrics[name]), name
    raise ValueError("No supported finite primary metric found")


def _variant_name(run_root, directory, config):
    explicit = config.get("variant_name")
    if explicit:
        return str(explicit)
    relative = os.path.relpath(directory, run_root).split(os.sep)
    return relative[0] if relative else config.get("local_topology", "run")


def discover_runs(run_root, preferred_metric=None):
    runs = []
    for directory, _, files in os.walk(run_root):
        if "config.json" not in files or "test_metrics.json" not in files:
            continue
        config = _load(os.path.join(directory, "config.json"))
        metrics = _load(os.path.join(directory, "test_metrics.json"))
        score, metric_name = _metric(metrics, preferred_metric)
        split_seed = int(config.get("split_seed", config.get("seed", 42)))
        model_seed = int(config.get("model_seed", config.get("seed", 42)))
        split_path = os.path.join(directory, "splits.json")
        split_signature = None
        if os.path.exists(split_path):
            split_signature = json.dumps(_load(split_path), sort_keys=True)
        intervention_path = os.path.join(directory, "intervention_metrics.json")
        if not os.path.exists(intervention_path):
            intervention_path = os.path.join(directory, "graph_role_metrics.json")
        adjacency_path = os.path.join(directory, "adjacency.json")
        if not os.path.exists(adjacency_path):
            topology_path = os.path.join(directory, "manifold_topology.json")
            adjacency = (
                _load(topology_path).get("local_adjacency_mean")
                if os.path.exists(topology_path)
                else None
            )
            if adjacency is not None and np.asarray(adjacency).ndim == 3:
                adjacency = np.asarray(adjacency, dtype=float).mean(axis=0).tolist()
        else:
            adjacency = _load(adjacency_path)
        diagnostics_path = os.path.join(directory, "diagnostics.json")
        diagnostics = _load(diagnostics_path) if os.path.exists(diagnostics_path) else {}
        efficiency_path = os.path.join(directory, "efficiency.json")
        if os.path.exists(efficiency_path):
            efficiency = _load(efficiency_path)
            diagnostics.setdefault("parameter_count", efficiency.get("parameters_total"))
            diagnostics.setdefault("forward_flops", efficiency.get("flops_per_patient_approx"))
            diagnostics.setdefault("latency_ms_per_patient", efficiency.get("latency_ms_per_patient"))
            diagnostics.setdefault("peak_gpu_memory_mb", efficiency.get("peak_gpu_memory_mb"))
        intervention_payload = _load(intervention_path) if os.path.exists(intervention_path) else {}
        if "interventions" in intervention_payload:
            intervention_payload = intervention_payload["interventions"]
        runs.append(
            {
                "variant": _variant_name(run_root, directory, config),
                "directory": directory,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "score": score,
                "metric": metric_name,
                "split_signature": split_signature,
                "interventions": intervention_payload,
                "adjacency": adjacency,
                "diagnostics": diagnostics,
            }
        )
    return runs


def aggregate(
    run_root,
    out_dir=None,
    preferred_metric=None,
    expected_variants=None,
    expected_split_seeds=None,
    expected_model_seeds=None,
    learned_variant="spd_cross_graph",
    uniform_variant="spd_uniform_graph",
    identity_variant="spd_identity_graph",
    euclidean_variant="euclidean_cross_graph",
):
    runs = discover_runs(run_root, preferred_metric)
    if not runs:
        raise ValueError(f"No completed Paper 4 runs found under {run_root}")
    by_variant = {}
    split_signatures = {}
    for run in runs:
        key = (run["split_seed"], run["model_seed"])
        if key in by_variant.setdefault(run["variant"], {}):
            raise ValueError(f"Duplicate run for {run['variant']} split/model seeds {key}")
        by_variant[run["variant"]][key] = run
        if run["split_signature"] is not None:
            split_key = run["split_seed"]
            previous = split_signatures.get(split_key)
            if previous is not None and previous != run["split_signature"]:
                raise ValueError(f"Data split mismatch for split_seed={split_key}")
            split_signatures[split_key] = run["split_signature"]

    expected_variants = list(expected_variants or [])
    expected_pairs = None
    if expected_split_seeds is not None and expected_model_seeds is not None:
        expected_pairs = {
            (int(split), int(model))
            for split in expected_split_seeds
            for model in expected_model_seeds
        }
    complete = True
    if expected_variants and not set(expected_variants).issubset(by_variant):
        complete = False
    if expected_pairs is not None:
        for variant in expected_variants or by_variant:
            if set(by_variant.get(variant, {})) != expected_pairs:
                complete = False

    payload = {
        "status": "factorial_complete" if complete else "prototype_incomplete",
        "primary_metric": preferred_metric or runs[0]["metric"],
        "variants": {},
        "paired_diagnostics": {},
        "fixed_split_edge_stability": {},
    }
    for variant, values in sorted(by_variant.items()):
        scores = [entry["score"] for entry in values.values()]
        by_split = {}
        for (split_seed, _), entry in values.items():
            by_split.setdefault(split_seed, []).append(entry["score"])
        payload["variants"][variant] = {
            "n": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1 if len(scores) > 1 else 0)),
            "hierarchical_bootstrap": hierarchical_bootstrap_mean(
                list(by_split.values()), seed=17
            ),
            "parameter_count": [
                entry["diagnostics"].get("parameter_count")
                for entry in values.values()
                if entry["diagnostics"].get("parameter_count") is not None
            ],
            "forward_flops": [
                entry["diagnostics"].get("forward_flops")
                for entry in values.values()
                if entry["diagnostics"].get("forward_flops") is not None
            ],
            "training_seconds": [
                entry["diagnostics"].get("training_seconds")
                for entry in values.values()
                if entry["diagnostics"].get("training_seconds") is not None
            ],
            "latency_ms_per_patient": [
                entry["diagnostics"].get("latency_ms_per_patient")
                for entry in values.values()
                if entry["diagnostics"].get("latency_ms_per_patient") is not None
            ],
            "peak_gpu_memory_mb": [
                entry["diagnostics"].get("peak_gpu_memory_mb")
                for entry in values.values()
                if entry["diagnostics"].get("peak_gpu_memory_mb") is not None
            ],
        }

    contrasts = {
        "geometry_gain": (learned_variant, euclidean_variant),
        "communication_gain": (uniform_variant, identity_variant),
        "allocation_gain": (learned_variant, uniform_variant),
    }
    for name, (left_name, right_name) in contrasts.items():
        if left_name not in by_variant or right_name not in by_variant:
            continue
        shared = sorted(set(by_variant[left_name]) & set(by_variant[right_name]))
        values = [
            by_variant[left_name][pair]["score"] - by_variant[right_name][pair]["score"]
            for pair in shared
        ]
        if values:
            payload["paired_diagnostics"][name] = {
                "values": values,
                "mean": float(np.mean(values)),
            }

    if learned_variant in by_variant and uniform_variant in by_variant:
        coadaptation = []
        for pair in sorted(set(by_variant[learned_variant]) & set(by_variant[uniform_variant])):
            learned = by_variant[learned_variant][pair]
            uniform_intervention = learned["interventions"].get("uniform", {})
            intervention_score = uniform_intervention.get(learned["metric"])
            if intervention_score is None and learned["metric"] == "map":
                delta = uniform_intervention.get("delta_map")
                intervention_score = learned["score"] + delta if delta is not None else None
            if intervention_score is not None:
                coadaptation.append(
                    coadaptation_diagnostics(
                        learned["score"],
                        float(intervention_score),
                        by_variant[uniform_variant][pair]["score"],
                    )
                )
        for name in ("intervention_cost", "retrained_advantage", "coadaptation_gap"):
            values = [entry[name] for entry in coadaptation]
            if values:
                payload["paired_diagnostics"][name] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                }

    for split_seed in sorted({run["split_seed"] for run in runs}):
        adjacencies = [
            entry["adjacency"]
            for (split, _), entry in by_variant.get(learned_variant, {}).items()
            if split == split_seed and entry["adjacency"] is not None
        ]
        if len(adjacencies) >= 2:
            payload["fixed_split_edge_stability"][str(split_seed)] = edge_stability(adjacencies)

    out_dir = out_dir or os.path.join(run_root, "aggregate_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "paper4_diagnostic_aggregate.json"), payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate factorial Paper 4 diagnostics")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--expected_variants", nargs="*", default=None)
    parser.add_argument("--expected_split_seeds", nargs="*", type=int, default=None)
    parser.add_argument("--expected_model_seeds", nargs="*", type=int, default=None)
    parser.add_argument("--learned_variant", default="spd_cross_graph")
    parser.add_argument("--uniform_variant", default="spd_uniform_graph")
    parser.add_argument("--identity_variant", default="spd_identity_graph")
    parser.add_argument("--euclidean_variant", default="euclidean_cross_graph")
    args = parser.parse_args(argv)
    payload = aggregate(
        args.run_root,
        args.out_dir,
        args.metric,
        args.expected_variants,
        args.expected_split_seeds,
        args.expected_model_seeds,
        args.learned_variant,
        args.uniform_variant,
        args.identity_variant,
        args.euclidean_variant,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["discover_runs", "aggregate"]
