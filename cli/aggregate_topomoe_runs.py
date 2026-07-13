"""Aggregate completed TopoMoE seed runs."""

import argparse
import json
import os

import numpy as np

from glioma.io.artifacts import save_json
from glioma.visualization.topomoe_aggregate import save_multiseed_figures


def _load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    ddof = 1 if len(finite) > 1 else 0
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite, ddof=ddof)), "n": int(len(finite))}


def _aggregate_numeric_dicts(payloads):
    keys = sorted(set().union(*(payload.keys() for payload in payloads)))
    result = {}
    for key in keys:
        values = [payload.get(key) for payload in payloads if payload.get(key) is not None]
        if not values:
            continue
        if all(isinstance(value, dict) for value in values):
            result[key] = _aggregate_numeric_dicts(values)
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = _stats(values)
    return result


def _discover_seed_dirs(run_root):
    paper_root = os.path.join(run_root, "paper2")
    if os.path.isdir(paper_root):
        run_root = paper_root
    return [
        os.path.join(run_root, name)
        for name in sorted(os.listdir(run_root))
        if name.startswith("seed_") and os.path.isdir(os.path.join(run_root, name))
    ]


def _aggregate_routing(payloads):
    families = payloads[0]["family_names"]
    result = {"family_names": families, "overall": {}, "by_node": {}, "by_grade": {}}
    for family in families:
        values = [
            payload["overall"]["mean"][family]
            for payload in payloads
            if family in payload.get("overall", {}).get("mean", {})
        ]
        if values:
            result["overall"][family] = _stats(values)
    for group_name in ("by_node", "by_grade"):
        labels = sorted(set().union(*(payload.get(group_name, {}).keys() for payload in payloads)))
        for label in labels:
            result[group_name][label] = {}
            for family in families:
                values = [
                    payload[group_name][label]["mean"][family]
                    for payload in payloads
                    if label in payload.get(group_name, {})
                ]
                result[group_name][label][family] = _stats(values)
    return result


def _aggregate_topology(payloads):
    prior = np.asarray([payload["A_prior"] for payload in payloads], dtype=np.float64)
    initial = np.asarray([payload["A_initial"] for payload in payloads], dtype=np.float64)
    effective = np.asarray([payload["A_effective"] for payload in payloads], dtype=np.float64)
    delta_vs_prior = effective - prior
    delta_vs_initial = effective - initial
    opened = (prior <= 0) & (effective > initial + 1e-6)
    ddof = 1 if len(payloads) > 1 else 0
    return {
        "seed_count": len(payloads),
        "A_prior_mean": prior.mean(axis=0).tolist(),
        "A_prior_std": prior.std(axis=0, ddof=ddof).tolist(),
        "A_initial_mean": initial.mean(axis=0).tolist(),
        "A_initial_std": initial.std(axis=0, ddof=ddof).tolist(),
        "A_effective_mean": effective.mean(axis=0).tolist(),
        "A_effective_std": effective.std(axis=0, ddof=ddof).tolist(),
        "delta_vs_prior_mean": delta_vs_prior.mean(axis=0).tolist(),
        "delta_vs_prior_std": delta_vs_prior.std(axis=0, ddof=ddof).tolist(),
        "delta_vs_initial_mean": delta_vs_initial.mean(axis=0).tolist(),
        "delta_vs_initial_std": delta_vs_initial.std(axis=0, ddof=ddof).tolist(),
        "new_edge_stability": opened.mean(axis=0).tolist(),
        "diagnostics": _aggregate_numeric_dicts([payload.get("diagnostics", {}) for payload in payloads]),
    }


def aggregate(run_root, out_dir=None):
    seed_dirs = _discover_seed_dirs(run_root)
    if len(seed_dirs) < 2:
        raise ValueError(f"Need at least two completed seed directories under {run_root}")
    required = ["anchor_vocab.json", "test_metrics.json", "routing_spectrum.json", "topomoe_topology.json", "intervention_metrics.json"]
    for seed_dir in seed_dirs:
        missing = [name for name in required if not os.path.exists(os.path.join(seed_dir, name))]
        if missing:
            raise FileNotFoundError(f"{seed_dir} is missing: {', '.join(missing)}")

    vocabularies = [_load(os.path.join(seed_dir, "anchor_vocab.json")) for seed_dir in seed_dirs]
    canonical = json.dumps(vocabularies[0], sort_keys=True, ensure_ascii=False)
    if any(json.dumps(vocab, sort_keys=True, ensure_ascii=False) != canonical for vocab in vocabularies[1:]):
        raise ValueError("Anchor vocabularies differ across seeds")

    metrics = [_load(os.path.join(seed_dir, "test_metrics.json")) for seed_dir in seed_dirs]
    routing = [_load(os.path.join(seed_dir, "routing_spectrum.json")) for seed_dir in seed_dirs]
    topology = [_load(os.path.join(seed_dir, "topomoe_topology.json")) for seed_dir in seed_dirs]
    interventions = [_load(os.path.join(seed_dir, "intervention_metrics.json")) for seed_dir in seed_dirs]
    out_dir = out_dir or os.path.join(run_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    aggregate_metrics = _aggregate_numeric_dicts(metrics)
    aggregate_routing = _aggregate_routing(routing)
    aggregate_topology = _aggregate_topology(topology)
    aggregate_interventions = _aggregate_numeric_dicts(interventions)
    save_json(os.path.join(out_dir, "aggregate_metrics.json"), aggregate_metrics)
    save_json(os.path.join(out_dir, "aggregate_routing_spectrum.json"), aggregate_routing)
    save_json(os.path.join(out_dir, "aggregate_topology.json"), aggregate_topology)
    save_json(os.path.join(out_dir, "aggregate_interventions.json"), aggregate_interventions)
    figures = save_multiseed_figures(aggregate_routing, aggregate_topology, aggregate_interventions, out_dir)
    manifest = {
        "status": "multi_seed_aggregate",
        "seeds": [os.path.basename(path).removeprefix("seed_") for path in seed_dirs],
        "seed_dirs": [os.path.basename(path) for path in seed_dirs],
        "seed_count": len(seed_dirs),
        "case_counts": [payload.get("case_count") for payload in metrics],
        "checkpoint_types": sorted({payload.get("checkpoint_type") for payload in metrics if payload.get("checkpoint_type")}),
        "topology_modes": sorted({payload.get("topology_mode") for payload in topology if payload.get("topology_mode")}),
        "figures": figures,
    }
    save_json(os.path.join(out_dir, "aggregate_manifest.json"), manifest)
    save_json(os.path.join(out_dir, "topomoe_aggregate_figure_manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate TopoMoE multi-seed runs")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args(argv)
    manifest = aggregate(args.run_root, args.out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["aggregate", "main"]
