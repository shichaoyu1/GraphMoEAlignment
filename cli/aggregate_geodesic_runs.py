"""Aggregate Paper 4 geodesic-fusion ablations across seeds."""

import argparse
import json
import os

import numpy as np

from glioma.io.artifacts import save_json
from glioma.visualization.geodesic_aggregate import save_geodesic_aggregate_figures


def _load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1 if len(finite) > 1 else 0)),
        "n": int(len(finite)),
    }


def _aggregate_numeric(payloads):
    keys = sorted(set().union(*(payload.keys() for payload in payloads)))
    result = {}
    for key in keys:
        values = [payload.get(key) for payload in payloads if payload.get(key) is not None]
        if not values:
            continue
        if all(isinstance(value, dict) for value in values):
            result[key] = _aggregate_numeric(values)
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = _stats(values)
    return result


def _variant_seed_dirs(run_root):
    variants = {}
    for name in sorted(os.listdir(run_root)):
        variant_root = os.path.join(run_root, name)
        if not os.path.isdir(variant_root):
            continue
        paper_root = os.path.join(variant_root, "paper4")
        if not os.path.isdir(paper_root):
            continue
        seeds = [
            os.path.join(paper_root, seed)
            for seed in sorted(os.listdir(paper_root))
            if seed.startswith("seed_") and os.path.isdir(os.path.join(paper_root, seed))
        ]
        if seeds:
            variants[name] = seeds
    paper_root = os.path.join(run_root, "paper4")
    if not variants and os.path.isdir(paper_root):
        seeds = [
            os.path.join(paper_root, seed)
            for seed in sorted(os.listdir(paper_root))
            if seed.startswith("seed_") and os.path.isdir(os.path.join(paper_root, seed))
        ]
        if seeds:
            variants["full_geodesic_graph"] = seeds
    return variants


def _validate_runs(variants):
    required = [
        "anchor_vocab.json",
        "splits.json",
        "test_metrics.json",
        "geodesic_diagnostics.json",
        "fusion_graph.json",
    ]
    canonical_vocab = None
    splits_by_seed = {}
    for variant, seed_dirs in variants.items():
        if len(seed_dirs) < 2:
            raise ValueError(f"Need at least two completed seeds for {variant}")
        for seed_dir in seed_dirs:
            missing = [name for name in required if not os.path.exists(os.path.join(seed_dir, name))]
            if missing:
                raise FileNotFoundError(f"{seed_dir} is missing: {', '.join(missing)}")
            vocab = json.dumps(_load(os.path.join(seed_dir, "anchor_vocab.json")), sort_keys=True, ensure_ascii=False)
            canonical_vocab = vocab if canonical_vocab is None else canonical_vocab
            if vocab != canonical_vocab:
                raise ValueError("Anchor vocabularies differ across Paper 4 runs")
            seed = os.path.basename(seed_dir)
            split = json.dumps(_load(os.path.join(seed_dir, "splits.json")), sort_keys=True)
            if seed in splits_by_seed and splits_by_seed[seed] != split:
                raise ValueError(f"Data splits differ across variants for {seed}")
            splits_by_seed[seed] = split


def _aggregate_graph(graphs):
    adjacency = np.asarray([graph["adjacency_mean"] for graph in graphs], dtype=np.float64)
    ddof = 1 if len(graphs) > 1 else 0
    stability = np.zeros_like(adjacency[0])
    for seed_matrix in adjacency:
        for region_idx in range(seed_matrix.shape[0]):
            off_diagonal = [
                (seed_matrix[region_idx, i, j], i, j)
                for i in range(seed_matrix.shape[1])
                for j in range(i + 1, seed_matrix.shape[2])
            ]
            for _, source, target in sorted(off_diagonal, reverse=True)[:2]:
                stability[region_idx, source, target] += 1
                stability[region_idx, target, source] += 1
    stability /= len(graphs)
    return {
        "modality_names": graphs[0].get("modality_names", []),
        "region_names": graphs[0].get("region_names", []),
        "adjacency_mean": adjacency.mean(axis=0).tolist(),
        "adjacency_std": adjacency.std(axis=0, ddof=ddof).tolist(),
        "edge_rank_stability": stability.tolist(),
    }


def _aggregate_diagnostics(payloads):
    result = {}
    for name in ("geodesic_energy", "linear_energy", "energy_ratio", "path_deviation"):
        values = [
            payload.get(name, {}).get("mean")
            for payload in payloads
            if isinstance(payload.get(name), dict) and payload.get(name, {}).get("mean") is not None
        ]
        if values:
            result[name] = _stats(values)
    result["by_region"] = _aggregate_numeric(
        [payload.get("by_region", {}) for payload in payloads]
    )
    result["by_pair"] = _aggregate_numeric([payload.get("by_pair", {}) for payload in payloads])
    return result


def aggregate(run_root, out_dir=None):
    variants = _variant_seed_dirs(run_root)
    if not variants:
        raise ValueError(f"No Paper 4 seed runs found under {run_root}")
    _validate_runs(variants)
    payload = {"status": "multi_seed_ablation_aggregate", "variants": {}}
    for variant, seed_dirs in variants.items():
        metrics = [_load(os.path.join(path, "test_metrics.json")) for path in seed_dirs]
        diagnostics = [_load(os.path.join(path, "geodesic_diagnostics.json")) for path in seed_dirs]
        graphs = [_load(os.path.join(path, "fusion_graph.json")) for path in seed_dirs]
        payload["variants"][variant] = {
            "seeds": [os.path.basename(path).removeprefix("seed_") for path in seed_dirs],
            "metrics": _aggregate_numeric(metrics),
            "diagnostics": _aggregate_diagnostics(diagnostics),
            "graph": _aggregate_graph(graphs),
        }

    out_dir = out_dir or os.path.join(run_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "aggregate_geodesic.json"), payload)
    figures = save_geodesic_aggregate_figures(payload, out_dir)
    manifest = {
        "status": payload["status"],
        "variants": list(payload["variants"]),
        "seed_count_by_variant": {
            name: len(entry["seeds"]) for name, entry in payload["variants"].items()
        },
        "figures": figures,
    }
    save_json(os.path.join(out_dir, "aggregate_manifest.json"), manifest)
    save_json(os.path.join(out_dir, "paper4_aggregate_figure_manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Paper 4 geodesic-fusion runs")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.run_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["aggregate", "main"]
