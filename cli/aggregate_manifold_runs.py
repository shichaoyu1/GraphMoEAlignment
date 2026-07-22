"""Aggregate the five-variant, three-seed Paper 4 manifold protocol."""

import argparse
import json
import os
import shutil

import numpy as np

from glioma.io.artifacts import save_json
from glioma.visualization.manifold_fusion import (
    save_ablation_evidence,
    save_manifold_overview,
    save_multiseed_topology,
    save_scale_to_manifold,
)


EXPECTED_VARIANTS = [
    "hierarchical_spd_graph",
    "euclidean_hierarchical_graph",
    "spd_local_only",
    "spd_no_anchor_family",
    "latent_concat",
]


def _load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _metric_value(metrics, name):
    aliases = {"recall@1": ["recall@1", "r@1"], "map": ["map", "mAP"], "mrr": ["mrr", "MRR"]}
    for key in aliases.get(name, [name]):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def _stats(values, seed=17):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"values": [], "mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
        "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "n": int(len(values)),
    }


def _seed_dirs(run_root, variant):
    paper_root = os.path.join(run_root, variant, "paper4")
    if not os.path.isdir(paper_root):
        return []
    return [
        os.path.join(paper_root, name)
        for name in sorted(os.listdir(paper_root))
        if name.startswith("seed_") and os.path.isdir(os.path.join(paper_root, name))
    ]


def aggregate(run_root, out_dir=None):
    payload = {"status": "prototype_incomplete", "variants": {}, "topology_stability": {}}
    canonical_splits = {}
    for variant in EXPECTED_VARIANTS:
        directories = _seed_dirs(run_root, variant)
        if not directories:
            continue
        metrics = []
        for directory in directories:
            for required in ("anchor_vocab.json", "splits.json", "test_metrics.json"):
                if not os.path.exists(os.path.join(directory, required)):
                    raise FileNotFoundError(f"{directory} is missing {required}")
            seed = os.path.basename(directory)
            split = json.dumps(_load(os.path.join(directory, "splits.json")), sort_keys=True)
            if seed in canonical_splits and canonical_splits[seed] != split:
                raise ValueError(f"Data split mismatch for {seed}")
            canonical_splits[seed] = split
            metrics.append(_load(os.path.join(directory, "test_metrics.json")))
        payload["variants"][variant] = {
            "seeds": [os.path.basename(path).removeprefix("seed_") for path in directories],
            "metrics": {
                name: _stats([_metric_value(metric, name) for metric in metrics])
                for name in ("recall@1", "map", "mrr")
            },
        }

    main_dirs = _seed_dirs(run_root, "hierarchical_spd_graph")
    topologies = [
        _load(os.path.join(path, "manifold_topology.json"))
        for path in main_dirs
        if os.path.exists(os.path.join(path, "manifold_topology.json"))
    ]
    if topologies:
        local = np.asarray([entry["local_adjacency_mean"] for entry in topologies], dtype=float)
        upper = np.asarray([entry["upper_adjacency_mean"] for entry in topologies], dtype=float)
        payload["topology_stability"] = {
            "upper_node_names": topologies[0].get("upper_node_names", []),
            "local_mean": local.mean(axis=0).tolist(),
            "local_std": local.std(axis=0, ddof=1 if len(local) > 1 else 0).tolist(),
            "upper_mean": upper.mean(axis=0).tolist(),
            "upper_std": upper.std(axis=0, ddof=1 if len(upper) > 1 else 0).tolist(),
        }

    complete = set(payload["variants"]) == set(EXPECTED_VARIANTS) and all(
        len(payload["variants"][name]["seeds"]) == 3 for name in EXPECTED_VARIANTS
    )
    payload["status"] = "final_multiseed" if complete else "prototype_incomplete"
    out_dir = out_dir or os.path.join(run_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "aggregate_manifold.json"), payload)
    figures = [save_manifold_overview(out_dir)]
    npz_paths = [
        os.path.join(path, "manifold_graph_records.npz")
        for path in main_dirs
        if os.path.exists(os.path.join(path, "manifold_graph_records.npz"))
    ]
    if npz_paths:
        arrays = [np.load(path) for path in npz_paths]
        records = {
            "manifold_fusion": {
                name: np.concatenate([entry[name] for entry in arrays], axis=0)
                for name in ("raw_scales", "raw_spd_traces", "condition_numbers")
            }
        }
        figures.append(save_scale_to_manifold(records, out_dir))
        for entry in arrays:
            entry.close()
    if payload["topology_stability"]:
        figures.append(save_multiseed_topology(payload["topology_stability"], out_dir))
    case_source = next(
        (
            os.path.join(path, "paper4_case_semantic_flow.png")
            for path in main_dirs
            if os.path.exists(os.path.join(path, "paper4_case_semantic_flow.png"))
        ),
        None,
    )
    if case_source:
        case_target = os.path.join(out_dir, "paper4_case_semantic_flow.png")
        shutil.copyfile(case_source, case_target)
        figures.append(os.path.basename(case_target))
    figures.append(save_ablation_evidence({}, out_dir, aggregate_payload=payload))
    manifest = {
        "status": payload["status"],
        "expected_variants": EXPECTED_VARIANTS,
        "completed_variants": list(payload["variants"]),
        "seed_count_by_variant": {
            name: len(entry["seeds"]) for name, entry in payload["variants"].items()
        },
        "figures": figures,
        "source_artifacts": ["aggregate_manifold.json"],
    }
    save_json(os.path.join(out_dir, "paper4_manifold_figure_manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Paper 4 manifold fusion runs")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.run_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_VARIANTS", "aggregate", "main"]
