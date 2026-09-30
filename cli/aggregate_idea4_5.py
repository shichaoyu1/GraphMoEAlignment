"""Aggregate Idea4.5 v3 runs at the independent data-seed level."""

import argparse
import csv
import json
import os

import numpy as np

from glioma.io.artifacts import save_json


METRICS = ("delta_abs_rich_to_sparse", "delta_rel_rich_vs_sparse", "A_r", "A_s", "J")
GROUP_FIELDS = (
    "dataset", "information_condition", "signal_form", "rich_fraction", "geometry",
    "local_topology", "edge_normalization", "readout_mode", "training_optimizer",
)


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate completed Idea4.5 v3 run directories")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4519)
    return parser


def _discover(root):
    summaries, inventory = [], []
    for directory, _, files in os.walk(root):
        if "run_status.json" not in files:
            continue
        with open(os.path.join(directory, "run_status.json"), encoding="utf-8") as handle:
            status = json.load(handle)
        item = {"path": os.path.relpath(directory, root), "status": status.get("status", "unknown")}
        if item["status"] == "completed" and "run_summary.json" in files:
            with open(os.path.join(directory, "run_summary.json"), encoding="utf-8") as handle:
                item["summary"] = json.load(handle)
                summaries.append(item)
        inventory.append({key: value for key, value in item.items() if key != "summary"})
    return summaries, inventory


def _interval(values, bootstrap, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), None, None
    means = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(bootstrap)
    ])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    discovered, inventory = _discover(args.root)
    rows = []
    for item in discovered:
        summary, config = item["summary"], item["summary"]["config"]
        for audit_optimizer, metrics in summary["audit_means"].items():
            row = {field: config.get(field) for field in GROUP_FIELDS}
            row.update({
                "audit_optimizer": audit_optimizer,
                "data_seed": int(config["data_seed"]), "train_seed": int(config["train_seed"]),
                "path": item["path"], **{metric: float(metrics[metric]) for metric in METRICS},
                "test_sparse_brier": float(summary["test"]["sparse"]["brier"]),
                "test_rich_brier": float(summary["test"]["rich"]["brier"]),
            })
            rows.append(row)
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS + ("audit_optimizer",))
        grouped.setdefault(key, []).append(row)
    rng = np.random.default_rng(args.seed)
    aggregates = []
    for key, group in grouped.items():
        # Training seeds are repeated measurements inside an independent data seed.
        data_clusters = {}
        for row in group:
            data_clusters.setdefault(row["data_seed"], []).append(row)
        record = dict(zip(GROUP_FIELDS + ("audit_optimizer",), key))
        record.update({"n_data_seeds": len(data_clusters), "n_runs": len(group)})
        for metric in METRICS + ("test_sparse_brier", "test_rich_brier"):
            cluster_values = [np.mean([row[metric] for row in cluster]) for cluster in data_clusters.values()]
            mean, low, high = _interval(cluster_values, args.bootstrap, rng)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        aggregates.append(record)
    save_json(os.path.join(out_dir, "run_inventory.json"), inventory)
    save_json(os.path.join(out_dir, "aggregation.json"), aggregates)
    with open(os.path.join(out_dir, "aggregation.csv"), "w", newline="", encoding="utf-8") as handle:
        if aggregates:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    completed = sum(item["status"] == "completed" for item in inventory)
    save_json(os.path.join(out_dir, "summary.json"), {
        "completed_runs": completed, "noncompleted_runs": len(inventory) - completed,
        "groups": len(aggregates), "independent_unit": "data_seed",
        "train_seed_handling": "averaged within data_seed before bootstrap",
    })
    return aggregates


if __name__ == "__main__":
    main()
