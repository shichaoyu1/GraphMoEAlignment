"""Aggregate Idea2 cross-validation runs into patient-level paper evidence."""

import argparse
import csv
import json
import os

import numpy as np

from glioma.semantic.paper2_statistics import (
    METRICS,
    aggregate_interventions,
    aggregate_patient_metrics,
    find_artifacts,
    paired_bootstrap,
    routing_contrasts,
    summarize_method,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Patient-clustered Idea2 audit for Neurocomputing submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=SCORE_KEY|PATH",
        help="Method name, score field (direct_scores/routed_scores), and run root",
    )
    parser.add_argument(
        "--routing",
        action="append",
        default=[],
        metavar="POLICY=PATH",
        help="Named routing sensitivity run root",
    )
    parser.add_argument("--baseline", default="direct_only")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noninferiority_margin", type=float, default=-0.01)
    return parser


def _parse_method(spec):
    name, remainder = spec.split("=", 1)
    score_key, root = remainder.split("|", 1)
    if score_key not in {"direct_scores", "routed_scores"}:
        raise ValueError(f"Unsupported score key in {spec}")
    return name.strip(), score_key.strip(), os.path.abspath(root.strip())


def _parse_named_root(spec):
    name, root = spec.split("=", 1)
    return name.strip(), os.path.abspath(root.strip())


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _plot_model_summary(summaries, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summaries)
    colors = ["#0077BB", "#009988", "#EE7733", "#CC3311", "#BBBBBB"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for metric_idx, (axis, metric) in enumerate(zip(axes, METRICS)):
        means = np.asarray([summaries[name]["metrics"][metric]["mean"] for name in labels])
        intervals = np.asarray([summaries[name]["metrics"][metric]["ci95"] for name in labels])
        lower = means - intervals[:, 0]
        upper = intervals[:, 1] - means
        y = np.arange(len(labels))
        axis.errorbar(
            means,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#222222",
            ecolor="#555555",
            capsize=3,
        )
        for row, (value, color) in enumerate(zip(means, colors)):
            axis.scatter(value, row, color=color, edgecolor="black", linewidth=0.4, zorder=3)
        axis.set_title(metric)
        axis.set_xlabel("Patient-level estimate (95% CI)")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_yticks(y, labels if metric_idx == 0 else [])
        axis.invert_yaxis()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"figure2_model_performance.{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_effects(effects, title, filename, out_dir, limit=18):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name, payload in effects.items():
        metric_payload = payload.get("metrics", {}).get("map", payload)
        mean = metric_payload.get("mean_delta", float("nan"))
        ci = metric_payload.get("ci95", [float("nan"), float("nan")])
        if np.isfinite(mean):
            rows.append((name, float(mean), float(ci[0]), float(ci[1])))
    rows = sorted(rows, key=lambda item: abs(item[1]), reverse=True)[:limit]
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(rows) + 1.2)))
    y = np.arange(len(rows))
    means = np.asarray([row[1] for row in rows])
    intervals = np.asarray([[row[2], row[3]] for row in rows])
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - intervals[:, 0], intervals[:, 1] - means]),
        fmt="o",
        color="#0077BB",
        ecolor="#555555",
        capsize=3,
    )
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_yticks(y, [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Patient-level ΔmAP (95% paired CI)")
    axis.set_title(title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{filename}.{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _submission_gates(comparisons, interventions, routing, margin):
    gates = {}
    prior_vs_direct = comparisons.get("prior_guided_router_vs_direct_only")
    if prior_vs_direct:
        metric = prior_vs_direct["metrics"]["map"]
        gates["routed_noninferiority"] = {
            "pass": bool(metric["ci95"][0] > margin),
            "margin": margin,
            "evidence": metric,
        }
    else:
        gates["routed_noninferiority"] = {"pass": False, "reason": "comparison_missing"}

    structure_checks = []
    unstructured = comparisons.get("prior_guided_router_vs_unstructured_family_moe")
    if unstructured:
        structure_checks.append(unstructured["metrics"]["map"]["ci95"][0] > 0)
    for scenario in ("uniform_routing", "patient_shuffled_routing", "node_shuffled_routing"):
        if scenario in interventions:
            structure_checks.append(interventions[scenario]["metrics"]["map"]["ci95"][1] < 0)
    gates["structured_routing"] = {
        "pass": bool(structure_checks and all(structure_checks)),
        "checks_available": len(structure_checks),
    }

    learned = comparisons.get("prior_plus_learned_vs_prior_guided_router")
    if learned:
        metric = learned["metrics"]["map"]
        gates["learned_topology_primary_claim"] = {
            "pass": bool(metric["mean_delta"] >= 0.005 and metric["ci95"][0] > 0),
            "evidence": metric,
        }
    else:
        gates["learned_topology_primary_claim"] = {"pass": False, "reason": "comparison_missing"}

    required = {"region_rules", "all_patient_anchors", "family_supervision_off"}
    habitat_checks = []
    if required.issubset(routing):
        for name in sorted(required):
            all_rows = routing[name].get("all", {})
            complete_rows = routing[name].get("molecular_complete", {})
            for rows in (all_rows, complete_rows):
                molecular = rows.get("Edema - Enhancing | molecular")
                pathology = rows.get("Edema - Enhancing | pathology")
                habitat_checks.extend(
                    [
                        bool(molecular and molecular["ci95"][0] > 0),
                        bool(pathology and pathology["ci95"][1] < 0),
                    ]
                )
    gates["habitat_specific_allocation"] = {
        "pass": bool(habitat_checks and all(habitat_checks)),
        "checks_available": len(habitat_checks),
        "required_routing_runs": sorted(required),
    }
    return gates


def _write_csv(path, summaries):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "metric", "n_patients", "mean", "ci95_low", "ci95_high"])
        for method, summary in summaries.items():
            for metric, values in summary["metrics"].items():
                writer.writerow(
                    [method, metric, summary["n_patients"], values["mean"], *values["ci95"]]
                )


def _write_report(path, payload):
    lines = [
        "# Idea2 Neurocomputing statistical audit",
        "",
        f"Patient-clustered paired bootstrap: {payload['bootstrap']:,} resamples.",
        "",
        "## Submission gates",
        "",
    ]
    for name, gate in payload["submission_gates"].items():
        lines.append(f"- **{name}**: {'PASS' if gate.get('pass') else 'FAIL / NOT ASSESSABLE'}")
    lines.extend(
        [
            "",
            "The habitat claim remains a model-routing statement unless every sensitivity run passes.",
            "Interventions are model-internal sanity checks, not biological causal evidence.",
        ]
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main(args=None):
    args = build_parser().parse_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    methods = {}
    method_sources = {}
    for spec in args.method:
        name, score_key, root = _parse_method(spec)
        files = find_artifacts(root, "patient_level_records.json")
        if not files:
            raise FileNotFoundError(f"No patient_level_records.json under {root}")
        methods[name] = aggregate_patient_metrics(files, score_key)
        method_sources[name] = {"score_key": score_key, "files": files}
    if args.baseline not in methods:
        raise ValueError(f"Baseline {args.baseline} was not supplied")

    summaries = {
        name: summarize_method(values, n_bootstrap=args.bootstrap, seed=args.seed + index * 17)
        for index, (name, values) in enumerate(methods.items())
    }
    comparisons = {}
    baseline = methods[args.baseline]
    for index, (name, values) in enumerate(methods.items()):
        if name == args.baseline:
            continue
        comparisons[f"{name}_vs_{args.baseline}"] = paired_bootstrap(
            baseline,
            values,
            n_bootstrap=args.bootstrap,
            seed=args.seed + 1000 + index,
        )
    if {"prior_guided_router", "unstructured_family_moe"}.issubset(methods):
        comparisons["prior_guided_router_vs_unstructured_family_moe"] = paired_bootstrap(
            methods["unstructured_family_moe"],
            methods["prior_guided_router"],
            n_bootstrap=args.bootstrap,
            seed=args.seed + 2001,
        )
    if {"prior_guided_router", "prior_plus_learned"}.issubset(methods):
        comparisons["prior_plus_learned_vs_prior_guided_router"] = paired_bootstrap(
            methods["prior_guided_router"],
            methods["prior_plus_learned"],
            n_bootstrap=args.bootstrap,
            seed=args.seed + 2002,
        )

    prior_files = method_sources.get("prior_guided_router", {}).get("files", [])
    interventions = (
        aggregate_interventions(prior_files, n_bootstrap=args.bootstrap, seed=args.seed + 3000)
        if prior_files
        else {}
    )
    routing = {}
    for spec in args.routing:
        name, root = _parse_named_root(spec)
        files = find_artifacts(root, "routing_records.json")
        if files:
            routing[name] = routing_contrasts(files, n_bootstrap=args.bootstrap, seed=args.seed + 4000)

    gates = _submission_gates(
        comparisons,
        interventions,
        routing,
        args.noninferiority_margin,
    )
    payload = {
        "analysis_unit": "patient",
        "bootstrap": int(args.bootstrap),
        "baseline": args.baseline,
        "method_sources": method_sources,
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "interventions": interventions,
        "routing_contrasts": routing,
        "submission_gates": gates,
    }
    _save_json(os.path.join(args.out_dir, "paper2_statistical_audit.json"), payload)
    _write_csv(os.path.join(args.out_dir, "table2_patient_metrics.csv"), summaries)
    _write_report(os.path.join(args.out_dir, "SUBMISSION_GATE_REPORT.md"), payload)
    _plot_model_summary(summaries, args.out_dir)
    _plot_effects(interventions, "Routing intervention audit", "figure4_routing_interventions", args.out_dir)
    if routing:
        flattened = {
            f"{policy}: {name}": values
            for policy, strata in routing.items()
            for name, values in strata.get("all", {}).items()
        }
        _plot_effects(flattened, "Within-patient routing contrasts", "figure3_routing_contrasts", args.out_dir)
    return payload


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
