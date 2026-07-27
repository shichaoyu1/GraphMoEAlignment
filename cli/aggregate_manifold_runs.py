"""Aggregate Paper 4 graph-evidence screening and confirmation runs."""

import argparse
import json
import os

import numpy as np

from glioma.io.artifacts import save_json
from glioma.visualization.manifold_fusion import (
    save_graph_interventions,
    save_manifold_ablation,
    save_multiseed_topology,
)


SCREEN_VARIANTS = [
    "spd_cross_graph",
    "spd_identity_graph",
    "spd_uniform_graph",
    "spd_local_only",
    "spd_no_anchor_family",
    "euclidean_cross_graph",
    "latent_concat",
    "hemis",
    "gmu",
    "mbt_style",
]
CONFIRM_CORE_VARIANTS = [
    "spd_cross_graph",
    "spd_identity_graph",
    "spd_uniform_graph",
    "spd_local_only",
    "spd_no_anchor_family",
    "euclidean_cross_graph",
]
PUBLISHED_BASELINES = ["latent_concat", "hemis", "gmu", "mbt_style"]
LEGACY_VARIANTS = [
    "hierarchical_spd_graph",
    "euclidean_hierarchical_graph",
    "spd_local_only",
    "spd_no_anchor_family",
    "latent_concat",
]
EXPECTED_VARIANTS = SCREEN_VARIANTS


def _load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _metric_value(metrics, name):
    aliases = {"recall@1": ["recall@1", "r@1"], "map": ["map", "mAP"], "mrr": ["mrr", "MRR"]}
    for key in aliases.get(name, [name]):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"values": [], "mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
        "n": int(len(values)),
    }


def _paired_stats(values):
    result = _stats(values)
    finite = [float(value) for value in values if np.isfinite(value)]
    result.update(
        {
            "full_better_count": int(sum(value < 0 for value in finite)),
            "variant_better_count": int(sum(value > 0 for value in finite)),
            "tie_count": int(sum(value == 0 for value in finite)),
        }
    )
    return result


def _seed_dirs(run_root, variant):
    paper_root = os.path.join(run_root, variant, "paper4")
    if not os.path.isdir(paper_root):
        return []
    directories = [
        os.path.join(paper_root, name)
        for name in os.listdir(paper_root)
        if name.startswith("seed_") and os.path.isdir(os.path.join(paper_root, name))
    ]
    return sorted(directories, key=lambda path: int(os.path.basename(path).removeprefix("seed_")))


def _available_variants(run_root):
    if not os.path.isdir(run_root):
        return set()
    return {
        name for name in os.listdir(run_root)
        if os.path.isdir(os.path.join(run_root, name, "paper4"))
    }


def _best_published_baseline(run_root):
    scores = {}
    for variant in PUBLISHED_BASELINES:
        values = []
        for directory in _seed_dirs(run_root, variant):
            manifest_path = os.path.join(directory, "checkpoint_manifest.json")
            if not os.path.exists(manifest_path):
                continue
            direct = _load(manifest_path).get("direct", {})
            if direct.get("metric") == "map" and np.isfinite(direct.get("score", np.nan)):
                values.append(float(direct["score"]))
        if values:
            scores[variant] = float(np.mean(values))
    if not scores:
        return None, {}
    best = max(scores, key=lambda name: (scores[name], name))
    return best, scores


def _variant_protocol(run_root, stage):
    available = _available_variants(run_root)
    if stage == "legacy" or (stage == "auto" and "hierarchical_spd_graph" in available):
        return LEGACY_VARIANTS, [42, 43, 44], "legacy"
    if stage == "screen" or stage == "auto":
        return SCREEN_VARIANTS, [42, 43, 44], "screen"
    aggregate_path = os.path.join(run_root, "aggregate", "aggregate_manifold.json")
    best = _load(aggregate_path).get("best_published_baseline") if os.path.exists(aggregate_path) else None
    if best is None:
        best, _ = _best_published_baseline(run_root)
    if best is None:
        raise ValueError("Confirmation requires a completed screening aggregate")
    return CONFIRM_CORE_VARIANTS + [best], [42, 43, 44, 45, 46], "confirm"


def _macro_metric(metrics, group, name):
    return float(
        metrics.get("subgroups", {}).get("macro", {}).get(group, {}).get(name, float("nan"))
    )


def aggregate(run_root, out_dir=None, stage="auto", expected_seeds=None):
    variants, default_seeds, resolved_stage = _variant_protocol(run_root, stage)
    expected_seeds = default_seeds if expected_seeds is None else [int(value) for value in expected_seeds]
    payload = {
        "status": "prototype_incomplete",
        "stage": resolved_stage,
        "expected_seeds": expected_seeds,
        "variants": {},
        "paired_deltas_vs_spd_cross_graph": {},
        "topology_stability": {},
        "graph_role": {},
    }
    canonical_splits = {}
    canonical_vocab = None
    per_variant_by_seed = {}
    for variant in variants:
        directories = _seed_dirs(run_root, variant)
        if not directories:
            continue
        seeds = [int(os.path.basename(path).removeprefix("seed_")) for path in directories]
        if sorted(seeds) != sorted(expected_seeds):
            raise ValueError(f"{variant} seeds {seeds} do not match expected {expected_seeds}")
        metrics = []
        parameter_counts = []
        by_seed = {}
        for directory, seed in zip(directories, seeds):
            for required in ("anchor_vocab.json", "splits.json", "test_metrics.json"):
                if not os.path.exists(os.path.join(directory, required)):
                    raise FileNotFoundError(f"{directory} is missing {required}")
            vocab = json.dumps(_load(os.path.join(directory, "anchor_vocab.json")), sort_keys=True)
            if canonical_vocab is None:
                canonical_vocab = vocab
            elif canonical_vocab != vocab:
                raise ValueError(f"Anchor vocabulary mismatch in {directory}")
            split = json.dumps(_load(os.path.join(directory, "splits.json")), sort_keys=True)
            if seed in canonical_splits and canonical_splits[seed] != split:
                raise ValueError(f"Data split mismatch for seed_{seed}")
            canonical_splits[seed] = split
            entry = _load(os.path.join(directory, "test_metrics.json"))
            metrics.append(entry)
            by_seed[seed] = entry
            for manifest_name in ("paper4_manifold_figure_manifest.json", "paper4_baseline_manifest.json"):
                manifest_path = os.path.join(directory, manifest_name)
                if os.path.exists(manifest_path):
                    parameter_counts.append(_load(manifest_path).get("parameter_count", {}))
                    break
        per_variant_by_seed[variant] = by_seed
        payload["variants"][variant] = {
            "seeds": seeds,
            "metrics": {
                name: _stats([_metric_value(metric, name) for metric in metrics])
                for name in ("recall@1", "map", "mrr")
            },
            "macro": {
                group: {
                    name: _stats([_macro_metric(metric, group, name) for metric in metrics])
                    for name in ("recall@1", "map", "mrr")
                }
                for group in ("grade", "node", "target_family")
            },
            "parameter_count": parameter_counts[0] if parameter_counts else {},
        }

    reference = per_variant_by_seed.get("spd_cross_graph", {})
    for variant, values in per_variant_by_seed.items():
        if variant == "spd_cross_graph":
            continue
        shared = sorted(set(reference).intersection(values))
        payload["paired_deltas_vs_spd_cross_graph"][variant] = {
            name: _paired_stats([
                _metric_value(values[seed], name) - _metric_value(reference[seed], name)
                for seed in shared
            ])
            for name in ("recall@1", "map", "mrr")
        }

    main_dirs = _seed_dirs(run_root, "spd_cross_graph") or _seed_dirs(run_root, "hierarchical_spd_graph")
    topologies = [
        _load(os.path.join(path, "manifold_topology.json"))
        for path in main_dirs if os.path.exists(os.path.join(path, "manifold_topology.json"))
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

    graph_roles = [
        _load(os.path.join(path, "graph_role_metrics.json"))
        for path in main_dirs if os.path.exists(os.path.join(path, "graph_role_metrics.json"))
    ]
    if graph_roles:
        structural_keys = sorted(graph_roles[0].get("structural", {}))
        intervention_names = sorted(graph_roles[0].get("interventions", {}))
        payload["graph_role"] = {
            "structural": {
                key: {
                    "seed_means": _stats([
                        entry["structural"].get(key, {}).get("mean", np.nan)
                        for entry in graph_roles
                    ]),
                    "seed_medians": _stats([
                        entry["structural"].get(key, {}).get("median", np.nan)
                        for entry in graph_roles
                    ]),
                }
                for key in structural_keys if isinstance(graph_roles[0]["structural"].get(key), dict)
            },
            "identity_changed_fraction": _stats([
                entry["structural"].get("identity_changed_fraction", np.nan)
                for entry in graph_roles
            ]),
            "interventions": {
                name: {
                    metric: _paired_stats([
                        entry["interventions"][name].get(metric, np.nan)
                        for entry in graph_roles
                    ])
                    for metric in ("delta_map", "delta_mrr", "delta_recall@1")
                } | {
                    "patient_ap_mean_delta": _paired_stats([
                        entry["interventions"][name].get("patient_ap", {}).get("mean_delta", np.nan)
                        for entry in graph_roles
                    ]),
                    "patient_ap_ci_full_favor_count": int(sum(
                        entry["interventions"][name]
                        .get("patient_ap", {})
                        .get("ci95", [np.nan, np.nan])[1] < 0
                        for entry in graph_roles
                    )),
                }
                for name in intervention_names
            },
        }

    best, validation_scores = _best_published_baseline(run_root)
    if resolved_stage == "confirm":
        selected = next((name for name in variants if name in PUBLISHED_BASELINES), None)
        best = selected or best
        screening_path = os.path.join(run_root, "aggregate", "aggregate_manifold.json")
        if os.path.exists(screening_path):
            validation_scores = _load(screening_path).get(
                "published_baseline_validation_map", validation_scores
            )
    payload["best_published_baseline"] = best
    payload["published_baseline_validation_map"] = validation_scores
    paired = payload["paired_deltas_vs_spd_cross_graph"]
    graph_role = payload.get("graph_role", {})
    structural = graph_role.get("structural", {})
    local_cross = structural.get("local_offdiagonal_mass", {}).get("seed_means", {}).get("mean", np.nan)
    upper_cross = structural.get("upper_offdiagonal_mass", {}).get("seed_means", {}).get("mean", np.nan)
    local_update = structural.get("local_update_ratio", {}).get("seed_medians", {}).get("mean", np.nan)
    changed = graph_role.get("identity_changed_fraction", {}).get("mean", np.nan)
    threshold = 4 if resolved_stage == "confirm" else len(expected_seeds) + 1
    identity_delta = paired.get("spd_identity_graph", {}).get("map", {})
    local_only_delta = paired.get("spd_local_only", {}).get("map", {})
    uniform_delta = paired.get("spd_uniform_graph", {}).get("map", {})
    no_anchor_delta = paired.get("spd_no_anchor_family", {}).get("map", {})
    euclidean_delta = paired.get("euclidean_cross_graph", {}).get("map", {})
    best_delta = paired.get(best, {}).get("map", {}) if best else {}
    identity_patient = graph_role.get("interventions", {}).get("identity", {})
    payload["evidence_checks"] = {
        "graph_structurally_active": bool(
            np.isfinite(local_cross) and local_cross > 0
            and np.isfinite(upper_cross) and upper_cross > 0
            and np.isfinite(local_update) and local_update > 1e-3
            and np.isfinite(changed) and changed > 0.95
        ),
        "graph_improves_retrieval": bool(
            resolved_stage == "confirm"
            and identity_delta.get("full_better_count", 0) >= threshold
            and local_only_delta.get("full_better_count", 0) >= threshold
            and identity_patient.get("patient_ap_ci_full_favor_count", 0) >= threshold
        ),
        "learned_topology_supported": bool(
            resolved_stage == "confirm" and uniform_delta.get("full_better_count", 0) >= threshold
        ),
        "anchor_guided_graph_supported": bool(
            resolved_stage == "confirm" and no_anchor_delta.get("full_better_count", 0) >= threshold
        ),
        "manifold_fusion_supported": bool(
            resolved_stage == "confirm"
            and euclidean_delta.get("full_better_count", 0) >= threshold
            and best_delta.get("full_better_count", 0) >= threshold
        ),
        "note": "False means evidence is incomplete or inconsistent; it is not converted into a positive claim.",
    }
    complete = set(payload["variants"]) == set(variants)
    payload["status"] = "final_multiseed" if complete else "prototype_incomplete"
    out_dir = out_dir or os.path.join(run_root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "aggregate_manifold.json"), payload)

    figures = [save_manifold_ablation({}, out_dir, aggregate_payload=payload)]
    if payload["topology_stability"]:
        figures.append(save_multiseed_topology(payload["topology_stability"], out_dir))
    if payload["graph_role"]:
        graph_figure_payload = {
            "interventions": {
                name: {
                    metric: values.get("mean", np.nan)
                    for metric, values in entry.items()
                    if isinstance(values, dict) and metric.startswith("delta_")
                }
                for name, entry in payload["graph_role"]["interventions"].items()
            }
        }
        figures.append(save_graph_interventions(graph_figure_payload, out_dir))
    manifest = {
        "status": payload["status"],
        "stage": resolved_stage,
        "expected_variants": variants,
        "completed_variants": list(payload["variants"]),
        "expected_seeds": expected_seeds,
        "best_published_baseline": best,
        "figures": figures,
        "source_artifacts": ["aggregate_manifold.json"],
    }
    save_json(os.path.join(out_dir, "paper4_manifold_figure_manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Paper 4 manifold-fusion evidence")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--stage", default="auto", choices=["auto", "screen", "confirm", "legacy"])
    parser.add_argument("--expected_seeds", nargs="*", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.run_root, args.out_dir, args.stage, args.expected_seeds), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_VARIANTS",
    "SCREEN_VARIANTS",
    "CONFIRM_CORE_VARIANTS",
    "PUBLISHED_BASELINES",
    "aggregate",
    "main",
]
