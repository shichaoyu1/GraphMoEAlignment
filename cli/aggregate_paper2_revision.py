"""Aggregate paired revision runs, rejecting incompatible targets and galleries."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def load_runs(root, score="primary"):
    paths = sorted(Path(root).rglob("test_records.json"))
    if not paths:
        raise ValueError("No revision test records: " + str(root))
    runs = {}
    design = None
    for path in paths:
        if not (path.parent / "completed.json").is_file():
            raise ValueError("Incomplete run: " + str(path.parent))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        # A method root must not silently mix different architectures/recipes.
        recipe = {key: value for key, value in config.items() if key not in {
            "out_dir", "cv_fold", "seed", "splits_file", "reference_run"}}
        if design is not None and recipe != design:
            raise ValueError("Mixed experiment recipes in " + str(root))
        design = recipe
        records = json.loads(path.read_text(encoding="utf-8"))
        if score == "direct":
            if "direct_metrics" not in records:
                raise ValueError("No direct score replay in " + str(path))
            records["metrics"] = records["direct_metrics"]
        membership = json.loads((path.parent / "splits.json").read_text(encoding="utf-8"))
        records["split_membership"] = {name: sorted(map(str, ids)) for name, ids in membership.items()}
        if records.get("schema") != "paper2_revision_v1" or records.get("split") != "test":
            raise ValueError("Unsupported record schema")
        key = (config["cv_seed"], config["cv_fold"], config["seed"])
        if key in runs:
            raise ValueError("Duplicate fold/seed run: " + str(key))
        runs[key] = records
    return runs


def paired_summary(left, right, bootstrap=10000, seed=2026, protocol="full_gallery"):
    if set(left) != set(right):
        raise ValueError("Matched fold and seed coverage is required; no silent intersections")
    values = defaultdict(lambda: defaultdict(list))
    subject_folds = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("target_policy", "candidate_policy", "anchor_keys", "family_ids", "subject_ids", "node_names", "query_targets", "observed_candidates"):
            if a[field] != b[field]:
                raise ValueError("Incompatible paired experiment: " + field)
        if a.get("split_membership") != b.get("split_membership"):
            raise ValueError("Train/validation/test memberships differ")
        am, bm = a["metrics"][protocol]["patient_metrics"], b["metrics"][protocol]["patient_metrics"]
        if set(am) != set(bm):
            raise ValueError("Effective patient denominators differ")
        for subject in am:
            fold = key[:2]
            if subject in subject_folds and subject_folds[subject] != fold:
                raise ValueError("Repeated test patient across folds or CV assignments")
            subject_folds[subject] = fold
            for metric in ("map", "mrr", "hit@1"):
                values[subject][metric].append(bm[subject][metric] - am[subject][metric])
    if not values or bootstrap <= 0:
        raise ValueError("No paired patients or invalid bootstrap count")
    subjects = sorted(values)
    rng = np.random.default_rng(seed)
    result = {"n_patients": len(subjects), "n_paired_runs": len(left), "protocol": protocol,
              "analysis_unit": "patient after within-run nodes and within-patient seeds", "metrics": {}}
    for metric in ("map", "mrr", "hit@1"):
        deltas = np.asarray([np.mean(values[subject][metric]) for subject in subjects])
        # Avoid allocating bootstrap x cohort size for large independent cohorts.
        means = np.asarray([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(bootstrap)])
        result["metrics"][metric] = {"mean_delta": float(deltas.mean()), "ci95": np.percentile(means, [2.5, 97.5]).tolist()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline_score", choices=["primary", "direct"], default="primary")
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--protocol", choices=["full_gallery", "observed_fields"], default="full_gallery")
    args = parser.parse_args(argv)
    result = paired_summary(load_runs(args.baseline, args.baseline_score), load_runs(args.candidate), args.bootstrap, args.seed, args.protocol)
    result["baseline_score"] = args.baseline_score
    out = Path(args.out_file)
    # Exclusive creation prevents replacing an earlier analysis.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return result


if __name__ == "__main__":
    main()
