"""Recompute the Idea 3 molecular AP diagnostics from saved out-of-fold predictions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from glioma.cli.idea3_anchor import atomic_csv, groups
from glioma.cli.utsw_authority import save_json
from glioma.data.idea3 import ANCHORS
from glioma.eval.idea3 import TARGET_COLS, check_pairs, interval, weighted_ap


def resample_by_fold(folds, repetitions):
    rng = np.random.default_rng(2026)
    weights = np.ones((repetitions + 1, len(folds)), dtype=np.int32)
    for fold in np.unique(folds):
        mask = folds == fold
        n = int(mask.sum())
        weights[1:, mask] = rng.multinomial(n, np.full(n, 1 / n), size=repetitions)
    return weights


def field_ap(labels, scores, weights):
    positive = weighted_ap(labels, scores, weights)
    negative = weighted_ap(np.where(labels < 0, -1, 1 - labels), 1 - scores, weights)
    return {"ap_positive": positive, "ap_negative": negative, "ap_balanced": (positive + negative) / 2}


def aggregate_supported(draws):
    supported = [draw for draw in draws if np.isfinite(draw[0])]
    return np.mean(supported, axis=0) if supported else np.full_like(draws[0], np.nan)


def auc_pairs(labels, scores, folds):
    delta = scores[labels == 1, None] - scores[None, labels == 0]
    same = folds[labels == 1, None] == folds[None, labels == 0]
    credit = (delta > 0) + .5 * (delta == 0)
    return {f"auc_{key}": float(credit[mask].mean()) if mask.any() else None
            for key, mask in (("within_pairs", same), ("cross_pairs", ~same))}


def mean(values):
    return float(values.mean()) if len(values) else None


def score_metrics(labels, scores, folds):
    valid = labels >= 0
    y, p = labels[valid], scores[valid]
    return {"n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
            "n_missing": int((~valid).sum()), "score_mean": mean(p),
            "score_sd": float(p.std()) if len(p) else None,
            "positive_score_mean": mean(p[y == 1]), "negative_score_mean": mean(p[y == 0]),
            "brier": mean((p - y) ** 2), **auc_pairs(labels, scores, folds)}


def diagnose(frame, seed, repetitions, reference_method="fixed", candidate_method="probe"):
    if reference_method == candidate_method:
        raise ValueError("Choose two different prediction methods")
    predictions = {method: groups(frame, method, seed) for method in (reference_method, candidate_method)}
    all_weights = tuple(predictions[reference_method])
    if len(all_weights) < 2 or tuple(predictions[candidate_method]) != all_weights:
        raise ValueError("Expected both prediction methods at the same two or more weights")
    ref = predictions[reference_method][all_weights[0]]
    folds = ref.fold.to_numpy()
    fold_ids = np.unique(folds)
    if len(fold_ids) < 2:
        raise ValueError("Cross-fold diagnosis requires at least two folds")
    subsets = {"pooled": np.ones(len(ref), dtype=bool), **{str(fold): folds == fold for fold in fold_ids}}
    draws = resample_by_fold(folds, repetitions)
    rates, contrasts, kept, leave_out = [], [], {}, []
    for method, weights in predictions.items():
        for weight, rows in weights.items():
            check_pairs(ref, rows, TARGET_COLS)
            for field in ANCHORS:
                labels = rows[field].fillna(-1).to_numpy()
                scores = rows[f"p_{field}"].to_numpy(dtype=float)
                if not np.isin(labels, [-1, 0, 1]).all() or not ((0 <= scores) & (scores <= 1)).all():
                    raise ValueError(f"Invalid scores or labels for {method}, {weight}, {field}")
                labels = labels.astype(int)
                fold_ranks = scores.copy()
                for fold in fold_ids:
                    mask = folds == fold
                    fold_ranks[mask] = (rankdata(scores[mask]) - .5) / mask.sum()
                results = {}
                for scope, mask in subsets.items():
                    ap = field_ap(labels[mask], scores[mask], draws[:, mask])
                    rates.append({"method": method, "weight": weight, "field": field, "scope": scope,
                                  **{key: interval(v)["estimate"] for key, v in ap.items()},
                                  **score_metrics(labels[mask], scores[mask], folds[mask])})
                    results[scope] = ap
                results["fold_mean"] = {key: aggregate_supported([results[str(fold)][key] for fold in fold_ids])
                                        for key in results["pooled"]}
                results["rank_pooled"] = field_ap(labels, fold_ranks, draws)
                kept[method, weight, field] = results
    for field in (*ANCHORS, "macro"):
        members = ANCHORS if field == "macro" else (field,)
        for scope in (*subsets, "fold_mean", "rank_pooled"):
            for metric in ("ap_balanced", "ap_positive", "ap_negative"):
                effects = {}
                for method in predictions:
                    low, high = [aggregate_supported([kept[method, w, f][scope][metric] for f in members])
                                 for w in (all_weights[0], all_weights[-1])]
                    effects[method] = high - low
                    contrasts.append({"method": method, "field": field, "scope": scope, "metric": metric,
                                      "low": interval(low)["estimate"], "high": interval(high)["estimate"],
                                      **interval(high - low)})
                gap = effects[candidate_method] - effects[reference_method]
                contrasts.append({"method": f"{candidate_method}_minus_{reference_method}",
                                  "field": field, "scope": scope, "metric": metric,
                                  "low": None, "high": None, **interval(gap)})
        for fold in fold_ids:
            keep = folds != fold
            for method, weights in predictions.items():
                endpoints = []
                for w in (all_weights[0], all_weights[-1]):
                    rows = weights[w].iloc[keep]
                    scores = [field_ap(rows[f].fillna(-1).to_numpy(dtype=int), rows[f"p_{f}"].to_numpy(),
                                       np.ones((1, len(rows)), dtype=int))["ap_balanced"] for f in members]
                    endpoints.append(float(aggregate_supported(scores)[0]))
                leave_out.append({"method": method, "field": field, "excluded_fold": int(fold),
                                  "low": endpoints[0], "high": endpoints[1], "delta": endpoints[1] - endpoints[0]})
    return {"n_patients": len(ref), "folds": fold_ids.tolist(), "seed": seed,
            "low_weight": all_weights[0], "high_weight": all_weights[-1], "bootstrap_repetitions": repetitions,
            "endpoint_contrasts": contrasts, "leave_one_fold_out": leave_out}, pd.DataFrame(rates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--reference-method", default="fixed")
    parser.add_argument("--candidate-method", default="probe")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    report, detail = diagnose(pd.read_csv(args.predictions, float_precision="round_trip"), args.seed, args.bootstrap,
                              args.reference_method, args.candidate_method)
    report["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(detail, args.output / "by_field_fold.csv")
    save_json(args.output / "diagnosis.json", report)


if __name__ == "__main__":
    main()
