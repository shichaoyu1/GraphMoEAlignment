"""Patient-paired evaluation for Idea 3."""

import numpy as np

from glioma.data.idea3 import ANCHORS
from glioma.eval.utsw_authority import metrics as grade_metrics


TARGET_COLS = [*ANCHORS, "phenotype"]
PROB_COLS = [f"p_{field}" for field in ANCHORS] + ["pred_phenotype"]


def check_pairs(a, b, columns):
    if a.subject_id.duplicated().any() or b.subject_id.duplicated().any():
        raise ValueError("A patient occurs more than once in a prediction group")
    if not a[["subject_id", "fold", *columns]].reset_index(drop=True).equals(
        b[["subject_id", "fold", *columns]].reset_index(drop=True)
    ):
        raise ValueError("Prediction groups do not have the same patients, folds and targets")


def patient_draws(n, repetitions):
    rng = np.random.default_rng(2026)
    return np.vstack([np.ones((1, n), dtype=np.int32), rng.multinomial(n, np.full(n, 1 / n), size=repetitions)])


def weighted_ap(y, score, weights):
    observed = y >= 0
    ranked = np.flatnonzero(observed)[np.argsort(-score[observed], kind="stable")]
    if not len(ranked):
        return np.full(len(weights), np.nan)
    hits = np.flatnonzero(np.r_[np.diff(score[ranked]) != 0, True])
    count = weights[:, ranked]
    positives = (count * y[ranked]).cumsum(axis=1)[:, hits]
    total = count.cumsum(axis=1)[:, hits]
    precision = positives / np.maximum(total, 1)
    ap = (np.diff(positives, prepend=0, axis=1) * precision).sum(axis=1)
    enough = (positives[:, -1] > 0) & (positives[:, -1] < total[:, -1])
    return np.divide(ap, positives[:, -1], out=np.full(len(ap), np.nan), where=enough)


def anchor_draws(frame, weights):
    stats = {}
    for field in ANCHORS:
        y = frame[field].fillna(-1).to_numpy(dtype=int)
        score = frame[f"p_{field}"].to_numpy()
        y_negative = np.where(y < 0, -1, 1 - y)
        stats[f"{field}_ap"] = (weighted_ap(y, score, weights) + weighted_ap(y_negative, 1 - score, weights)) / 2
    stats["anchor_ap"] = np.mean(list(stats.values()), axis=0)
    phenotype = frame.phenotype.notna().to_numpy()
    error = (frame.phenotype - frame.pred_phenotype).abs().to_numpy()[phenotype]
    count = weights[:, phenotype].sum(axis=1)
    stats["phenotype_mae"] = np.divide(weights[:, phenotype] @ error, count,
                                       out=np.full(len(count), np.nan), where=count > 0)
    return stats


def interval(draws):
    resamples = draws[1:][np.isfinite(draws[1:])]
    return {"estimate": float(draws[0]) if np.isfinite(draws[0]) else None,
            "ci95": np.quantile(resamples, [.025, .975]).tolist() if len(resamples) else None,
            "bootstrap_n": len(resamples)}


def summarize_anchor(frame, weights):
    return {"n": len(frame), "phenotype_n": int(frame.phenotype.notna().sum()),
            "anchor_counts": {field: {str(i): int(frame[field].eq(i).sum()) for i in (0, 1)}
                              for field in ANCHORS},
            **{key: interval(draws) for key, draws in anchor_draws(frame, weights).items()}}


def endpoint_contrast(high, low, weights):
    check_pairs(high, low, TARGET_COLS)
    hi, lo = anchor_draws(high, weights), anchor_draws(low, weights)
    return {key: hi[key] - lo[key] for key in ("anchor_ap", "phenotype_mae")}


def tradeoff_pass(contrasts):
    return all(value["estimate"] is not None and value["estimate"] >= .02 and
               value["ci95"] is not None and value["ci95"][0] > 0 for value in contrasts.values())


def hypervolume(points):
    """Union area above (0, 0) for benefit coordinates in [0, 1]."""
    volume, height = 0., 0.
    for x, y in sorted(points, reverse=True):
        if y > height:
            volume += x * (y - height)
            height = y
    return volume


def grade_brier_by_patient(frame):
    residual = frame.active & frame.label.lt(2) & frame.status.lt(2)
    prob = frame.loc[residual, ["p2", "p3"]].to_numpy()
    prob /= prob.sum(axis=1, keepdims=True).clip(1e-12)
    target = np.eye(2)[frame.loc[residual, "label"].to_numpy(dtype=int)]
    scores = np.full(len(frame), np.nan)
    scores[residual] = ((prob - target) ** 2).mean(axis=1)
    return scores


def summarize_grade(frame, repetitions):
    groups = {(model, condition): group.sort_values("subject_id") for (model, condition), group
              in frame.groupby(["model", "condition"], sort=False)}
    reports, contrasts = {}, {}
    for (model, condition), group in groups.items():
        reports[f"{model}/{condition}"] = grade_metrics(group)
    for condition in frame.condition.unique():
        candidate = groups["read_only_residual", condition]
        for model in ("molecular_projected", "learned_joint", "acf"):
            control = groups[model, condition]
            check_pairs(candidate, control, ["label", "active", "image_subject_id"])
            difference = grade_brier_by_patient(candidate) - grade_brier_by_patient(control)
            # Interventions which revoke the rule have zero eligible residual cases.
            usable = difference[np.isfinite(difference)]
            if len(usable):
                draws = patient_draws(len(usable), repetitions)
                contrasts[f"{condition}/{model}"] = {"n": len(usable), **interval(draws @ usable / len(usable))}
    return {"metrics": reports, "residual_brier_candidate_minus_control": contrasts}


def interpolate_predictions(lower, upper, weight, low_weight, high_weight):
    check_pairs(lower, upper, TARGET_COLS)
    ratio = (weight - low_weight) / (high_weight - low_weight)
    interp = lower.copy()
    interp[PROB_COLS] = (1 - ratio) * lower[PROB_COLS].to_numpy() + ratio * upper[PROB_COLS].to_numpy()
    return interp
