"""Diagnostics that separate Paper 4 geometry, communication, and allocation effects."""

import numpy as np


def diagnostic_gains(
    scores,
    *,
    learned="spd_cross_graph",
    uniform="spd_uniform_graph",
    identity="spd_identity_graph",
    euclidean="euclidean_cross_graph",
):
    """Return matched score contrasts; unavailable contrasts remain absent."""
    result = {}
    if learned in scores and euclidean in scores:
        result["geometry_gain"] = float(scores[learned] - scores[euclidean])
    if uniform in scores and identity in scores:
        result["communication_gain"] = float(scores[uniform] - scores[identity])
    if learned in scores and uniform in scores:
        result["allocation_gain"] = float(scores[learned] - scores[uniform])
    return result


def coadaptation_diagnostics(trained_base, base_under_alternative, trained_alternative):
    """Contrast test-time replacement with end-to-end retraining."""
    return {
        "intervention_cost": float(trained_base - base_under_alternative),
        "retrained_advantage": float(trained_alternative - trained_base),
        "coadaptation_gap": float(trained_alternative - base_under_alternative),
    }


def _offdiagonal(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    row, col = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row, col]


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = _rank(scores)
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - positive.sum() * (positive.sum() + 1) / 2)
        / (positive.sum() * negative.sum())
    )


def _binary_auprc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if not (labels == 1).any():
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ordered = labels[order] == 1
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def edge_recovery(adjacency, planted_adjacency):
    """Score undirected learned edges against a planted binary graph."""
    scores = _offdiagonal(0.5 * (np.asarray(adjacency) + np.asarray(adjacency).T))
    labels = (_offdiagonal(planted_adjacency) > 0).astype(int)
    score_ranks = _rank(scores)
    label_ranks = _rank(labels)
    correlation = np.corrcoef(score_ranks, label_ranks)[0, 1]
    return {
        "edge_auroc": _binary_auroc(labels, scores),
        "edge_auprc": _binary_auprc(labels, scores),
        "edge_spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
    }


def edge_stability(adjacencies, top_k=3):
    """Measure pairwise edge-rank correlation and top-k overlap."""
    vectors = [_offdiagonal(0.5 * (np.asarray(value) + np.asarray(value).T)) for value in adjacencies]
    correlations = []
    overlaps = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            left = vectors[left_index]
            right = vectors[right_index]
            correlation = np.corrcoef(_rank(left), _rank(right))[0, 1]
            if np.isfinite(correlation):
                correlations.append(float(correlation))
            count = min(int(top_k), len(left))
            left_top = set(np.argsort(-left)[:count].tolist())
            right_top = set(np.argsort(-right)[:count].tolist())
            union = left_top | right_top
            overlaps.append(float(len(left_top & right_top) / len(union)) if union else 1.0)
    return {
        "pair_count": len(overlaps),
        "edge_spearman_mean": float(np.mean(correlations)) if correlations else float("nan"),
        "topk_jaccard_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
    }


def hierarchical_bootstrap_mean(values_by_split, iterations=2000, seed=0):
    """Bootstrap split seeds first and model seeds second."""
    groups = [np.asarray(values, dtype=float) for values in values_by_split if len(values)]
    if not groups:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(iterations)):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        values = []
        for index in selected_groups:
            group = groups[index]
            values.extend(group[rng.integers(0, len(group), size=len(group))].tolist())
        samples.append(float(np.mean(values)))
    return {
        "mean": float(np.mean(np.concatenate(groups))),
        "ci95": np.quantile(samples, [0.025, 0.975]).astype(float).tolist(),
    }


__all__ = [
    "diagnostic_gains",
    "coadaptation_diagnostics",
    "edge_recovery",
    "edge_stability",
    "hierarchical_bootstrap_mean",
]
