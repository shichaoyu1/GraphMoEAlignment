"""Fit bounded MRI mixtures to development patients at a fixed swap-risk budget."""

import numpy as np
from scipy.optimize import lsq_linear


BUDGETS = {"0": 0., "0.01": .01, "0.02": .02, "unlimited": None}
READ_ONLY = "read_only_residual"
MIXTURES = {
    "constant": READ_ONLY, "adaptive": READ_ONLY,
    "free_constant": "learned_joint", "acf_constant": "acf",
}


def _bounded_fit(x, y, tie_x, tie_y):
    """Least squares on the unit box, then minimize the second loss among ties."""
    if not x.any():
        if not tie_x.any():
            return np.zeros(x.shape[1])
        x, y, tie_x, tie_y = tie_x, tie_y, x, y
    weights = lsq_linear(x, y, bounds=(0., 1.), method="bvls", tol=1e-13).x.clip(0, 1)
    if x.shape[1] == 1 or np.linalg.matrix_rank(x) == 2:
        return weights

    # With one observed disagreement level, the two weights have one free direction.
    row = x[np.argmax(np.linalg.norm(x, axis=1))]
    direction = np.array([-row[1], row[0]]) / np.linalg.norm(row)
    movable = direction != 0
    ends = np.array([-weights[movable], 1 - weights[movable]]) / direction[movable]
    lower, upper = ends.min(axis=0).max(), ends.max(axis=0).min()
    slope = tie_x @ direction
    if np.linalg.norm(slope) > np.finfo(float).eps * max(tie_x.shape) * np.linalg.norm(tie_x):
        step = slope @ (tie_y - tie_x @ weights) / (slope @ slope)
    else:
        # An equal-coefficient solution has the same predictions on both sets.
        step = (weights[1] - weights[0]) / (direction[0] - direction[1])
    return (weights + np.clip(step, lower, upper) * direction).clip(0, 1)


def _design(rows, source, adaptive):
    delta = (rows[source] - rows.molecular_projected).to_numpy()
    x = delta[:, None]
    if adaptive:
        d = abs(delta)
        x = np.column_stack((delta * (1 - d), delta * d))
    # One vote per recipient, irrespective of how many donors that recipient has.
    counts = rows.groupby("subject_id").subject_id.transform("size").to_numpy()
    scale = np.sqrt(counts * rows.subject_id.nunique())
    return x / scale[:, None], (rows.label - rows.molecular_projected).to_numpy() / scale


def fit_mixture(own, swap, budget):
    """Minimize own Brier subject to swap Brier <= report Brier + budget.

    For a binding budget, bisect the relative weight of the two convex least
    squares losses. Keep the feasible side, including the minimum-swap endpoint.
    The arguments are patient-weighted (design, target) pairs from the same fold.
    """
    x, y = own
    z, t = swap

    def excess(weights):
        change = z @ weights
        return change @ (change - 2 * t)

    best_own = _bounded_fit(x, y, z, t)
    if budget is None or excess(best_own) <= budget:
        return best_own
    feasible = _bounded_fit(z, t, x, y)
    minimum_excess = excess(feasible)
    if minimum_excess >= budget:
        # A zero budget can expose only the minimum-swap face. Its ties were
        # already broken by own risk. Zero is always feasible in exact arithmetic.
        return feasible if minimum_excess == budget else np.zeros_like(feasible)
    low, high = 0., 1.
    for _ in range(48):
        weight = (low + high) / 2
        a, b = np.sqrt(1 - weight), np.sqrt(weight)
        trial = _bounded_fit(np.vstack((a * x, b * z)), np.r_[a * y, b * t], z, t)
        if excess(trial) <= budget:
            feasible, high = trial, weight
        else:
            low = weight
    return feasible


def fit_development(rows):
    """Fit every budget using the checkpoint's own development fold only."""
    own, donor = rows.condition.eq("own"), rows.condition.eq("donor")
    fits = []
    for method, source in MIXTURES.items():
        adaptive = method == "adaptive"
        clean = _design(rows[own], source, adaptive)
        swap = _design(rows[donor], source, adaptive)
        for name, budget in BUDGETS.items():
            weights = fit_mixture(clean, swap, budget)
            fits.append(dict(method=method, source=source, budget=name,
                             a=float(weights[0]), b=float(weights[-1]),
                             dev_own_brier=float(np.sum((clean[0] @ weights - clean[1]) ** 2)),
                             dev_swap_brier=float(np.sum((swap[0] @ weights - swap[1]) ** 2)),
                             dev_report_brier=float(clean[1] @ clean[1])))
    return fits
