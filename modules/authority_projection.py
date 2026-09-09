"""Exact task projections and an optional differentiable general convex projection."""

from functools import lru_cache

import torch
import torch.nn as nn


def project_allowed_simplex(p, allowed):
    """Euclidean projection onto the simplex restricted to admitted classes."""
    if bool((~allowed.any(-1)).any()):
        raise ValueError("Empty allowed simplex")
    values = p.masked_fill(~allowed, -torch.inf)
    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1)
    rank = torch.arange(1, p.shape[-1] + 1, device=p.device, dtype=p.dtype)
    keep = ordered - (cumulative - 1.) / rank > 0
    rho = keep.sum(-1).clamp_min(1)
    threshold = ((cumulative.gather(-1, (rho - 1).unsqueeze(-1)).squeeze(-1) - 1.) / rho)
    return (p - threshold.unsqueeze(-1)).clamp_min(0).masked_fill(~allowed, 0.)


def project_disjoint_orders(p, directions, active):
    result = p.clamp(0, 1)
    parts = []
    for r in range(directions.shape[-1]):
        # Pool the original coordinates before clipping: clipping first is wrong
        # for points outside the box, even though sigmoid predictions are inside.
        left, right = p[:, 2*r], p[:, 2*r+1]
        violates = torch.where(directions[:, r].bool(), left < right, left > right) & active[:, r]
        average = ((left + right) / 2).clamp(0, 1)
        parts.extend([torch.where(violates, average, left.clamp(0, 1)),
                      torch.where(violates, average, right.clamp(0, 1))])
    parts.extend(result[:, k] for k in range(2 * directions.shape[-1], result.shape[-1]))
    return torch.stack(parts, -1)


def project_registered(p, compiled):
    # Conflicting/no-input rows never enter a solver and are never counted as predictions.
    if compiled.task == "s1":
        result = project_allowed_simplex(p, compiled.allowed_classes)
    else:
        result = project_disjoint_orders(p, compiled.directions, compiled.relation_active[:, :2])
    return torch.where((compiled.status < 2)[:, None], result, p)


@lru_cache(maxsize=16)
def _convex_layer(classes, rows, simplex):
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except ImportError as error:
        raise RuntimeError("General constraints require requirements-authority.txt") from error
    variable = cp.Variable(classes)
    initial = cp.Parameter(classes)
    B, b = cp.Parameter((rows, classes)), cp.Parameter(rows)
    constraints = [variable >= 0, variable <= 1, B @ variable >= b]
    if simplex:
        constraints.append(cp.sum(variable) == 1)
    problem = cp.Problem(cp.Minimize(.5 * cp.sum_squares(variable - initial)), constraints)
    if not problem.is_dpp():
        raise RuntimeError("Authority projection is not DPP compliant")
    return CvxpyLayer(problem, parameters=[initial, B, b], variables=[variable])


class GeneralAuthorityProjection(nn.Module):
    """Return per-row statuses: 0 solved, 2 infeasible, 4 solver failure.

    The compiled sparse task projections avoid this general CPU solver in the main grid.
    """

    def forward(self, p, B, b, simplex=True):
        from scipy.optimize import linprog
        try:
            from diffcp.cone_program import SolverError
            numerical_errors = (RuntimeError, ValueError, SolverError)
        except ImportError:
            numerical_errors = (RuntimeError, ValueError)
        layer = _convex_layer(p.shape[-1], B.shape[-2], simplex)
        outputs, statuses = [], []
        for i in range(len(p)):
            matrix, bound = B[i].detach().cpu().double(), b[i].detach().cpu().double()
            try:
                feasibility = linprog([0.] * p.shape[-1], A_ub=-matrix.numpy(), b_ub=-bound.numpy(),
                                      A_eq=[[1.] * p.shape[-1]] if simplex else None,
                                      b_eq=[1.] if simplex else None, bounds=(0, 1), method="highs")
            except (RuntimeError, ValueError):
                outputs.append(p[i]); statuses.append(4); continue
            if feasibility.status == 2:
                outputs.append(p[i]); statuses.append(2); continue
            if not feasibility.success:
                outputs.append(p[i]); statuses.append(4); continue
            try:
                solved, = layer(p[i].cpu().double(), B[i].cpu().double(), b[i].cpu().double(),
                                solver_args={"eps": 1e-8, "max_iters": 10000})
                residual = (bound - matrix @ solved.detach()).clamp_min(0).max()
                domain_error = max(float((-solved).clamp_min(0).max()), float((solved-1).clamp_min(0).max()))
                sum_error = abs(float(solved.sum())-1) if simplex else 0
                if not bool(torch.isfinite(solved).all()) or max(float(residual), domain_error, sum_error) > 1e-6:
                    raise RuntimeError("Projection did not reach feasibility tolerance")
                outputs.append(solved.to(p)); statuses.append(0)
            except numerical_errors:
                outputs.append(p[i]); statuses.append(4)
        return torch.stack(outputs), torch.tensor(statuses, device=p.device)
