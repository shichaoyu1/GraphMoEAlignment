"""Risk curves, fixed-reference hypervolume and exact small-task frontiers."""

import numpy as np
from scipy.optimize import linprog


def nondominated(points):
    points = np.unique(np.asarray(points), axis=0)
    return np.array([p for p in points if not np.any(np.all(points <= p + 1e-12, 1)
                                                   & np.any(points < p - 1e-12, 1))])


def hypervolume(points, reference=(1.01, 1.01)):
    front = nondominated(points)
    area, height = 0., reference[1]
    for x, y in sorted(front.tolist()):
        if x < reference[0] and y < height:
            area += (reference[0] - x) * (height - y)
            height = y
    return float(area)


def exact_oracle(environment, preferences):
    table = environment.risk_table.cpu().numpy()
    deterministic = nondominated(((table[0, :, None] + table[1, None]) / 2).reshape(-1, 2))
    mixed = []
    actions = table.shape[1]
    for weight in preferences:
        equality = np.zeros((2, 2 * actions + 1))
        equality[0, :actions] = 1
        equality[1, actions:2 * actions] = 1
        inequality = np.zeros_like(equality)
        inequality[:, :-1] = (table.reshape(-1, 2) * [.5 * weight, .5 * (1 - weight)]).T
        inequality[:, -1] = -1
        cost = np.zeros(2 * actions + 1)
        cost[-1] = 1
        solution = linprog(cost, A_ub=inequality, b_ub=np.zeros(2), A_eq=equality,
                           b_eq=np.ones(2), bounds=(0, None), method='highs')
        if not solution.success:
            raise RuntimeError(solution.message)
        mixed.append((solution.x[:-1] @ table.reshape(-1, 2) / 2).tolist())
    return {'deterministic_front': deterministic.tolist(), 'randomized_tchebycheff': mixed}
