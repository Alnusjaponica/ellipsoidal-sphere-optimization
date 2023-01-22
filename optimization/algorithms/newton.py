from typing import List, Optional, Tuple

import numpy as np

from optimization._timer import stop_watch
from optimization.algorithms.manifold import gen_proj, ret
from optimization.algorithms.parameter import Parameter
from optimization.problem.problem import Problem


@stop_watch
def newton(
    problem: Problem, param: Parameter, x0: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    if x0 is None:
        x0 = problem.generate_x0()
    x_i = x0

    sols = [x_i]
    y = [problem.f(x_i)]
    grad = []

    for _ in range(param.max_iter):
        # projection matrix
        P, Q = gen_proj(problem, x_i)
        # calculate grad
        g_i = problem.grad(x_i)
        grad_m = P @ g_i

        grad.append(float(np.linalg.norm(grad_m)))
        if np.linalg.norm(grad_m) < param.eps:
            break

        # update x_i
        # ------------------------
        # calculate hess without projection
        H_i = problem.hess(x_i) - x_i.T @ problem.B @ g_i / (x_i.T @ problem.B @ problem.B @ x_i) * problem.B

        # calculate alpha := Q H^-1
        numerator = Q @ np.linalg.solve(H_i, g_i)
        denominator = Q @ np.linalg.solve(H_i, problem.B @ x_i)
        alpha = numerator[0] / denominator[0]

        # calculate xi_i := H^-1 (alpha * Bx - Df(x))
        xi_i = np.linalg.solve(H_i, alpha[0] * problem.B @ x_i - g_i)
        x_i = ret(x_i, problem.B, xi_i)
        # ------------------------

        sols.append(x_i)
        y.append(problem.f(x_i))

    return sols, y, grad
