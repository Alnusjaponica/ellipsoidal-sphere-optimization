from typing import List, Optional, Tuple

import numpy as np

from optimization._timer import stop_watch
from optimization.algorithms.manifold import mgrad, ret
from optimization.algorithms.parameter import Parameter
from optimization.problem.problem import Problem


@stop_watch
def gradient_descent(
    problem: Problem, param: Parameter, x0: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    if x0 is None:
        x0 = problem.generate_x0()
    x_i = x0

    sols = [x_i]
    y = [problem.f(x_i)]
    grad = []

    for _ in range(param.max_iter):
        grad_m = mgrad(problem, x_i)
        grad.append(float(np.linalg.norm(grad_m)))
        if np.linalg.norm(grad_m) < param.eps:
            break

        # update x_i
        # ------------------------
        z_i = -grad_m

        alpha = param.alpha
        rhs = param.rho * grad_m.T @ z_i
        f_x = problem.f(x_i)
        while problem.f(ret(x_i, problem.B, alpha * z_i)) > f_x + alpha * rhs:
            alpha *= param.delta
        x_i = ret(x_i, problem.B, alpha * z_i)
        # ------------------------

        sols.append(x_i)
        y.append(problem.f(x_i))

    return sols, y, grad
