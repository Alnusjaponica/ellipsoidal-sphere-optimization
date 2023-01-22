import copy
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from optimization._timer import stop_watch
from optimization.algorithms.manifold import gen_proj, ret
from optimization.algorithms.parameter import Parameter
from optimization.problem.problem import Problem


def resolve_exception(eta: np.ndarray, p: np.ndarray, Delta: float, f: Callable[[Any], float]) -> np.ndarray:
    a = (p.T @ p)[0, 0]
    b = (eta.T @ p)[0, 0]
    c = (eta.T @ eta - Delta**2)[0, 0]
    tau = (-b + np.sqrt(b**2 - a * c)) / a

    return eta + tau * p


def tCG(grad_m: np.ndarray, hess_m: np.ndarray, Delta: float, f: Callable[[Any], float]) -> np.ndarray:
    eta_i = np.zeros_like(grad_m)
    r_i = copy.deepcopy(grad_m)
    p_i = -copy.deepcopy(grad_m)

    while True:
        pi_i = p_i.T @ hess_m @ p_i
        if pi_i <= 0:
            return resolve_exception(eta_i, p_i, Delta, f)

        alpha_i = r_i.T @ r_i / pi_i
        eta_i += alpha_i * p_i

        if np.linalg.norm(eta_i) >= Delta:
            return resolve_exception(eta_i, p_i, Delta, f)

        esc_ri_sq = r_i.T @ r_i
        r_i += alpha_i * hess_m @ p_i
        if np.linalg.norm(r_i) <= min(0.1 * float(np.linalg.norm(grad_m)), 0.1):
            return eta_i

        beta = r_i.T @ r_i / esc_ri_sq
        p_i = -r_i + beta * p_i


@stop_watch
def trust_region(
    problem: Problem, param: Parameter, x0: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    if x0 is None:
        x0 = problem.generate_x0()
    x_i = x0
    Delta = param.Delta_0

    sols = [x_i]
    y = [problem.f(x_i)]
    grad = []

    for _ in range(param.max_iter):
        # projection matrix
        P, _ = gen_proj(problem, x_i)
        # calculate grad
        g_i = problem.grad(x_i)
        grad_m = P @ g_i

        grad.append(float(np.linalg.norm(grad_m)))
        if np.linalg.norm(grad_m) < param.eps:
            break

        # update x_i
        # ------------------------
        Hi = problem.hess(x_i)
        hess_m = P @ (Hi - x_i.T @ problem.B @ g_i / (x_i.T @ problem.B @ problem.B @ x_i) * problem.B)

        xi_i = tCG(grad_m, hess_m, Delta, problem.f)
        z_i = ret(x_i, problem.B, xi_i)
        numerator = problem.f(x_i) - problem.f(z_i)
        denominator = -grad_m.T @ xi_i - xi_i.T @ hess_m @ xi_i / 2
        rho = numerator / denominator
        if rho < 1 / 4:
            Delta *= 1 / 4
        elif rho > 3 / 4 and abs(np.linalg.norm(xi_i) - Delta) < param.eps:
            Delta *= 2

        if rho > 1 / 8:
            x_i = z_i
        # ------------------------

        sols.append(x_i)
        y.append(problem.f(x_i))

    return sols, y, grad
