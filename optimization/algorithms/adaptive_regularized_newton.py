import copy
from typing import List, Optional, Tuple

import numpy as np

from optimization._timer import stop_watch
from optimization.algorithms.manifold import gen_proj, ret
from optimization.algorithms.parameter import Parameter
from optimization.problem.problem import Problem


def MtCG(grad_m: np.ndarray, hess_m: np.ndarray) -> np.ndarray:
    T, theta, eps = 0.01, 1.0, 0.0
    eta_i = np.zeros_like(grad_m)
    r_i = -copy.deepcopy(grad_m)
    p_i = -copy.deepcopy(grad_m)

    for i in range(len(grad_m)):
        pi_i = p_i.T @ hess_m @ p_i
        curvature = pi_i / (p_i.T @ p_i)
        if curvature <= eps:
            s_k = -grad_m if i == 0 else eta_i
            if curvature <= -eps:
                tau_k = p_i.T @ grad_m / pi_i
                return s_k + tau_k * p_i
            return s_k

        alpha_i = r_i.T @ r_i / pi_i
        eta_i += alpha_i * p_i
        esc_ri_sq = r_i.T @ r_i
        r_i -= alpha_i * hess_m @ p_i
        if np.linalg.norm(r_i) <= min(float(np.linalg.norm(grad_m) ** theta), T):
            return eta_i

        beta = r_i.T @ r_i / esc_ri_sq
        p_i = beta * p_i + r_i
    return eta_i


@stop_watch
def adaptive_regularized_newton(
    problem: Problem, param: Parameter, x0: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    if x0 is None:
        x0 = problem.generate_x0()
    x_i = x0
    sigma = param.sigma_0

    sols = [x_i]
    y = [problem.f(x_i)]
    grad = []

    for _ in range(param.max_iter):
        # calculate grad
        P, _ = gen_proj(problem, x_i)
        g_i = problem.grad(x_i)
        grad_m = P @ g_i

        grad.append(float(np.linalg.norm(grad_m)))
        if np.linalg.norm(grad_m) < param.eps:
            break

        # update x_i
        # ------------------------
        # solve subproblem approximately
        Hi = problem.hess(x_i)
        hess_m = P @ (
            Hi - x_i.T @ problem.B @ g_i / (x_i.T @ problem.B @ problem.B @ x_i) * problem.B
        ) + sigma * np.eye(problem.n)

        xi_i = MtCG(grad_m, hess_m)

        # curve search by backtracking
        alpha = param.alpha
        rhs = param.rho * grad_m.T @ xi_i
        z_i = ret(x_i, problem.B, alpha * xi_i)
        m_zi = (z_i - x_i).T @ (Hi + sigma * np.eye(problem.n)) @ (z_i - x_i) / 2 + g_i.T @ (z_i - x_i)
        while m_zi > alpha * rhs:
            alpha *= param.delta
            z_i = ret(x_i, problem.B, alpha * xi_i)
            m_zi = (z_i - x_i).T @ (Hi + sigma * np.eye(problem.n)) @ (z_i - x_i) / 2 + g_i.T @ (z_i - x_i)

        rho = (problem.f(z_i) - problem.f(x_i)) / m_zi
        if rho < param.eta1:
            sigma *= param.gamma2
        else:
            x_i = z_i
            if rho >= param.eta2:
                sigma *= param.gamma0
            else:
                sigma *= param.gamma1
        # ------------------------

        sols.append(x_i)
        y.append(problem.f(x_i))

    return sols, y, grad
