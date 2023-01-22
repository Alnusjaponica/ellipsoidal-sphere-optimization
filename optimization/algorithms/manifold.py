from typing import Optional, Tuple

import numpy as np

from optimization.problem.problem import Problem


def ret(x: np.ndarray, B: np.ndarray, xi: np.ndarray) -> np.ndarray:
    zi = xi + x
    return zi / np.sqrt(zi.T @ B @ zi)


def gen_proj(problem: Problem, x_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Q = problem.B @ x_i @ x_i.T @ problem.B / (x_i.T @ problem.B @ problem.B @ x_i)
    P = np.eye(problem.n) - Q
    return P, Q


def mgrad(problem: Problem, x_i: np.ndarray, P: Optional[np.ndarray] = None) -> np.ndarray:
    if P is None:
        P, _ = gen_proj(problem, x_i)
    gi = problem.grad(x_i)
    grad_m = P @ gi
    return grad_m
