import numpy as np
import sympy as sy


class Problem:
    """Objective function and constraint"""

    def __init__(self, x: sy.Matrix, func: sy.Symbol, mu: float = -1, seed: float = 42, radius: float = 1.5) -> None:
        self.x = x
        self.n = x.shape[0]
        self.mu = mu
        self.seed = seed
        self.r = radius

        o = sy.Matrix([0.0] * self.n)  # type: ignore[no-untyped-call]
        f0 = func.subs(zip(self.x, o))  # type: ignore[no-untyped-call]

        # set f: R^n -> R
        self._f = sy.lambdify((self.x), func - f0, "numpy")

        # set h: R^n -> R
        H = np.array(sy.hessian(func, self.x).subs(zip(self.x, o)), dtype="float")  # type: ignore[no-untyped-call]
        self.B = H / (self.r**2)
        h_ = (sy.transpose(self.x) * H * self.x / 2)[0, 0]
        self._h = sy.lambdify((self.x), h_, "numpy")

        # set objective function F := f - h and its grad and hess
        F_ = func + mu * h_
        self._F = sy.lambdify((self.x), F_, "numpy")

        grad_ = sy.diff(F_, self.x)  # type: ignore[no-untyped-call]
        self._grad = sy.lambdify((self.x), grad_, "numpy")

        hess_ = sy.hessian(F_, self.x)  # type: ignore[no-untyped-call]
        self._hess = sy.lambdify((self.x), hess_, "numpy")

    def generate_x0(self) -> np.ndarray:
        np.random.seed(self.seed)
        _x0 = np.random.rand(self.n, 1)
        x0 = self.normalize(_x0)
        return x0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x / np.sqrt(x.T @ self.B @ x)

    def f(self, x: np.ndarray) -> float:
        return self._f(*x.T[0])

    def h(self, x: np.ndarray) -> np.ndarray:
        return self._h(*x.T[0])

    def F(self, x: np.ndarray) -> float:
        return self._F(*x.T[0])

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._grad(*x.T[0])

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self._hess(*x.T[0])
