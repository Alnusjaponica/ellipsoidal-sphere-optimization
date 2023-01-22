import sympy as sy

from optimization.problem.problem import Problem


def generate_Himmelblau(mu: float = -1, seed: float = 42, radius: float = 1.5) -> Problem:
    n = 2
    xc = sy.Matrix([-2.805118, 3.131312])  # type: ignore[no-untyped-call]
    x = sy.Matrix(sy.symbols(f"x1:{n+1}", real=True))  # type: ignore[no-untyped-call]
    func = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    return Problem(x, func.subs(zip(x, x + xc)), mu=mu, seed=seed, radius=radius)


def generate_Beale(mu: float = -1, seed: float = 42, radius: float = 1.5) -> Problem:
    n = 2
    x = sy.Matrix(sy.symbols(f"x1:{n+1}", real=True))  # type: ignore[no-untyped-call]
    func = (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )
    xc = sy.Matrix([3, 0.5])  # type: ignore[no-untyped-call]

    return Problem(x, func.subs(zip(x, x + xc)), mu=mu, seed=seed, radius=radius)


def generate_Rosenbrock(n: int = 2, mu: float = -1, seed: float = 42, radius: float = 1.5) -> Problem:
    x = sy.Matrix(sy.symbols(f"x1:{n+1}", real=True))  # type: ignore[no-untyped-call]
    xc = sy.Matrix([1 for _ in range(n)])  # type: ignore[no-untyped-call]
    func = sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(n - 1)])

    return Problem(x, func.subs(zip(x, x + xc)), mu=mu, seed=seed, radius=radius)
