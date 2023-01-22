from dataclasses import dataclass


@dataclass
class Parameter:
    """Class for keeping hyper parameters for algorithms"""

    # stopping criteria
    eps: float = 1e-6
    max_iter: int = 10**3

    # parameters for backtracking
    alpha: float = 1.0
    delta: float = 0.2
    rho: float = 1e-1

    # trust region
    Delta_0: float = 1.0

    # adaptive regularized newton
    eta1: float = 0.01
    eta2: float = 0.9
    gamma0: float = 0.2
    gamma1: float = 1.1
    gamma2: float = 10.0
    sigma_0: float = 10.0

    def __post_init__(self) -> None:
        assert 0 <= self.alpha <= 1
        assert 0 < self.delta < 1
        assert 0 < self.rho < 1
        assert 0 < self.Delta_0
        assert 0 < self.eta1 <= self.eta2 < 1
        assert 0 < self.gamma0 < 1 < self.gamma1 <= self.gamma2
        assert 0 < self.sigma_0
