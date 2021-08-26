import numpy as np
import pulp
import scipy as sp
import scipy.optimize


def check_arguments(a: np.ndarray, b: np.ndarray):
    if len(a.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = a.shape

    if not (m < n):
        raise Exception("System must be under-determined (m < n)")


def solve_lp(a: np.ndarray, b: np.ndarray, p: float = 2.0) -> np.ndarray:
    """
    Minimizer of ||x||_p^p
    Given Ax=b
    By minimizing ||Ax-b||_2^2 + ||x||_p^p
    """

    check_arguments(a, b)
    m, n = a.shape

    def objective(x: np.ndarray) -> np.ndarray:
        return np.sum((a @ x - b) ** 2) + np.sum(np.abs(x) ** p)

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2 * a.T @ (a @ x - b) + p * np.sign(x) * np.abs(x) ** (p - 1)

    x0 = np.zeros(shape=(n,))
    solution = sp.optimize.minimize(objective, x0, method="L-BFGS-B", jac=gradient)
    return solution.x


def solve_l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimizer of ||Ax+b||_1 as a linear program
    """
    check_arguments(a, b)
    m, n = a.shape

    model = pulp.LpProblem(name="", sense=pulp.LpMinimize)
    x = np.array([pulp.LpVariable(name=f"x_{i}", cat=pulp.LpContinuous) for i in range(n)])
    y = np.array([pulp.LpVariable(name=f"y_{i}", cat=pulp.LpContinuous) for i in range(n)])
    for i in range(n):
        model.addConstraint(y[i] >= x[i])
        model.addConstraint(y[i] >= -x[i])
    for i in range(m):
        model.addConstraint(pulp.lpSum(a[i, :] * x) == b[i])
    model.setObjective(pulp.lpSum(y))
    status = model.solve()

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(status)

    return np.array([v.value() for v in x])
