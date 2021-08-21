import numpy as np
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

    a_ub = np.empty(shape=(2 * n, 2 * n))
    a_ub[0:n, 0:n] = +np.identity(n)
    a_ub[n:2 * n, 0:n] = -np.identity(n)
    a_ub[0:n, n:2 * n] = -np.identity(n)
    a_ub[n:2 * n, n:2 * n] = -np.identity(n)
    b_ub = np.zeros(shape=(2 * n))

    c = np.empty(shape=(2 * n))
    c[0:n] = 0
    c[n:2 * n] = 1

    a_eq = np.empty(shape=(m, 2 * n))
    a_eq[:, 0:n] = a
    a_eq[:, n:2 * n] = 0
    b_eq = b

    solution = sp.optimize.linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None) for _ in range(2 * n)],
        method="simplex",
    )
    x1 = solution.x
    return x1[0:n]
