import os
import time

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


def solve_lp(a: np.ndarray, b: np.ndarray, p: float = 1.0) -> np.ndarray:
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
    t2 = time.time()
    solution = sp.optimize.minimize(objective, x0, method="L-BFGS-B", jac=gradient)
    t3 = time.time()
    print(f"model solving.... {t3-t2}s")
    return solution.x


def solve_l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimizer of ||x||_1
    Given Ax=b

    Linear program
    Minimize 1^T y (sum of all elements of y)
    Given y >= |x| and Ax=b
    """
    check_arguments(a, b)
    m, n = a.shape
    t1 = time.time()
    model = pulp.LpProblem(name="", sense=pulp.LpMinimize)
    x = np.array([pulp.LpVariable(name=f"x_{j}", cat=pulp.LpContinuous) for j in range(n)])
    y = np.array([pulp.LpVariable(name=f"y_{j}", cat=pulp.LpContinuous) for j in range(n)])
    for j in range(n):
        model.addConstraint(y[j] >= x[j])
        model.addConstraint(y[j] >= -x[j])
    for i in range(m):
        model.addConstraint(pulp.LpAffineExpression({x[j]: a[i, j] for j in range(n)}) == b[i])
    model.setObjective(pulp.lpSum(y))
    t2 = time.time()
    print(f"model building... {t2-t1}s")
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, mip=False, threads=os.cpu_count(), options=["barrier"]))
    t3 = time.time()
    print(f"model solving.... {t3-t2}s")
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(status)

    return np.array([v.value() for v in x])
