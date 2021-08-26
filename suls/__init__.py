from typing import List

import numpy as np
import pulp
import scipy as sp
import scipy.optimize


def mat_stack(mat: List[List[np.ndarray]]) -> np.ndarray:
    """
    [[A, B],
     [C, D]]
    """
    row_list = []
    for mat_row in mat:
        row = np.hstack(mat_row)
        row_list.append(row)
    out = np.vstack(row_list)
    return out


def vec_stack(vec: List[np.ndarray]) -> np.ndarray:
    """
    [A, B]
    """
    return np.hstack(vec)


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

    c = vec_stack([
        np.zeros(shape=(n,)), np.ones(shape=(n,))
    ])

    a_ub = mat_stack([
        [+np.identity(n), -np.identity(n)],
        [-np.identity(n), -np.identity(n)]
    ])
    b_ub = np.zeros(shape=(2 * n))

    a_eq = mat_stack([
        [a, np.zeros(shape=(m, n))]
    ])
    b_eq = b

    x1 = linprog(c=c, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq)
    return x1[0:n]


def linprog(c: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray, a_eq: np.ndarray, b_eq: np.ndarray):
    """
    linprog: solve linear program
    minimize c^T x
    such that
    a_ub x \leq b_ub
    a_eq x = b_eq
    """
    model = pulp.LpProblem(name="", sense=pulp.LpMinimize)
    # create variable
    n = c.shape[0]
    x = np.array([pulp.LpVariable(name=f"x_{i}]", cat=pulp.LpContinuous) for i in range(n)])
    # add ub constraint
    m_ub, n = a_ub.shape
    for i in range(m_ub):
        model.addConstraint(pulp.lpSum(a_ub[i, :] * x) <= b_ub[i])
    # add eq constraint
    m_eq, n = a_eq.shape
    for i in range(m_eq):
        model.addConstraint(pulp.lpSum(a_eq[i, :] * x) == b_eq[i])
    # add objective
    model.setObjective(pulp.lpSum(c * x))
    # solve
    status = model.solve()

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(status)

    return np.array([v.value() for v in x])
