import time

import matplotlib.pyplot as plt
import numpy as np

from suls import solve_lp, solve_l1

np.random.seed(1234)


def draw_hist(x: np.ndarray, title: str = "norm", draw: bool = False):
    hist, edge = np.histogram(x, bins=101, range=[-0.1, +0.1])
    center = np.array([0.5 * (edge[i] + edge[i + 1]) for i in range(len(hist))])
    print(f"middle [{edge[int(len(hist) / 2)]}, {edge[1 + int(len(hist) / 2)]}] occurences: {hist[int(len(hist) / 2)]}")
    if draw:
        plt.title(title)
        plt.xlabel("values")
        plt.ylabel("occurrences")
        plt.bar(center, hist, width=(center[1] - center[0]))
        plt.show()


def norm_p(x: np.ndarray, p: float = 2.0) -> float:
    return np.sum(np.abs(x) ** p) ** (1 / p)


n = 1000
m = 200
A = np.random.random(size=(m, n)).astype(dtype=np.float64)
b = np.random.random(size=(m)).astype(dtype=np.float64)

for p in [1, 1.000001, 1.5, 2.0]:
    t0 = time.time()
    if p == 1:
        x = solve_l1(A, b)
    else:
        x = solve_lp(A, b, p)
    t1 = time.time()
    print(f"L^{p} time: {t1 - t0}")
    print(f"\tconstraints: {np.max(np.abs(A @ x - b))}")
    print(f"\tL^p norm: {norm_p(x, p)}")
    draw_hist(x, f"L^{p} norm", draw=True)

pass
