import time

import matplotlib.pyplot as plt
import numpy as np

from suls import solve_lp, solve_l1

np.random.seed(1234)


def draw_hist(ax: plt.Axes, x: np.ndarray, title: str = ""):
    hist, edge = np.histogram(x, bins=101, range=[-0.1, +0.1])
    center = np.array([0.5 * (edge[i] + edge[i + 1]) for i in range(len(hist))])
    print(f"middle [{edge[int(len(hist) / 2)]}, {edge[1 + int(len(hist) / 2)]}] occurences: {hist[int(len(hist) / 2)]}")
    ax.bar(center, hist, width=(center[1] - center[0]))
    ax.set_title(title)
    ax.set_xlabel("values")
    ax.set_ylabel("occurrences")
    ax.set_title(title)


def norm_p(x: np.ndarray, p: float = 2.0) -> float:
    return np.sum(np.abs(x) ** p) ** (1 / p)


if __name__ == "__main__":
    n = 2000
    m = 400
    A = np.random.random(size=(m, n)).astype(dtype=np.float64)
    b = np.random.random(size=(m, )).astype(dtype=np.float64)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax = ax.flatten()

    ax_i = 0

    t0 = time.time()
    x = solve_l1(A, b)
    t1 = time.time()
    print(f"L^{1} time: {t1 - t0}")
    print(f"\tconstraints: {np.max(np.abs(A @ x - b))}")
    print(f"\tL^{1} norm: {norm_p(x, 1)}")
    draw_hist(ax[ax_i], x, f"L^{1} norm")
    ax_i += 1

    for p in [1, 2]:
        t0 = time.time()
        x = solve_lp(A, b, p)
        t1 = time.time()
        print(f"L^{p} time: {t1 - t0}")
        print(f"\tconstraints: {np.max(np.abs(A @ x - b))}")
        print(f"\tL^{p} norm: {norm_p(x, p)}")
        draw_hist(ax[ax_i], x, f"L^{p} norm")
        ax_i += 1

    plt.tight_layout()
    plt.show()
