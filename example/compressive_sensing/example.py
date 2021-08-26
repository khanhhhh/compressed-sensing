from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

import discrete_fourier
import suls

np.random.seed(1234)


def mat_stack(mat: list[list[np.ndarray]]) -> np.ndarray:
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


def sine(T: float) -> Callable[[np.ndarray], np.ndarray]:
    def sine_(t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * t / T)

    return sine_


def create_signal(t: np.ndarray) -> np.ndarray:
    return sine(9)(t) + 2 * sine(3)(t + 2) - 3 * sine(1)(t - 0.5)


def plot_signal(*args: np.ndarray):
    for i in range(0, len(args), 2):
        if i + 1 >= len(args):
            break
        t = args[i]
        f = args[i + 1]
        argsort = np.argsort(t)
        plt.plot(t[argsort], f[argsort], linewidth=1, marker="o", markersize=3)
    plt.show()


def create_sensing_matrix(D: int, N: int) -> np.ndarray:
    """
    create_sensing_matrix
    :param D: number of measure signal data points
    :param N: number of true signal data points
    :return: sensing matrix
    """
    M = np.zeros(shape=(D, N), dtype=np.int64)
    indices = np.random.choice(np.arange(N), size=D, replace=False)
    for d in range(D):
        M[d, indices[d]] = 1
    return M


def reconstruct(measure_signal: np.ndarray, sensing_matrix: np.ndarray,
                inverse_fourier_matrix: np.ndarray) -> np.ndarray:
    """
    reconstruct
    :param measure_signal:
    :param sensing_matrix:
    :param inverse_fourier_matrix:
    :return: reconstruct signal
    """
    A = sensing_matrix @ inverse_fourier_matrix
    n, m = A.shape
    F = mat_stack([
        [np.real(A), -np.imag(A)],
        [np.imag(A), np.real(A)],
    ]).astype(np.float64)
    x = mat_stack([
        [measure_signal],
        [np.zeros(shape=(n,), dtype=np.float64)],
    ]).flatten()
    y = suls.solve_l1(F, x)
    f = np.empty(shape=(m,), dtype=np.complex128)

    for i in range(m):
        f[i] = complex(y[i], y[i + m])
    reconstruct_signal = inverse_fourier_matrix @ f
    return reconstruct_signal


sampling_rate = 0.05
t = np.arange(-3, +3, sampling_rate)

true_signal = create_signal(t)

plot_signal(t, true_signal)

N = len(t)
K = 3 * N

fourier = discrete_fourier.forward(N, K)
inverse = discrete_fourier.backward(K, N)

D = int(0.3 * N)  # 30% of signal
sensing_matrix = create_sensing_matrix(D, N)
measure_signal = sensing_matrix @ true_signal
measure_t = sensing_matrix @ t
plot_signal(t, true_signal, measure_t, measure_signal)

reconstruct_signal = reconstruct(measure_signal=measure_signal, sensing_matrix=sensing_matrix,
                                 inverse_fourier_matrix=inverse)
print("max diff: ", np.max(np.abs(reconstruct_signal)))

plot_signal(t, true_signal, t, reconstruct_signal)
