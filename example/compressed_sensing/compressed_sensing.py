import numpy as np

import suls


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


def reconstruct(measure_signal: np.ndarray, measure_matrix: np.ndarray,
                inverse_fourier_matrix: np.ndarray) -> np.ndarray:
    """
    reconstruct
    :param measure_signal:
    :param measure_matrix:
    :param inverse_fourier_matrix:
    :return: reconstruct signal
    """
    A = measure_matrix @ inverse_fourier_matrix
    n, m = A.shape
    F = mat_stack([
        [np.real(A), -np.imag(A)],
        [np.imag(A), np.real(A)],
    ]).astype(np.float64)
    x = mat_stack([
        [measure_signal],
        [np.zeros(shape=(n,), dtype=np.float64)],
    ]).flatten()
    y = suls.solve_lp(F, x, 1)
    f = np.empty(shape=(m,), dtype=np.complex128)

    for i in range(m):
        f[i] = complex(y[i], y[i + m])
    reconstruct_signal = inverse_fourier_matrix @ f
    return reconstruct_signal
