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


def reconstruct_real(
        measure_signal: np.ndarray,
        measure_matrix: np.ndarray,
        inverse_transform_matrix: np.ndarray,
) -> np.ndarray:
    """
    reconstruct_complex
    :param measure_signal:
    :param measure_matrix:
    :param inverse_transform_matrix:
    :return: reconstruct signal
    """
    measure_signal = measure_signal.astype(np.float64)
    measure_matrix = measure_matrix.astype(np.float64)
    inverse_transform_matrix = inverse_transform_matrix.astype(np.float64)
    #
    A = measure_matrix @ inverse_transform_matrix
    y = suls.solve_lp(A, measure_signal)
    reconstruct_signal = inverse_transform_matrix @ y
    return reconstruct_signal


def reconstruct_complex(
        measure_signal: np.ndarray,
        measure_matrix: np.ndarray,
        inverse_transform_matrix: np.ndarray,
) -> np.ndarray:
    """
    reconstruct_complex
    :param measure_signal:
    :param measure_matrix:
    :param inverse_transform_matrix:
    :return: reconstruct signal
    """
    measure_signal = measure_signal.astype(np.complex128)
    measure_matrix = measure_matrix.astype(np.complex128)
    inverse_transform_matrix = inverse_transform_matrix.astype(np.complex128)
    #
    A = measure_matrix @ inverse_transform_matrix
    n, m = A.shape
    F = mat_stack([
        [np.real(A), -np.imag(A)],
        [np.imag(A), np.real(A)],
    ]).astype(np.float64)
    x = mat_stack([
        [measure_signal.real],
        [measure_signal.imag],
    ]).flatten()
    y = suls.solve_lp(F, x)
    f = np.empty(shape=(m,), dtype=np.complex128)
    f.real, f.imag = y[0:m], y[m:2*m]
    reconstruct_signal = inverse_transform_matrix @ f
    return reconstruct_signal
