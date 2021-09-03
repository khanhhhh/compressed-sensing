import numpy as np


def create_measure_matrix(d: int, n: int) -> np.ndarray:
    """
    create_measure_matrix
    :param d: number of measure signal data points
    :param n: number of true signal data points
    :return: measure matrix
    """
    M = np.zeros(shape=(d, n), dtype=np.int64)
    indices = np.random.choice(np.arange(n), size=d, replace=False)
    for i in range(d):
        M[i, indices[i]] = 1
    return M
