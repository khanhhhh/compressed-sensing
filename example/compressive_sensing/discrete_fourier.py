import numpy as np


def forward(N: int, K: int) -> np.ndarray:
    """
    forward
    :param N: time dimension
    :param K: freq dimension
    :return: transform matrix
    """
    M = np.empty(shape=(K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            M[k, n] = complex(0, - 2 * np.pi * k * n / N)
    return np.exp(M)


def backward(K: int, N: int) -> np.ndarray:
    """
    forward
    :param N: time dimension
    :param K: freq dimension
    :return: transform matrix
    """
    M = np.empty(shape=(N, K), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            M[n, k] = complex(0, + 2 * np.pi * k * n / N)
    return np.exp(M) / N
