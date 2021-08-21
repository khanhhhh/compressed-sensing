from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import suls

np.random.seed(1234)


def forward_transform(N: int, K: int) -> np.ndarray:
    M = np.empty(shape=(K, N), dtype=np.complex128)
    for k in range(K):
        for n in range(N):
            M[k, n] = complex(0, - 2 * np.pi * k * n / N)
    return np.exp(M)


def backward_transform(K: int, N: int) -> np.ndarray:
    M = np.empty(shape=(N, K), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            M[n, k] = complex(0, + 2 * np.pi * k * n / N)
    return np.exp(M) / N


# signal
def sine(T: float) -> Callable[[np.ndarray], np.ndarray]:
    def sine_(t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * t / T)

    return sine_


def square(T: float) -> Callable[[np.ndarray], np.ndarray]:
    def rounddown(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.int64)
        y[x < 0] -= 1
        return y

    def square_(t: np.ndarray) -> np.ndarray:
        return rounddown(2 * t / T) % 2

    return square_


def ramp(T: float) -> Callable[[np.ndarray], np.ndarray]:
    def remdiv(a_: np.ndarray, b: float) -> np.ndarray:
        a = np.copy(a_)
        while True:
            if np.sum(a < 0) == 0 and np.sum(a >= b) == 0:
                return a
            a[a < 0] += b
            a[a >= b] -= b

    def ramp_(t: np.ndarray) -> np.ndarray:
        return remdiv(t, T)

    return ramp_


def signal(t: np.ndarray) -> np.ndarray:
    return sine(9)(t) + 2 * sine(3)(t + 2) - 3 * sine(1)(t - 0.5)


# signal end

def sensing(p: float, N: int) -> np.ndarray:
    D = int(p * N)
    M = np.zeros(shape=(D, N), dtype=np.int64)
    indices = np.random.choice(np.arange(N), size=D, replace=False)
    for d in range(D):
        M[d, indices[d]] = 1
    return M


def reconstruct(measure: np.ndarray, sensing: np.ndarray, inverse_fourier: np.ndarray) -> np.ndarray:
    A = sensing @ inverse_fourier
    n, m = A.shape
    F = np.empty(shape=(2 * n, 2 * m), dtype=np.float64)
    F[0:n, 0:m] = np.real(A)
    F[0:n, m:2 * m] = -np.imag(A)
    F[n:2 * n, 0:m] = np.imag(A)
    F[n:2 * n, m:2 * m] = np.real(A)
    x = np.empty(shape=(2 * n,), dtype=np.float64)
    x[0:n] = measure
    x[n:2 * n] = np.zeros(shape=(n,), dtype=np.float64)

    f = suls.solve_l1(F, x)
    f_ = np.empty(shape=(m,), dtype=np.complex128)

    for i in range(m):
        f_[i] = complex(f[i], f[i + m])
    y = inverse_fourier @ f_
    return y


sampling_rate = 0.05
t = np.arange(-2, +2, sampling_rate)
f = signal(t)

plt.ylim([-10, +10])
plt.plot(t, f)
plt.show()

N = len(t)
K = 2 * N

fourier = forward_transform(N, K)
inverse_fourier = backward_transform(K, N)

p = 0.3
sensing_mat = sensing(p, N)
measure = sensing_mat @ f

plt.ylim([-10, +10])
plt.scatter(sensing_mat @ t, sensing_mat @ f)
plt.show()

reconstructed_signal = reconstruct(measure=measure, sensing=sensing_mat, inverse_fourier=inverse_fourier)
print(np.max(np.abs(reconstructed_signal)))
plt.ylim([-10, +10])
plt.plot(t, f, t, reconstructed_signal)
plt.show()
