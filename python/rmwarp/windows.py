import numpy as np, scipy as sp


def window_time_weighted(win):
    count = len(win)
    return win * (np.arange(count,dtype=np.float32) - np.float32(count-1)/2)


def window_time_derivative(win):
    n = len(win)
    freq = sp.fft(win)
    base_idx = n//2 + 1
    base_mul = - ( n - 1) * 0.5
    norm_mul = 2 * np.pi / n
    for i in range(n):
        idx = ( i + base_idx) % n
        mul = ( i + base_mul ) * norm_mul
        freq[idx] *= -mul
    return sp.ifft(freq).imag
