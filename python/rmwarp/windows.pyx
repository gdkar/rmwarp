import numpy as np, scipy as sp, scipy.signal as ss
cimport numpy as np


def kbd_window( N, alpha=12.):
    N = int(N)
    assert (N%2)==0
    M = N//2
    _w = ss.kaiser(M + 1, np.float64(alpha*np.pi),sym=False)
    _n = _w.sum()
    _half = np.sqrt(_w.cumsum() / _n)[:M]
    return np.concatenate((_half,_half[::-1]))

def xiph_vorbis_window( N ):
    return np.sin(
        np.pi/2 * np.sin(
            np.pi/N * (np.arange(N) + 0.5))**2
        ).astype(np.float32)
