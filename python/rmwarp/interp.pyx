# cython: np_pythran=False

import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss
cimport numpy as np
from .refft import ReFFT
from .respectrum cimport ReSpectrum
from . import basic

cimport numpy as np
cimport cython

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, weak_ptr, unique_ptr, allocator
from libcpp.cast cimport *

cpdef cubic_hermite( p0, m0, p1, m1, cython.floating x,cython.floating x_lo = 0,cython.floating x_hi = 1):
    cdef cython.floating x_dist = (x_hi-x_lo)**-1
    cdef cython.floating t      = (x-x_lo) * x_dist
    cdef cython.floating t2 = t**2
    cdef cython.floating t3 = t * t2
    cdef cython.floating h00 = 2*t3-3*t2+1
    cdef cython.floating h10 = (t3-2*t2+t)*x_dist
    cdef cython.floating h01 = -2*t3 + 3*t2
    cdef cython.floating h11 = (t3-t2)*x_dist
    return h00*p0 + h10 * m0 + h01 * p1 + h11 * m1

cpdef linear_interp(
    p0, p1
  , cython.floating x
  , cython.floating x_lo = 0
  , cython.floating x_hi = 1):
    cdef cython.floating x_dist = (x_hi-x_lo)**-1
    cdef cython.floating t = (x-x_lo)*x_dist
    return p0 * (1-t) + p1*t

cpdef np.ndarray windowed_diff(np.ndarray a, int w):
    cdef int l = a.shape[0]
    cdef np.ndarray res = np.zeros_like(a)
    if w >= l:
        res[::] = a[-1]
        return res
    if l > 2 * w:
        res[:w]   = a[w:2*w];
        res[w:-w] = a[2*w:] - a[:-2*w]
        res[-w:]  = a[-1]   - a[-w*2:-w]
    else:
        res[:-w] = a[w:]
        res[-w:] = a[-1] - a[:w]
    return res

cpdef np.ndarray find_runs(np.ndarray a, int base = 0):
    shape = [a.shape[_] for _ in range(a.ndim)]
    shape[0] += 2
    cdef np.ndarray asbool = np.zeros(shape=tuple(shape),dtype=np.bool8)
    asbool[1:-1] = a!=0
    return ((asbool[1:] ^ asbool[:-1]).nonzero()[0]).reshape((-1,2)) + base

cpdef np.ndarray find_tagged_runs(np.ndarray a, int tag, int base = 0):
    shape = [a.shape[_] for _ in range(a.ndim)]
    shape[0] += 2
    cdef np.ndarray asbool = np.zeros(shape=tuple(shape),dtype=np.bool8)
    asbool[1:-1] = a!=0
    cdef np.ndarray runs = ((asbool[1:] ^ asbool[:-1]).nonzero()[0]).reshape((-1,2)) + base
    return np.hstack([runs,np.ones_like(runs[::,None,0]) * tag])
cpdef np.ndarray binary_dilation(np.ndarray a, int w):
    cdef np.ndarray res = a != 0
    for x in range(1, w):
        res[1:]  += res[:-1]
        res[:-1] += res[1:]
    return res

cpdef np.ndarray binary_erosion(np.ndarray a, int w):
    cdef np.ndarray res = a != 0
    for x in range(1,w):
        res[1:]  *= res[:-1]
        res[:-1] *= res[1:]
    return res

cpdef np.ndarray binary_closing(np.ndarray a, int w):
    return binary_erosion(binary_dilation(a,w),w)

cpdef np.ndarray binary_opening(np.ndarray a, int w):
    return binary_dilation(binary_erosion(a,w),w)

cpdef np.ndarray binary_smooth(np.ndarray a, int w):
    return ~binary_closing(~binary_closing(a,w),w)

