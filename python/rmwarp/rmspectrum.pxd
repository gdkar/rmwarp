import pyfftw as fftw, numpy as np, scipy as sp
cimport numpy as np
from libc.stdint  cimport int64_t

cdef class RMSpectrum:
    cdef readonly np.ndarray X
    cdef readonly np.ndarray mag
    cdef readonly np.ndarray M
    cdef readonly np.ndarray Phi
    cdef readonly np.ndarray dM_dt
    cdef readonly np.ndarray dPhi_dt
    cdef readonly np.ndarray dM_dw
    cdef readonly np.ndarray dPhi_dw
    cdef readonly np.ndarray d2Phi_dtdw
    cdef readonly np.ndarray lgda
    cdef readonly np.ndarray lgdw
    cdef readonly np.ndarray lgd
    cdef readonly int __size
    cdef public   int when
    cdef _resize(self, n)
    cdef _update_group_delay(self, float __epsilon)
