import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss
from .rmspectrum import RMSpectrum
from .rmspectrum cimport RMSpectrum
from .basic cimport time_derivative_window,time_weighted_window

cimport numpy as np


cdef class RMFFT:
    cdef np.ndarray __h
    cdef np.ndarray __Dh
    cdef np.ndarray __Th
    cdef np.ndarray __TDh
    cdef object __fft_real
    cdef object __fft_complex
    cdef object __fft_rereal

    cdef np.ndarray __X
    cdef np.ndarray __X_Dh
    cdef np.ndarray __X_Th
    cdef np.ndarray __X_TDh
    cdef np.ndarray __dPhi_dt
    cdef np.ndarray __dM_dw
    cdef np.ndarray __dPhi_dw
    cdef np.ndarray __d2Phi_dtdw
    cdef readonly object __plan_r2c
    cdef readonly object __plan_c2r
    cdef readonly int    __n
    cdef public float    __epsilon
    cdef _resize(self, int n)
    cpdef setWindow(self, win)
    cdef RMSpectrum _process(self, src, RMSpectrum dst, when = *)
    cdef _synthesize(self, RMSpectrum src)
