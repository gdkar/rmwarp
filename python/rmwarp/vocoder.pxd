
import framer
from . cimport respectrum
#from rmspectrum import RMSpectrum
from .respectrum cimport ReSpectrum
#from rmfft import RMFFT
from .refft cimport ReFFT
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss
cdef cubic_hermite(p0,m0,p1,m1,float x, float x_lo, float x_hi)
cdef linear_interp(p0, p1, float x, float x_lo, float x_hi)

cdef class Vocoder:
    cdef readonly list      spec
    cdef readonly ReFFT     fft
    cdef readonly ReSpectrum __spec_acc
    cdef readonly object    framer
    cdef readonly int       __frame_size
    cdef readonly int       __hop_size
    cdef readonly int       __hop_size_out
    cdef readonly int       __max_frames
    cdef readonly double    __time_ratio
    cdef readonly double    __time_in
    cdef readonly double    __time_out
    cdef readonly double    __time_origin
    cdef readonly np.ndarray __unit
    cdef readonly np.ndarray __accumulator
    cdef readonly np.ndarray __windowAccumulator
    cpdef analyze_frame(self);
    cpdef interpolate_spec(self, when)
    cpdef advance_spec(self, ReSpectrum spec)
    cpdef synthesize_frame(self,ReSpectrum spec)
