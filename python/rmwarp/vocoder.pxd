from libc.stdint cimport int64_t, int32_t, int16_t
from libc.stdint cimport *
from libc.stddef cimport *
from libcpp.vector cimport vector
from libcpp.deque  cimport deque
cimport libcpp.limits
cimport libcpp.memory
cimport libcpp.utility

cimport cython.operator

import framer
cimport respectrum

#from rmspectrum import RMSpectrum
from respectrum cimport ReSpec
#from rmfft import RMFFT
from refft cimport ReFFT
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss
cpdef cubic_hermite(p0,m0,p1,m1,float x, float x_lo, float x_hi)
cpdef linear_interp(p0, p1, float x, float x_lo, float x_hi)
cpdef np.ndarray windowed_diff(np.ndarray a, int w)
cpdef np.ndarray binary_dilation(np.ndarray a, int w)
cpdef np.ndarray binary_erosion(np.ndarray a, int w)
cpdef np.ndarray binary_closing(np.ndarray a, int w)
cpdef np.ndarray find_runs(np.ndarray a)
cdef class Vocoder:
    cdef object __weakref__
    cdef readonly list      spec
    cdef readonly ReFFT     fft
    cdef readonly object    framer
    cdef readonly int       __frame_size
    cdef readonly int       __resets
    cdef readonly int       __hop_size
    cdef readonly int       __hop_size_out
    cdef readonly double    __shaping
    cdef readonly int       __reset_granularity
    cdef readonly int       __reset_width
    cdef readonly int       __max_frames
    cdef readonly double    __time_ratio
    cdef readonly double    __time_out
    cdef readonly double    __time_origin
    cdef readonly int       __frame_index
    cdef readonly np.ndarray  __unit
    cdef readonly np.ndarray __Phi_table
    cdef readonly np.ndarray __M_table
    cdef readonly np.ndarray __lgd_table
    cdef readonly np.ndarray __d2Phi_dtdw_table
    cdef readonly list       __reset_list
    cdef readonly list       __reset_segs

    cdef readonly np.ndarray __onset_table
    cdef readonly np.ndarray __window
    cdef readonly np.ndarray __accumulator
    cdef readonly np.ndarray __windowAccumulator

    cdef readonly np.ndarray __reset_curr
    cdef readonly np.ndarray __reset_view
    cdef readonly np.ndarray __reset_last_idx
    cdef readonly np.ndarray __reset_last_ref
    cdef readonly int        __oldest_active

    cpdef analyze_frame(self);
#    cpdef synthesize_frame(self)
#    cpdef np.ndarray make_frame(self, double pts)
