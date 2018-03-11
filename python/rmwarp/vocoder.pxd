from libc.stdint cimport int64_t, int32_t, int16_t
from libc.stdint cimport *
from libc.stddef cimport *
from libcpp.vector cimport vector
from libcpp.deque  cimport deque
cimport libcpp.limits
cimport libcpp.memory
cimport libcpp.utility

cimport cython.operator

from rmwarp.framer cimport NpFramer, Framer, NpImageFramer, NpImageSpectrogram, Spectrogram



from .respectrum cimport ReSpec
cimport rmwarp.interp

from .refft cimport ReFFT
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss

ctypedef np.float32_t float_type

cdef class ResetSegment:
    cdef int         lo
    cdef int         hi
    cdef int         tag_min
    cdef int         tag_max
    cdef np.ndarray  tag_data

cdef class Vocoder:
    cdef object __weakref__
    cdef readonly list      spec
    cdef readonly ReFFT     fft
    cdef readonly object    framer
    cdef readonly int           __frame_size
    cdef readonly int           __resets
    cdef readonly int           __hop_size
    cdef readonly int           __hop_size_out
    cdef readonly float_type    __shaping
    cdef readonly int           __reset_granularity
    cdef readonly int           __reset_width
    cdef readonly int           __weight_width
    cdef readonly int           __max_frames
    cdef readonly double        __time_ratio
    cdef readonly double        __time_out
    cdef readonly double        __time_origin
    cdef readonly int           __frame_index
    cdef readonly float_type    __lgd_threshold
    cdef readonly float_type    __lgd_threshold_accute
    cdef readonly float_type    __d2Phi_threshold
    cdef readonly float_type    __d2Phi_threshold_accute

    cdef readonly np.ndarray    __unit
    cdef readonly np.ndarray    __Phi_table
    cdef readonly np.ndarray    __M_table
    cdef readonly np.ndarray    __lgd_table

    cdef readonly np.ndarray    __d2Phi_dtdw_table
    cdef readonly list          __reset_list
    cdef readonly np.ndarray    __reset_curr

    cdef readonly list          __reset_segs
    cdef readonly list          __reset_ring

    cdef readonly np.ndarray    __onset_table
    cdef readonly np.ndarray    __window
    cdef readonly np.ndarray    __accumulator
    cdef readonly np.ndarray    __windowAccumulator

    cpdef analyze_frame(self);
