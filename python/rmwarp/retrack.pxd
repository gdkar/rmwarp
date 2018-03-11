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
#cimport respectrum

#from rmspectrum import RMSpectrum
from .respectrum cimport ReSpec
cimport rmwarp.interp
#from rmfft import RMFFT
from .refft cimport ReFFT
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss

ctypedef np.float32_t float_type

cdef class ReOdf:
    cdef object             __weakref__
    cdef readonly double    __rate
    cdef readonly int       __frame_size
    cdef readonly int       __hop_size
    cdef readonly double    __shaping
    cdef readonly double    __d2Phi_threshold
    cdef readonly double    __lgd_threshold
    cdef readonly int       __opening
    cdef readonly int       __closing
    cdef readonly int64_t   __when
    cdef readonly ReFFT     fft
    cdef readonly list      spec
    cdef readonly np.ndarray neigh

cdef class ReTrack:
    cdef object         __weakref__
    cdef readonly object            framer
    cdef readonly ReOdf             odf

    cdef readonly double            acf_duration
    cdef readonly int               acf_size

    cdef readonly np.ndarray        odf_buf
    cdef readonly np.ndarray        cum_buf

    cdef readonly np.ndarray        acf_buf
    cdef readonly np.ndarray        comb_out
    cdef readonly np.ndarray        tempo_obs
    cdef readonly np.ndarray        delta_buf
    cdef readonly np.ndarray        tempo_table
