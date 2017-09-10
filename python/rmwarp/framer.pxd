import av, numpy as np, scipy as sp, scipy.signal as ss, scipy.fftpack as sf
import scipy.fftpack as fp
import itertools as it
cimport numpy as np

cdef class Framer:
    cdef readonly object f
    cdef readonly object r
    cdef readonly object c
    cdef readonly object d
    cdef readonly object s
    cdef readonly int frame_size
    cdef readonly int hop_size
    cpdef read(self, frame_size = *)
    cpdef seek(self, pts)

cdef class NpFramer(Framer):
    cdef public bint transpose
    cdef object dtype
    cpdef read(self, frame_size = *)

cdef class NpImageFramer(Framer):
    cdef readonly np.ndarray image
    cpdef read(self, frame_size = *)

cdef class Spectrogram(NpFramer):
    cdef readonly np.ndarray h
    cpdef read(self, frame_size = *)

cdef class NpImageSpectrogram(Framer):
    cdef readonly int real_size
    cdef readonly np.ndarray h
    cdef readonly np.ndarray image
    cpdef read(self, frame_size = *)

cdef class RMSpectrogram(NpFramer):
    cdef readonly np.ndarray h
    cdef readonly np.ndarray Th
    cdef readonly np.ndarray Dh
    cdef readonly np.ndarray unit
    cdef readonly np.ndarray complex_unit
    cpdef read(self, frame_size = *)


cdef class DFCollection:
    cdef readonly object funcs
    cdef readonly object curves
    cdef readonly Spectrogram spectrogram
