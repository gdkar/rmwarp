from ._refft cimport ReFFT
from ._respectrum cimport ReSpectrum
from libcpp.vector cimport vector

cdef class ReFft:
    cdef ReFFT m_d
