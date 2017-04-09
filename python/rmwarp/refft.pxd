from ._refft cimport ReFFT as _ReFFT
from ._respectrum cimport ReSpectrum
from libcpp.vector cimport vector

cdef class ReFFT:
    cdef _ReFFT m_d
