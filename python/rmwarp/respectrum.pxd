from libcpp.vector cimport vector
from ._respectrum cimport ReSpectrum

cdef class ReSpec:
    cdef ReSpectrum m_d
