from libcpp.vector cimport vector
from libcpp.deque  cimport deque
cimport libcpp.memory
cimport libcpp.utility
import numpy as np
cimport numpy


cdef extern from *:
    ctypedef float * float_ptr "float*"

def cxx_kaiser_window( size, alpha=12.):
    cdef float[::1] buf = np.zeros((size,),dtype=np.float32)
    cdef float_ptr ptr = &buf[0]
    cdef float_ptr pend= ptr + size
    make_kaiser_window[float_ptr](ptr,pend,alpha)
    return np.asarray(buf,dtype=np.float32)
