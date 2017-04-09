from libc.stdint cimport *
from libcpp.vector cimport vector
#from _respectrum cimport ReSpectrum
cdef extern from "Simd.hpp" namespace "RMWarp::bs" nogil:
    cdef cppclass allocator[T]:
        pass
cdef extern from "ReSpectrum.hpp" namespace "RMWarp" nogil:
    cdef cppclass ReSpectrum

cdef extern from "ReFFT.hpp" namespace "RMWarp" nogil:
    cdef cppclass ReFFT:
        ctypedef float value_type
        ctypedef size_t size_type
        ctypedef float* pointer
        ctypedef const float* const_pointer
        ctypedef allocator[value_type] allocator_type
        ctypedef vector[value_type,allocator[value_type]] vector_type
        ReFFT()
        ReFFT(int _size)
        It setWindow[It](It wbegin, It wend)
        @staticmethod
        ReFFT Kaiser(int _size, float alpha)
        void update_group_delay(ReSpectrum &spec) const
        void process[It](It src, It send, ReSpectrum &dst, int64_t when)
        void process[It](It src, ReSpectrum &dst, int64_t when )
        void inverse[It,iIt](It dst, iIt _M, iIt _Phi)
        void inverseCepstral[I,O](O dst, I src)
        void set_epsilon(value_type )
        value_type epsilon() const
        int spacing() const
        int size() const
        int coefficients() const
        float lo_mul()
        float hi_mul()
        int bin_minimum()
        void set_lo_mul(float)
        void set_hi_mul(float)
        void set_bin_minimum(int)

