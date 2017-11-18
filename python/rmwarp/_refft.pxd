from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp cimport bool
from cpython cimport bool as pybool
#from _respectrum cimport ReSpectrum
cdef extern from "Simd.hpp" namespace "RMWarp::bs" nogil:
    cdef cppclass allocator[T]:
        pass
cdef extern from "ReSpectrum.hpp" namespace "RMWarp" nogil:
    cdef cppclass ReSpectrum

ctypedef float * floatp
ctypedef const float * cfloatp
cdef extern from "ReFFT.hpp" namespace "RMWarp" nogil:
    cdef cppclass ReFFT:
        ctypedef float value_type
        ctypedef size_t size_type
        ctypedef float* pointer
        ctypedef const float* const_pointer
        ctypedef allocator[value_type] allocator_type
        ctypedef vector[value_type,allocator[value_type]] vector_type
        float m_epsilon
        allocator_type m_alloc
        vector_type m_h
        vector_type m_Dh
        vector_type m_Th
        vector_type m_TDh

        vector_type m_flat
        vector_type m_split

        vector_type m_X
        vector_type m_X_Dh
        vector_type m_X_Th
        vector_type m_X_TDh

        ReFFT()
        ReFFT(int _size)
        ReFFT(const ReFFT& )
        ReFFT(floatp wbegin, floatp wend)
        ReFFT(floatp wbegin, floatp wend, floatp dt_begin, floatp dt_end)
#        ReFFT[It,A](It wbegin, It wend, const A&)
#        ReFFT[It,A](It wbegin, It wend,It dt_begin, It dt_end, const A&)

        It setWindow[It](It wbegin, It wend)

        @staticmethod
        ReFFT Kaiser(int _size, float alpha)

        void _finish_process(ReSpectrum & dst, int64_t _when);

        void swap(ReFFT & o);
        void initPlans();
        void setWindow[It](It wbegin, It wend, It dt_begin, It dt_end)

        void process[It]( It src, It send, ReSpectrum & dst, int64_t when )
        void process[It]( It src, ReSpectrum & dst, int64_t when )
        void inverse[It,iIt]( It dst, iIt _M, iIt _Phi)
        void inverseCepstral[I,O](O dst, I src)
        int spacing()
        int size()
        int coefficients()
        const float* h_data()
        const float* Th_data()
        const float* Dh_data()
        const float* TDh_data()
        value_type time_width()
        value_type freq_width()
