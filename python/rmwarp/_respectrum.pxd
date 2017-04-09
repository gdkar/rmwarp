from libc.stdint cimport int64_t, int32_t, int16_t
from libc.stdint cimport *
from libc.stddef cimport *
from libcpp.vector cimport vector
cdef extern from "rmwarp/Simd.hpp" namespace "RMWarp::bs" nogil:
    cdef cppclass allocator[T]:
        pass
cdef extern from "rmwarp/ReSpectrum.hpp" namespace "RMWarp" nogil:
    cdef cppclass ReSpectrum:
        ctypedef float value_type
        ctypedef size_t size_type
        ctypedef float* pointer
        ctypedef const float* const_pointer
        ctypedef allocator[value_type] allocator_type
        ctypedef vector[value_type,allocator[value_type]] vector_type
        ReSpectrum()
        ReSpectrum(int)
        ReSpectrum(const ReSpectrum&)
        pointer X_real();

        pointer X_imag();

        pointer mag_data();

        pointer M_data();

        pointer Phi_data();

        pointer dM_dt_data();

        pointer dM_dw_data();

        pointer dPhi_dt_data();

        pointer dPhi_dw_data();

        pointer d2Phi_dtdw_data();

        pointer local_group_delay();
        pointer local_group_delay_weight();
        pointer local_group_delay_acc();
        pointer local_time();

        value_type epsilon() const;
        int size() const;
        int coefficients() const;
        int spacing() const;
        int64_t when() const;
        void set_when(int64_t _when);
        void resize(int _size);
        void reset(int _size, int64_t _when);