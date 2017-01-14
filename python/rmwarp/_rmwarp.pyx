from libc.stdint      cimport *
from libcpp.algorithm cimport *
from libcpp.utility   cimport *
from libcpp.numeric   cimport *
from libcpp.vector    cimport *

import numpy as np
cimport numpy as cnp
cdef extern from "ReassignedSpectrum.hpp" namespace "RMWarp" nogil:
    cdef cppclass RMSpectrum:
        RMSpectrum() except +
        RMSpectrum(int _size) except +
        int size()
        int spacing()
        int coefficients()
        int64_t when()
        void set_when(int64_t _when)
        void resize(int)
        float *X_real()
        float *X_imag()
        float *M_data()
        float *Phi_data()
        float *dM_dt_data()
        float *dPhi_dt_data()
        float *dM_dw_data()
        float *dPhi_dw_data()
        float *d2Phi_dtdw_data()

cdef extern from "ReassignedFFT.hpp" namespace "RMWarp" nogil:
    cdef cppclass RMFFT:
        RMFFT() except +
        RMFFT(int _size) except +
        RMFFT[It](int _size, It _W)except +
        RMFFT[It](It wbegin, It wend) except +
        It setWindow[It](int _size, It _W)
        int size()
        int spacing()
        int coefficients()
        void process(const float *const src, RMSpectrum & dst, int64_t when )


cdef class PyRMSpectrum(object):
    cdef readonly np.ndarray X
    cdef readonly np.ndarray X_log
    cdef readonly np.ndarray dM_dt
    cdef readonly np.ndarray dPhi_dt
    cdef readonly np.ndarray dM_dw
    cdef readonly np.ndarray dPhi_dw
    cdef readonly np.ndarray d2Phi_dtdw
    def __cinit__(self):
        self._d = RMSpectrum(_size)
    def __len__(self):
        return self._d.coefficients()
    property coefficients:
        def __get__(self):
            return self._d.coefficients()
    property X;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.X_real()
            cdef cnp.float[:_c] i = <cnp.float[:_c]>self._d.X_imag()
            cdef cnp.complex64_t[:_c] a = r + np.complex(0,1) * i
            return a
    property M;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.M_data()
            return np.asarray(r)

    property Phi;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.M_data()
            return np.asarray(r)
    property dM_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dM_dt_data()
            return np.asarray(r)
    property dPhi_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dPhi_dt_data()
            return np.asarray(r)
    property dM_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dM_dt_data()
            return np.asarray(r)
    property dM_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dM_dt_data()
            return np.asarray(r)
    property dPhi_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dPhi_dt_data()
            return np.asarray(r)
    property dM_dt;
        def __get__(self):
            cdef int _c = self.coefficients
            cdef cnp.float[:_c] r = <cnp.float[:_c]>self._d.dM_dt_data()
            return np.asarray(r)

