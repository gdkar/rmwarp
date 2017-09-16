from libcpp.vector cimport vector
from ._respectrum cimport ReSpectrum
from libc.stdint cimport int64_t
cimport numpy as np
import numpy as np, scipy as sp
cdef class ReSpec:
    def __cinit__(self, int size = 0):
        self.m_d = ReSpectrum(size)

    def __len__(self):
        return self.m_d.size()

    def resize(self, int size):
        self.m_d.resize(size)

    def unwrap_from(self, ReSpec other):
        if other is not None:
            self.m_d.unwrapFrom(other.m_d)

    @property
    def size(self):
        return self.m_d.size()

    @size.setter
    def size(self, int size):
        self.m_d.resize(size)

    @property
    def coefficients(self):
        return self.m_d.coefficients()

    @property
    def spacing(self):
        return self.m_d.spacing()

    @property
    def when(self):
        return self.m_d.when()

    @when.setter
    def when(self, int64_t when):
        self.m_d.set_when(when)

    @property
    def epsilon(self):
        return self.m_d.epsilon

    @epsilon.setter
    def epsilon(self,val):
        self.m_d.epsilon = val

    @property
    def X_imag(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.X_imag())
        return np.asarray(ref)
    @property
    def X_real(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.X_real())
        return np.asarray(ref)
    @property
    def mag(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.mag_data())
        return np.asarray(ref)
    @property
    def Phi(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.Phi_data())
        return np.asarray(ref)
    @property
    def M(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.M_data())
        return np.asarray(ref)
    @property
    def dPhi_dt(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.dPhi_dt_data())
        return np.asarray(ref)
    @property
    def dM_dt(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.dM_dt_data())
        return np.asarray(ref)
    @property
    def dPhi_dw(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.dPhi_dw_data())
        return np.asarray(ref)
    @property
    def dM_dw(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.dM_dw_data())
        return np.asarray(ref)
    @property
    def d2Phi_dtdw(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.d2Phi_dtdw_data())
        return np.asarray(ref)
    @property
    def d2M_dtdw(self):
        cdef float[:] ref = <float[:self.coefficients]>(self.m_d.d2M_dtdw_data())
        return np.asarray(ref)
