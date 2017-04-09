# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = rmfft.cpp


import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, weak_ptr, unique_ptr, allocator
from libcpp.cast cimport *

cimport libcpp.memory
cimport libcpp.utility
cimport libcpp.iterator
cimport cython
import  cython
from libcpp.cast cimport *
from .rmspectrum import RMSpectrum
from .rmspectrum cimport RMSpectrum


cimport numpy as np
cimport cython
cdef inline _align_to(val,quantum):
    return (val + quantum - 1) & ~(quantum - 1)



cdef class RMFFT:
    def __cinit__(self, size = 0, win = None):
        self.__epsilon = 1e-6
        if win is not None:
            if not size:
                size = len(win)
        self.size = size
        if win is not None:
            self.setWindow(win)

    def __len__(self):
        return self.size

    @property
    def n(self):
        return self.size

    @property
    def size(self):
        return self.__n

    @size.setter
    def size(self,n):
        self._resize(n)

    def resize(self,n):
        self._resize(n)

    cdef _resize(self, int n):
        n = max(n,0)
        if self.size == n:
            return
        self.__n = n
        cdef int spacing = self.spacing
        cdef int coef    = self.coef
        if self.size > 0:
            self.__h          = np.ndarray((n,),dtype=np.float32)
            self.__Dh         = np.ndarray((n,),dtype=np.float32)
            self.__Th         = np.ndarray((n,),dtype=np.float32)
            self.__TDh        = np.ndarray((n,),dtype=np.float32)
            self.__fft_real   = fftw.empty_aligned((n,),dtype=np.float32)
            self.__fft_complex= None
            self.__fft_rereal = None

            self.__X          = np.ndarray((coef,),dtype=np.complex64)
            self.__X_Dh       = np.ndarray((coef,),dtype=np.complex64)
            self.__X_Th       = np.ndarray((coef,),dtype=np.complex64)
            self.__X_TDh      = np.ndarray((coef,),dtype=np.complex64)
            self.__dPhi_dt    = np.ndarray((coef,),dtype=np.float32)
            self.__dM_dw      = np.ndarray((coef,),dtype=np.float32)
            self.__dPhi_dw    = np.ndarray((coef,),dtype=np.float32)
            self.__d2Phi_dtdw = np.ndarray((coef,),dtype=np.float32)
            self.__plan_r2c   = fftw.builders.rfft(
                self.__fft_real
            , overwrite_input = True
            , planner_effort = 'FFTW_ESTIMATE'
                )
            self.__fft_real    = self.__plan_r2c.input_array
            self.__fft_complex = self.__plan_r2c.output_array
            self.__plan_c2r   = fftw.builders.irfft(
                self.__fft_complex
            , overwrite_input = True
            , auto_align_input = False
            , planner_effort = 'FFTW_ESTIMATE'
                )
            self.__fft_rereal = self.__plan_c2r.output_array

        else:
            self.__plan_r2c = None
            self.__plan_c2r = None
            self.__h          = None
            self.__Dh         = None
            self.__Th         = None
            self.__TDh        = None
            self.__fft_real   = None
            self.__fft_complex= None

            self.__X          = None
            self.__X_Dh       = None
            self.__X_Th       = None
            self.__X_TDh      = None
            self.__dPhi_dt    = None
            self.__dM_dw      = None
            self.__dPhi_dw    = None
            self.__d2Phi_dtdw = None

    @property
    def X(self): return self.__X
    @property
    def X_Dh(self):return self.__X_Dh
    @property
    def X_Th(self):return self.__X_Th
    @property
    def X_TDh(self):return self.__X_TDh
    @property
    def X_real(self):return self.X.real

    @property
    def X_imag(self):return self.X.imag

    @property
    def X_Dh_real(self):return self.X_Dh.real
    @property
    def X_Dh_imag(self):return self.X_Dh.imag

    @property
    def X_Th_real(self):return self.X_Th.real
    @property
    def X_Th_imag(self):return self.X_Th.imag

    @property
    def X_TDh_real(self):return self.X_TDh.real
    @property
    def X_TDh_imag(self):return self.X_TDh.imag

    @cython.cdivision(True)
    cpdef setWindow(self, win):
        n = min(len(win),len(self))
        self.h[:n] = win[:n]
        if n < len(self):
            self.h[n:] = 0
        self.Dh[:] = time_derivative_window(self.h)
        self.Th[:] = time_weighted_window(self.h)
        self.TDh[:]= time_weighted_window(self.Dh)

    @property
    def h(self):return self.__h
    @property
    def Dh(self):return self.__Dh
    @property
    def Th(self):return self.__Th
    @property
    def TDh(self):return self.__TDh

    @property
    @cython.cdivision(True)
    def coef(self): return self.size//2 + 1

    @property
    def spacing(self): return _align_to(self.coef,16)

    def process(self,src,dst = None,when = 0):
        return self._process(src,dst,when)

    @cython.cdivision(True)
    cdef RMSpectrum _process(self, src, RMSpectrum dst, when = 0):
        cdef int coef = self.coef
        cdef int size = self.size
        fft_r = self.__fft_real
        fft_c = self.__fft_complex
        plan_r2c = self.__plan_r2c
        fft_r[::] = src[::] * self.h  ;plan_r2c.execute();self.X[:] = fft_c[:]
        fft_r[::] = src[::] * self.Dh ;plan_r2c.execute();self.X_Dh[:]  = fft_c[:]
        fft_r[::] = src[::] * self.Th ;plan_r2c.execute();self.X_Th[:]  = fft_c[:]
        fft_r[::] = src[::] * self.TDh;plan_r2c.execute();self.X_TDh[:] = fft_c[:]

        if dst is None:
            dst = RMSpectrum(self.size,when)
        else:
            dst.size = self.size
            dst.when = when

        dst.X[::] = self.X[::]

        _X = self.X[::]

        xabs = np.abs(_X)

        xnorm =  (_X * _X.conj()).real
        dst.mag[:] = np.sqrt(xnorm)
        dst.M[:]   = np.log (xnorm)
        dst.Phi[:] = np.angle(_X)

        _X_inv         = _X.conj() / ( xnorm + self.__epsilon)

        _Dh_over_X     = self.X_Dh * _X_inv
        dst.dM_dt[:]   = _Dh_over_X.real
        dst.dPhi_dt[:] = _Dh_over_X.imag

        _Th_over_X     = self.X_Th * _X_inv
        dst.dM_dw[:]   =-_Th_over_X.imag
        dst.dPhi_dw[:] = _Th_over_X.real

        _TDh_over_X = (self.X_TDh * _X_inv).real
        _Th_Dh_over_X2 = (_Dh_over_X * _Th_over_X).real

        dst.d2Phi_dtdw[:coef] = _TDh_over_X - _Th_Dh_over_X2

        dst._update_group_delay(self.__epsilon)
        return dst

    @cython.cdivision(True)
    cdef _synthesize(self, RMSpectrum spec):
        cdef np.ndarray[np.npy_float32,ndim=1] X_mag = np.exp(spec.M * np.float32(0.5))
        cdef np.ndarray[np.npy_float32,ndim=1] X_phase = spec.Phi[::]
        self.__fft_complex[::] = np.exp(spec.M * np.float32(0.5) + spec.Phi * np.float32(1j))
#        .real = np.cos(X_phase) * X_mag
 #       self.__fft_complex[::].imag = np.sin(X_phase) * X_mag
        self.__plan_c2r.execute()
        return  self.__fft_rereal[::] / self.size
#* self.h / self.size

    def synthesize(self, RMSpectrum spec):
        return self._synthesize(spec)
