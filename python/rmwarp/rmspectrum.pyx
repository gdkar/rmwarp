# -*- coding: utf-8 -*-

import pyfftw as fftw, numpy as np, scipy as sp
cimport numpy as np
from libc.stdint  cimport int64_t
cdef class RMSpectrum:
    def __init__(self, size = 0, when = 0):
        self._resize(size)
        self.when = when


    def reassigned_points(self, cond = None, _m =None):
        _t = self.dPhi_dw + self.when
        _w = -self.dPhi_dt + np.arange(self.coef) #* ( 2 * np.pi / self.size)
        if _m is None:
            _m = self.M
        elif isinstance(_m,str):
            _m = getattr(self,_m)
        if cond is not None:
            _where = np.where(cond(self.d2Phi_dtdw))
            _t = _t[_where]
            _w = _w[_where]
            _m = _m[_where]
        return _t,_w,_m

    @property
    def size(self): return int(self.__size)

    @size.setter
    def size(self, n):self._resize(n)

    def __len__(self):
        return self.n

    def resize(self,n): self._resize(n)

    cdef _resize(self, n):
        self.__size = max(int(n),0)
        coef = self.coef
        if self.__size:
            self.X          = np.ndarray((coef,),dtype=np.complex64)
            self.M          = np.ndarray((coef,),dtype=np.float32)
            self.Phi        = np.ndarray((coef,),dtype=np.float32)
            self.mag        = np.ndarray((coef,),dtype=np.float32)
            self.dM_dt      = np.ndarray((coef,),dtype=np.float32)
            self.dPhi_dt    = np.ndarray((coef,),dtype=np.float32)
            self.dM_dw      = np.ndarray((coef,),dtype=np.float32)
            self.dPhi_dw    = np.ndarray((coef,),dtype=np.float32)
            self.d2Phi_dtdw = np.ndarray((coef,),dtype=np.float32)
            self.lgda       = np.ndarray((coef,),dtype=np.float32)
            self.lgdw       = np.ndarray((coef,),dtype=np.float32)
            self.lgd        = np.ndarray((coef,),dtype=np.float32)
        else:
            self.X          = None
            self.M          = None
            self.Phi        = None
            self.mag        = None
            self.dM_dt      = None
            self.dPhi_dt    = None
            self.dM_dw      = None
            self.dPhi_dw    = None
            self.d2Phi_dtdw = None
            self.lgda       = None
            self.lgdw       = None
            self.lgd        = None

    cdef _update_group_delay(self, float __epsilon):
        fr = np.float32(0.95)
        ep = np.sqrt(__epsilon * fr/(1-fr))
        self.lgdw[ self.mag > ep] = 1
        self.lgdw[ self.mag <= ep] = 0
        self.lgda[:] = -self.lgdw[:] * self.dPhi_dw[:]
        self.lgdw = np.cumsum(self.lgdw[:])
        self.lgda = np.cumsum(self.lgda[:])
        hi_bound = lambda x: min(self.coef-1,max((x * 1200)//1024, x + 12))
        lo_bound = lambda x: max(0,min(x-12,(x * 860)//1024))
        bound = lambda x: (lo_bound(x),hi_bound(x))
        lgdad = lambda lo,hi: (self.lgda[hi] - self.lgda[lo])
        lgdwd = lambda lo,hi: ((self.lgdw[hi] - self.lgdw[lo]) + __epsilon)
        lgdd  = lambda lo,hi: lgdad(lo,hi)/lgdwd(lo,hi)
        val = lambda x: lgdd(*bound(x))
        self.lgd[:] = np.fromiter((val(x) for x in range(self.coef)),dtype=np.float32)
#        (self.lgda[hi_bound] - self.lgda[lo_bound]) / ( self.lgdw[hi_bound]-self.lgdw[lo_bound] + __epsilon)

    @property
    def coef(self): return self.size//2 + 1
    @property
    def spacing(self): return (self.coef + 15) & ~15

#    @property
#    def X(self): return self.__X

    @property
    def X_real(self):  return self.X.real
    @property
    def X_imag(self):  return self.X.imag
#    @property
#    def mag(self):  return self.__mag
#    @property
#    def M(self):  return self.__M
#    @property
#    def Phi(self):  return self.__Phi

#    @property
#    def dM_dt(self):  return self._dM_dt
#    @property
#    def dPhi_dt(self):  return self.__dPhi_dt
#    @property
#    def dM_dw(self):  return self.__dM_dw
#    @property
#    def dPhi_dw(self):  return self.__dPhi_dw
#    @property
#    def d2Phi_dtdw(self):  return self.__d2Phi_dtdw
    @property
    def local_group_delay(self):  return self.lgd
    @property
    def local_group_delay_weight(self):  return self.lgdw
    @property
    def local_group_delay_acc(self):  return self.lgda


