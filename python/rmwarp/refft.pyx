#from _refft cimport ReFFT
#from _respectrum cimport ReSpectrum
from .respectrum cimport ReSpec
from _refft cimport ReFFT as _ReFFT
from libcpp.vector cimport vector
from libc.stdint cimport int64_t,int32_t
cimport numpy as np
import  numpy as np

ctypedef float * floatp
cdef class ReFFT:
    def __cinit__(self,int size = 2048, *args, **kwargs):
        alpha = kwargs.pop('alpha',None)
        if alpha is not None:
            self.m_d = _ReFFT.Kaiser(size,alpha)
            return
        self.m_d = _ReFFT(size)
        win = kwargs.pop('win',None)

    def __len__(self):
        return self.m_d.size()

    def set_window(self, win):
#        cdef float[:] tmp = win[:]
        self.m_d = _ReFFT(<int32_t>(len(win)))
        cdef float[:] tmp = np.asarray(win,dtype=np.float32)
        self.m_d.setWindow(&tmp[0],&tmp[0] + len(tmp))

    @property
    def spacing(self):return self.m_d.spacing()

    @property
    def coefficients(self): return self.m_d.coefficients()

    @property
    def coef(self):return self.coefficients

    @property
    def size(self): return self.m_d.size();

    @property
    def epsilon(self): return self.m_d.m_epsilon

    @epsilon.setter
    def epsilon(self, float e):
        self.m_d.m_epsilon = e

    def process(self, data, ReSpec spec = None, int64_t when = 0):
        cdef float[:] tmp = data
        if spec is None:
            spec = ReSpec(self.size)
        self.m_d.process[floatp](&tmp[0],spec.m_d, when)
        self.update_group_delay(spec)
        return spec

    def inverse(self, _M, _Phi, float[:] dst = None):
        cdef vector[float] tmp_M = _M
        cdef vector[float] tmp_Phi= _Phi
        cdef vector[float] tmp_out
        tmp_out.resize(self.m_d.size())

        self.m_d.inverse[floatp,floatp](&tmp_out[0],&tmp_M[0],&tmp_Phi[0])
        return tmp_out
