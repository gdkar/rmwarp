import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss
from .rmspectrum import RMSpectrum
from . import basic
class RMFFT:
    def __init__(self, size = 0, win = None):
        if win is not None:
            if not size: size = len(win)
        self.size = size
        if win is not None:
            self.setWindow(win)

    def __len__(self):
        return self.size
    @property
    def size(self): return int(getattr(self,'__n',-1))

    @property
    def n(self):    return int(getattr(self,'__n',-1))

    @size.setter
    def size(self, n):
        n = max(int(n),0)
        if self.size == n:
            return
        setattr(self,'__n',n)
        if self.size > 0:
            self.__h          = np.ndarray((self.size,),dtype=np.float32)
            self.__Dh         = np.ndarray((self.size,),dtype=np.float32)
            self.__Th         = np.ndarray((self.size,),dtype=np.float32)
            self.__TDh        = np.ndarray((self.size,),dtype=np.float32)
            self.__fft_real   = fftw.empty_aligned((self.size,),dtype=np.float32)
            self.__fft_complex= None
            self.__fft_rereal = None

            self.__X          = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__X_Dh       = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__X_Th       = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__X_TDh      = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__X_         = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dPhi_dt    = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dM_dw      = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dPhi_dw    = np.ndarray((self.spacing,),dtype=np.float32)
            self.__d2Phi_dtdw = np.ndarray((self.spacing,),dtype=np.float32)
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
            self.__X_         = None
            self.__dPhi_dt    = None
            self.__dM_dw      = None
            self.__dPhi_dw    = None
            self.__d2Phi_dtdw = None

    @property
    def X_real(self):  return self.__X[:self.coef]
    @property
    def X_imag(self):  return self.__X[self.spacing:self.spacing+self.coef]
    @property
    def X_Dh_real(self):  return self.__X_Dh[:self.coef]
    @property
    def X_Dh_imag(self):  return self.__X_Dh[self.spacing:self.spacing+self.coef]
    @property
    def X_Th_real(self):  return self.__X_Th[:self.coef]
    @property
    def X_Th_imag(self):  return self.__X_Th[self.spacing:self.spacing+self.coef]
    @property
    def X_TDh_real(self):  return self.__X_TDh[:self.coef]
    @property
    def X_TDh_imag(self):  return self.__X_TDh[self.spacing:self.spacing+self.coef]

    def setWindow(self, win):
        n = min(len(win),len(self))
        self.h[:n] = win[:n]
        if n < len(self):
            self.h[n:] = 0
        self.Dh[:] = basic.time_derivative_window(self.h)
        self.Th[:] = basic.time_weighted_window(self.h)
        self.TDh[:]= basic.time_weighted_window(self.Dh)

    @property
    def h(self): return self.__h

    @property
    def Dh(self): return self.__Dh

    @property
    def Th(self): return self.__Th

    @property
    def TDh(self): return self.__TDh

    @property
    def coef(self): return self.size//2 + 1

    @property
    def spacing(self): return (self.coef + 15) & ~15

    def process(self, src, dst = None, when = 0):
        self.__fft_real[::] = src * self.h
        self.__plan_r2c.execute()
        self.X_real[::] = self.__fft_complex[::].real
        self.X_imag[::] = self.__fft_complex[::].imag

        self.__fft_real[::] = src * self.Dh
        self.__plan_r2c.execute()
        self.X_Dh_real[::] = self.__fft_complex[::].real
        self.X_Dh_imag[::] = self.__fft_complex[::].imag

        self.__fft_real[::] = src * self.Th
        self.__plan_r2c.execute()
        self.X_Th_real[::] = self.__fft_complex[::].real
        self.X_Th_imag[::] = self.__fft_complex[::].imag

        self.__fft_real[::] = src * self.TDh
        self.__plan_r2c.execute()
        self.X_TDh_real[::] = self.__fft_complex[::].real
        self.X_TDh_imag[::] = self.__fft_complex[::].imag

        if not isinstance(dst,RMSpectrum):
            dst = RMSpectrum(self.size,when)
        else:
            dst.size = self.size
            dst.when = when

        dst.size = self.size
        dst.when = when

        dst.X_real[::] = self.X_real
        dst.X_imag[::] = self.X_imag

        _X = self.X_real + self.X_imag * 1j

        dst.M[::] = np.log(np.abs(_X))
        dst.Phi[::] = np.angle(_X)

        _X_inv = _X.conj() / (_X * _X.conj() + 1e-10)

        _Dh_over_X = (self.X_Dh_real + self.X_Dh_imag * 1j) * _X_inv
        dst.dM_dt[::]   = _Dh_over_X.real
        dst.dPhi_dt[::] = _Dh_over_X.imag

        _Th_over_X = (self.X_Th_real + self.X_Th_imag * 1j) * _X_inv
        dst.dM_dw[::]   =-_Th_over_X.imag
        dst.dPhi_dw[::] = _Th_over_X.real

        _TDh_over_X = ((self.X_TDh_real + self.X_TDh_imag * 1j) * _X_inv).real
        _Th_Dh_over_X2 = (_Dh_over_X * _Th_over_X).real
        dst.d2Phi_dtdw[::] = _TDh_over_X - _Th_Dh_over_X2

        return dst
