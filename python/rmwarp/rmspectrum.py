import pyfftw as fftw, numpy as np, scipy as sp

class RMSpectrum:
    def __init__(self, size = 0, when = 0):
        self.resize(size)
        self.when = when

    @property
    def when(self): return int(self.__when)

    @when.setter
    def when(self,_when): self.__when = int(_when)

    @property
    def size(self): return int(self.__size)

    @size.setter
    def size(self, n):
        self.resize(n)

    def resize(self, n):
        self.__size = max(int(n),0)
        if self.__size:
            self.__X          = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__X_log      = np.ndarray((self.spacing * 2,),dtype=np.float32)
            self.__dM_dt      = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dPhi_dt    = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dM_dw      = np.ndarray((self.spacing,),dtype=np.float32)
            self.__dPhi_dw    = np.ndarray((self.spacing,),dtype=np.float32)
            self.__d2Phi_dtdw = np.ndarray((self.spacing,),dtype=np.float32)
        else:
            self.__X          = None
            self.__X_log      = None
            self.__dM_dt      = None
            self.__dPhi_dt    = None
            self.__dM_dw      = None
            self.__dPhi_dw    = None
            self.__d2Phi_dtdw = None

    @property
    def coef(self): return self.size//2 + 1

    @property
    def spacing(self): return (self.coef + 15) & ~15

    @property
    def X_real(self):  return self.__X[:self.coef]
    @property
    def X_imag(self):  return self.__X[self.spacing:self.spacing+self.coef]

    @property
    def M(self):       return self.__X_log[:self.coef]
    @property
    def Phi(self):     return self.__X_log[self.spacing:self.spacing+self.coef]

    @property
    def dM_dt(self):   return self.__dM_dt[:self.coef]
    @property
    def dPhi_dt(self): return self.__dPhi_dt[:self.coef]

    @property
    def dM_dw(self):   return self.__dM_dw[:self.coef]
    @property
    def dPhi_dw(self): return self.__dPhi_dw[:self.coef]

    @property
    def d2Phi_dtdw(self):return self.__d2Phi_dtdw[:self.coef]
