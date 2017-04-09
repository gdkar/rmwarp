from libcpp cimport vector, map as std_map, unordered_map, deque
from libcpp.string cimport string  as std_string

import framer
from . cimport respectrum
from .respectrum cimport ReSpectrum
from .refft cimport ReFFT
from .basic cimport time_derivative_window
from .basic cimport time_weighted_window
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss

cdef cubic_hermite( p0, m0, p1, m1, float x,float x_lo,float x_hi):
    cdef float x_dist = (x_hi-x_lo)**-1
    cdef float t = (x-x_lo)*x_dist
    cdef float h00 = 2*t**3-3*t**2+1
    cdef float h10 = (t**3-2*t**2+t)*x_dist
    cdef float h01 = -2*t**3 + 3*t**2
    cdef float h11 = (t**3-t**2)*x_dist

    return h00*p0 + h10 * m0 + h01 * p1 + h11 * m1

cdef linear_interp(p0, p1, float x, float x_lo, float x_hi):
    cdef float x_dist = (x_hi-x_lo)**-1
    cdef float t = (x-x_lo)*x_dist
    return p0 * (1-t) + p1*t

cdef class Vocoder:
    def __init__(self, filename, frame_size=2048, hop_size=256, hop_size_out = 128):
        self.framer = framer.NpFramer(filename, frame_size = frame_size, hop_size=hop_size, layout='mono')
        self.fft = ReFFT(frame_size, ss.hann(frame_size))
        self.__hop_size = hop_size
        self.__hop_size_out = hop_size_out
        self.__frame_size = frame_size
        self.__max_frames = 64
        self.__time_ratio = 1.0
        self.spec = [self.fft.process(next(_ for _ in self.framer if _.std() > 0.125))]
        self.__spec_acc = ReSpectrum(frame_size)
        self.__time_in    = self.framer.next_sample - self.frame_size/2
        self.__time_origin= -self.__time_in
        self.__time_out   = 0
        self.spec[0].when = self.input_time_nominal
        self.__accumulator = np.zeros((frame_size*4,),dtype=np.float32)
        self.__windowAccumulator = np.zeros((frame_size*4,),dtype=np.float32)
        self.analyze_frame()
    cpdef analyze_frame(self):
        f = next(self.framer)
        self.spec.append(self.fft.process(f, None,self.framer.next_sample - self.__frame_size//2))
        if len(self.spec) > self.__max_frames:
            self.spec = self.spec[-self.__max_frames:]

    cpdef interpolate_spec(self, when):
        while(self.spec[-1].when < when):
            self.analyze_frame()

        last_spec = self.spec[-1]
        prev_spec = self.spec[-2]

        cdef ReSpectrum res = ReSpectrum(self.__frame_size)
        res.size = self.__frame_size
        res.when = when
        res.M[::] = cubic_hermite(
            prev_spec.M
          , prev_spec.dM_dt
          , last_spec.M
          , last_spec.dM_dt
          , when,prev_spec.when
          , last_spec.when
            )
        res.dPhi_dt[::] = linear_interp (
            prev_spec.dPhi_dt
          , last_spec.dPhi_dt
          , when
          , prev_spec.when
          , last_spec.when
            );
        res.dPhi_dw[::] = linear_interp (
            prev_spec.dPhi_dw
          , last_spec.dPhi_dw
          , when
          , prev_spec.when
          , last_spec.when
            );
        return res
    cpdef advance_spec(self, ReSpectrum spec):
        x_dist = spec.when - self.spec_acc.when
        spec.Phi[::] = self.spec_acc.Phi + x_dist * (self.spec_acc.dPhi_dt + spec.dPhi_dt)*0.5
        spec.Phi[::] += self.unit * x_dist
        spec.Phi[::] = np.fmod(spec.Phi[::],np.pi*2)
        return spec

    cpdef synthesize_frame(self, ReSpectrum spec):
        res = self.fft.synthesize(spec)
        self.__accumulator[:len(res)][::] += res
        self.__windowAccumulator[:len(res)] += self.fft.h ** 2
        out = self.__accumulator[:self.hop_size_out] / (self.__windowAccumulator[:self.hop_size_out]+1e-8)
        self.__accumulator[:-self.hop_size_out] = self.__accumulator[self.hop_size_out:]
        self.__accumulator[-self.hop_size_out:] = 0
        self.__windowAccumulator[:-self.hop_size_out] = self.__windowAccumulator[self.hop_size_out:]
        self.__windowAccumulator[-self.hop_size_out:] = 0
        self.__time_out += self.hop_size_out
        self.__spec_acc = spec
        return out

    def next_frame(self):
        when = self.input_time_nominal
        spec = self.interpolate_spec(when)
        spec.when = self.__time_out + self.hop_size_out
        self.advance_spec(spec)
        return self.synthesize_frame(spec)
    property unit:
        def __get__(self):
            if self.__unit is None or len(self.__unit) != self.frame_size:
                self.__unit = np.arange(self.frame_size//2 + 1) * 2 * np.pi / self.frame_size
            return self.__unit
    property input_time_nominal:
        def __get__(self):
            return (self.__time_out - self.__time_origin) / self.__time_ratio

    property frame_size:
        def __get__(self):
            return self.__frame_size
        def __set__(self, size):
            if self.fft.size != size:
                self.spec.clear()
                self.fft.size = size
            self.__frame_size = size
    property hop_size:
        def __get__(self): return self.__hop_size
    property hop_size_out:
        def __get__(self): return self.__hop_size_out
    property spec_acc:
        def __get__(self): return self.__spec_acc
    property time_ratio:
        def __get__(self):return self.__time_ratio
        def __set__(self,val):self.__time_ratio = val
