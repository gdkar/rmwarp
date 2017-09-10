from libcpp cimport vector, map as std_map, unordered_map, deque
from libcpp.string cimport string  as std_string

from rmwarp.framer cimport NpFramer, Framer
#cimport respectrum
from .respectrum cimport ReSpec
from .refft cimport ReFFT
from .basic cimport time_derivative_window
from .basic cimport time_weighted_window
from .kaiserwindow import cxx_kaiser_window
from .interp cimport linear_interp,cubic_hermite,windowed_diff,find_runs,find_tagged_runs
import weakref
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss


cdef class Vocoder:
    def __init__(self
        , filename
        , frame_size=2048
        , hop_size=256
        , hop_size_out = 256
        , shaping=2.4
        , reset_width=12
        , reset_granularity=8
        , d2Phi_threshold=0.25
        , lgd_threshold=1.0
        , d2Phi_threshold_accute = 0.125
        , lgd_threshold_accute=0.6
        , weight_width=16):
        self.framer = NpFramer(filename, frame_size = frame_size, hop_size=hop_size, layout='mono',pad=True)
        self.__frame_size = frame_size
        self.__window = np.asarray(cxx_kaiser_window(self.frame_size, shaping)).astype(np.float32)
        self.fft = ReFFT(frame_size)
        self.fft.set_window(self.window)
#        self.fft.epsilon = 10 ** - 8
        self.__hop_size                 = hop_size
        self.__hop_size_out             = hop_size_out
        self.__shaping                  = shaping
        self.__reset_granularity        = reset_granularity
        self.__reset_width              = reset_width
        self.__weight_width             = weight_width
        self.__d2Phi_threshold          = d2Phi_threshold
        self.__lgd_threshold            = lgd_threshold
        self.__d2Phi_threshold_accute   = d2Phi_threshold_accute
        self.__lgd_threshold_accute     = lgd_threshold_accute

        self.__frame_index              = 0
        self.__max_frames               = 64
        self.__time_ratio               = 1.0

        frames = len(self.framer)
        resets = ((self.fft.coef + self.__reset_granularity-1)//self.__reset_granularity)
        self.__resets = resets
        self.__M_table           = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__Phi_table         = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__d2Phi_dtdw_table  = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__lgd_table         = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__onset_table       = np.zeros((frames,self.fft.coef),dtype=np.bool8)
        self.__reset_list        = [[(0,0)] for _ in range(self.__resets)]
        self.__reset_segs        = list()
        self.__reset_ring        = list()
        self.__reset_curr        =  np.zeros((self.fft.coef,),dtype=np.int32)
        self.spec = []
        self.__time_origin= 0
        self.__time_out   = 0
        self.__accumulator = np.zeros((frame_size*4,),dtype=np.float32)
        self.__windowAccumulator = np.zeros((frame_size*4,),dtype=np.float32)
        self.__windowAccumulator[0] = 1
        self.__unit = (np.arange(self.frame_size//2 + 1) * 2 * np.pi / self.frame_size).astype(np.float32)
        self.analyze_frame()

    cpdef analyze_frame(self):
        f = next(self.framer)
        when = self.framer.next_sample - self.__frame_size//2
        if len(self.spec) >= self.__max_frames:
            spec = self.spec.pop(0)
        else:
            spec = None
        self.spec.append(self.fft.process(f, spec, when))
        spec = self.spec[-1]
        cdef int idx = self.__frame_index
        self.__frame_index += 1
        cdef float pi2 = np.pi * 2
        cdef float over_pi2 = pi2 ** -1

        self.__M_table[idx]          = spec.M
        windowed_weight = windowed_diff(spec.weight,self.__weight_width)
        windowed_weight = np.where(windowed_weight, windowed_weight, 1.) ** -1
        self.__d2Phi_dtdw_table[idx] = windowed_diff(spec.d2Phi_dtdw_acc, self.__weight_width) * windowed_weight
        self.__lgd_table[idx]        = windowed_diff(spec.local_group_delay_acc,self.__weight_width) * windowed_weight/self.__hop_size
        self.__onset_table[idx] = (#~binary_closing(
#            ~binary_closing(
                (np.abs(self.__lgd_table[idx])<self.__lgd_threshold_accute)
              * (np.abs(self.__d2Phi_dtdw_table[idx]+1.0) < self.__d2Phi_threshold_accute))
#            , self.__reset_width
#              ), 2*self.__reset_granularity)

        cdef int nidx, tidx, rlo, rhi, lo, hi
        cdef int lookbehind = self.frame_size * 16// self.hop_size
        cdef int ring_size = self.frame_size * 2 // self.hop_size
        runs = find_runs(self.__onset_table[idx])
        if runs.shape[0]:
            self.__reset_ring.append(runs)
            if len(self.__reset_ring) > ring_size:
                rf = self.__reset_ring.pop(0)
        """
        for rlo,rhi in find_runs(self.__onset_table[idx]):
            lo = rlo // self.__reset_granularity
            hi = rhi // self.__reset_granularity
            tfirst = np.ones_like(self.__onset_table[idx,rlo:rhi])
            for tidx in range(idx, max(-1,idx - lookbehind), -1):
                tonset = (self.__lgd_table[tidx,rlo:rhi] >=  self.__lgd_threshold) * (np.abs(self.__d2Phi_dtdw_table[tidx,rlo:rhi] + 1.0) < self.__d2Phi_threshold) * tfirst
                tfirst = ~binary_closing(~binary_closing(tonset, self.__reset_width),self.__reset_granularity)
                here = np.unique((tfirst.nonzero()[0])//self.__reset_granularity) + lo
                if len(here):
                    for k in here:
                        lst = self.__reset_list[k]
                        if lst[-1][0] >= tidx - 1 or lst[-1][1] >= tidx:
                            lst[-1] = (tidx,idx)
                        else:
                            lst.append((tidx,idx))
                else:
                    break
        """
        if idx <= 0:
            self.__Phi_table[idx] = spec.Phi
            correction = None
        else:
            dPhi = (self.unit - 0.5*(self.spec[-1].dPhi_dt + self.spec[-2].dPhi_dt)) * self.hop_size
            unwrapped = self.__Phi_table[idx-1] + dPhi
            self.__Phi_table[idx] = spec.Phi + pi2 * np.round((unwrapped - spec.Phi)*over_pi2)

    def make_frame(self, double pts):
        cdef double frac = (pts - self.__time_origin) * self.rate / self.__hop_size
        cdef int idx          = int(np.floor(frac))
        try:
            ahead = self.frame_size * 64 / self.hop_size
            while idx + ahead >= self.__frame_index:
                self.analyze_frame()
        except StopIteration:
            if idx >= self.__frame_index:
                raise
#        frac -= <double>idx
        cdef np.ndarray M   = linear_interp(self.__M_table[idx],self.__M_table[idx+1],frac,idx+0.0,idx+1.0)
        cdef np.ndarray Phi = linear_interp(self.__Phi_table[idx],self.__Phi_table[idx+1],frac,idx+0.0,idx+1.0)
        cdef np.ndarray ref = np.zeros_like(Phi)
        cdef int x
        for x,(r,lst) in enumerate(zip(self.__reset_curr,self.__reset_list)):
            while r + 1 < len(lst):
                if lst[r+1][0] <= idx:
                    r += 1
                    self.__reset_curr[x] = r
                else:
                    break
            rx = lst[r][1]
            lo = x * self.__reset_granularity
            hi = (x+1)* self.__reset_granularity
            ref[lo:hi] = self.__Phi_table[rx][lo:hi]
        Phi = ref + (Phi-ref) * self.time_ratio
        return self.fft.inverse(M[::],Phi[::])

    def synthesize_frame(self):
        cdef double time_out = self.__time_out
        cdef int hop_size_out = self.hop_size_out
        self.__time_out += hop_size_out
        cdef int frame_size = self.frame_size
        cdef np.ndarray res = self.make_frame(time_out)
        self.accumulator[:frame_size]       += res * self.window[::]
        self.windowAccumulator[:frame_size] += self.window ** 2
        cdef np.ndarray norm = self.windowAccumulator[:hop_size_out]
        norm = np.where(norm, norm, 1.0) ** -1
        cdef np.ndarray out = self.accumulator[:hop_size_out] * norm
        self.accumulator[:-hop_size_out] = self.accumulator[hop_size_out:]
        self.accumulator[-hop_size_out:] = 0
        self.windowAccumulator[:-hop_size_out] = self.windowAccumulator[hop_size_out:]
        self.windowAccumulator[-hop_size_out:] = 0
        return np.where(~np.isnan(out),out, 0.0)

    def __next__(self):
        return self.synthesize_frame()

    def __iter__(self):
        return self

    @property
    def unit(self):
        return np.asarray(self.__unit)

    @property
    def rate(self):
        return self.time_ratio ** -1

    @rate.setter
    def rate(self,val):
        self.time_ratio = val ** -1

    @property
    def input_time(self):
        return (self.__time_out - self.__time_origin) * self.rate

    @input_time.setter
    def input_time(self,val):
        val = int(val)
        self.__time_origin = self.__time_out - val * self.time_ratio
        itime = self.input_time // self.hop_size
        cdef int x
        cdef int r
        for x,lst in enumerate(self.__reset_list):
            r = self.__reset_curr[x]
            while r > 0 and lst[r][0] >= itime:
                r -= 1
            self.__reset_curr[x] = r

    @property
    def output_time(self):
        return self.__time_out

    @property
    def frame_size(self):
        return self.__frame_size

    @property
    def hop_size(self):
        return self.__hop_size

    @property
    def hop_size_out(self):
        return self.__hop_size_out

    @hop_size_out.setter
    def hop_size_out(self, val):
        cdef int ival = int(val)
        if ival > 0 and ival <= self.frame_size / self.bandwidth:
            self.__hop_size_out = ival

    @property
    def shaping(self):
        return self.__shaping

    @property
    def bandwidth(self):
        return (self.__shaping**2.0+1)**0.5
    @property
    def time_ratio(self):
        return self.__time_ratio

    @time_ratio.setter
    def time_ratio(self,val):
        itime = self.input_time
        self.__time_ratio = val
        self.input_time = itime

    @property
    def reset_granularity(self):
        return self.__reset_granularity

    @property
    def reset_width(self):
        return self.__reset_width
    @property
    def window(self):
        return self.__window
    @property
    def accumulator(self):
        return self.__accumulator
    @property
    def windowAccumulator(self):
        return self.__windowAccumulator
