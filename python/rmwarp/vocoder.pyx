from libcpp cimport vector, map as std_map, unordered_map, deque
from libcpp.string cimport string  as std_string

import framer
cimport respectrum
from respectrum cimport ReSpec
from refft cimport ReFFT
from basic cimport time_derivative_window
from basic cimport time_weighted_window
from kaiserwindow import cxx_kaiser_window
import weakref
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss

cpdef cubic_hermite( p0, m0, p1, m1, float x,float x_lo,float x_hi):
    cdef float x_dist = (x_hi-x_lo)**-1
    cdef float t = (x-x_lo)*x_dist
    cdef float h00 = 2*t**3-3*t**2+1
    cdef float h10 = (t**3-2*t**2+t)*x_dist
    cdef float h01 = -2*t**3 + 3*t**2
    cdef float h11 = (t**3-t**2)*x_dist

    return h00*p0 + h10 * m0 + h01 * p1 + h11 * m1

cpdef linear_interp(p0, p1, float x, float x_lo, float x_hi):
    cdef float x_dist = (x_hi-x_lo)**-1
    cdef float t = (x-x_lo)*x_dist
    return p0 * (1-t) + p1*t

cpdef np.ndarray windowed_diff(np.ndarray a, int w):
    cdef int l = a.shape[0]
    cdef np.ndarray res = np.zeros_like(a)
    if w >= l:
        res[::] = a[-1]
        return res
    if l > 2 * w:
        res[:w]   = a[w:2*w];
        res[w:-w] = a[2*w:] - a[:-2*w]
        res[-w:]  = a[-1]   - a[-w*2:-w]
    else:
        res[:-w] = a[w:]
        res[-w:] = a[-1] - a[:w]
    return res

cpdef np.ndarray binary_dilation(np.ndarray a, int w):
    cdef np.ndarray res = a != 0
    for x in range(1, w):
        res[1:]  += res[:-1]
        res[:-1] += res[1:]
    return res

cpdef np.ndarray binary_erosion(np.ndarray a, int w):
    cdef np.ndarray res = a != 0
    for x in range(1,w):
        res[1:]  *= res[:-1]
        res[:-1] *= res[1:]
    return res

cpdef np.ndarray binary_closing(np.ndarray a, int w):
    return binary_erosion(binary_dilation(a,w),w)

cpdef np.ndarray binary_opening(np.ndarray a, int w):
    return binary_dilation(binary_erosion(a,w),w)

cpdef np.ndarray binary_smooth(np.ndarray a, int w):
    return ~binary_closing(~binary_closing(a,w),w)

cpdef np.ndarray find_runs(np.ndarray a):
    shape = [0] * a.ndim
    for x in range(a.ndim):
        shape[x] = a.shape[x]
    cdef np.ndarray asbool = np.zeros(shape=tuple([shape[0]+2,] + shape[1:]),dtype=np.bool8)
    asbool[1:-1] = a!=0
    changes = (asbool[1:] ^ asbool[:-1]).nonzero()[0]
    changes = changes.reshape(changes.shape[0]//2,2)
    changes[::,1] += 1
    return changes.astype(np.int32)

cdef class Vocoder:
    def __init__(self, filename, frame_size=2048, hop_size=256, hop_size_out = 256, shaping=2.4,reset_width=12,reset_granularity=8):
        self.framer = framer.NpFramer(filename, frame_size = frame_size, hop_size=hop_size, layout='mono',pad=True)
        self.__frame_size = frame_size
        self.__window = np.asarray(cxx_kaiser_window(self.frame_size, shaping)).astype(np.float32)
        self.fft = ReFFT(frame_size)
        self.fft.set_window(self.window)
#        self.fft.epsilon = 10 ** - 8
        self.__hop_size = hop_size
        self.__hop_size_out = hop_size_out
        self.__shaping           = shaping
        self.__reset_granularity = reset_granularity
        self.__reset_width       = reset_width
        self.__frame_index       = 0
        self.__max_frames = 64
        self.__time_ratio = 1.0
        frames = len(self.framer)
        resets = ((self.fft.coef + self.__reset_granularity-1)//self.__reset_granularity)
        self.__resets = resets
        self.__M_table           = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__Phi_table         = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__d2Phi_dtdw_table  = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__lgd_table         = np.zeros((frames,self.fft.coef),dtype=np.float32)
        self.__onset_table       = np.zeros((frames,self.fft.coef),dtype=np.bool8)
        self.__reset_list        = [[(0,0)] for _ in range(self.__resets)]
        self.__reset_segs        = [list() for _ in range(frames)]
        self.__reset_curr        =  np.zeros((self.fft.coef,),dtype=np.int32)
        self.__reset_view        = -np.ones((self.fft.coef,),dtype=np.int32)
        self.__reset_last_idx    = -np.ones((self.fft.coef,),dtype=np.int32)
        self.__reset_last_ref    = np.zeros((self.fft.coef,),dtype=np.int32)
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
        windowed_weight = windowed_diff(spec.weight,self.__reset_width)
        windowed_weight = np.where(windowed_weight, windowed_weight, 1.) ** -1
        self.__d2Phi_dtdw_table[idx] = windowed_diff(spec.d2Phi_dtdw_acc, self.__reset_width) * windowed_weight
        self.__lgd_table[idx]        = windowed_diff(spec.local_group_delay_acc,self.__reset_width) * windowed_weight/self.__hop_size
        self.__onset_table[idx] = (np.abs(self.__d2Phi_dtdw_table[idx]+1.0) < 0.25)
        __onset_table = (np.abs(self.__lgd_table[idx])<0.6) * (np.abs(self.__d2Phi_dtdw_table[idx]+1.0) < 0.125)
        cdef int depth  = self.frame_size // self.hop_size
        nz = __onset_table.nonzero()
        if nz:
            self.__reset_view[nz] = idx
            if self.__oldest_active < idx - depth:
                self.__oldest_active = idx
        cdef int max_hole = self.__reset_granularity * 2
        cdef int min_keep = self.__reset_granularity
        cdef int lookbehind = 8 * self.frame_size // self.hop_size
        cdef int stopbefore
        cdef int segidx
        cdef int oldest = self.__oldest_active
        cdef int[:,:] runs
        cdef int run_beg, run_end
        cdef int seg_beg, seg_end
        cdef int tidx
        cdef int widx
        if oldest >= 0 and oldest <= idx - depth:
            for run_beg,run_end in find_runs(binary_closing( self.__reset_view >= oldest, max_hole)):
                if run_end - run_beg < min_keep:
                    continue
                uvalues, ucounts = np.unique(self.__reset_view[run_beg:run_end],return_counts=True)
                if not oldest in uvalues:
                    continue
                offset = np.searchsorted(uvalues,oldest)
                segidx = uvalues[offset+ucounts[offset:].argmax()]

                stopbefore = max((oldest - lookbehind,0))
                self.__reset_last_ref[seg_beg:seg_end] = segidx
                row = np.ones_like(self.__onset_table[segidx,run_beg:run_end])
                rows = [(segidx,row)]
                for widx in range(segidx-1,stopbefore,-1):
                    row = binary_closing(self.__onset_table[widx,run_beg:run_end] * row,max_hole) * (widx > self.__reset_last_idx[run_beg:run_end])
                    row = ~binary_closing(~row,min_keep)
                    if row.any():
                        rows.append((widx,row))
                    else:
                        break
                blk = np.zeros_like(row)
                while rows:
                    widx,row = rows.pop(-1)
                    lst = self.__reset_segs[widx]
                    for seg_beg,seg_end in (find_runs(row * ~blk)):
                        seg_beg += run_beg
                        seg_end += run_beg
                        lst.append((seg_beg,seg_end,segidx))
                        self.__reset_last_idx[seg_beg:seg_end] = widx
                    blk = row
                self.__reset_view[run_beg:run_end] = -1
            tmp = self.__reset_view[self.__reset_view > oldest]
            if len(tmp):
                self.__oldest_active = tmp.min()
            else:
                self.__oldest_active = -1
#        self.__onset_table[idx] = ~binary_closing(
#            ~binary_closing(
#                              , self.__reset_width
#              ), 2*self.__reset_granularity)

        cdef int nidx
        for rlo,rhi in find_runs(__onset_table):
            if idx > 0 and np.all(self.__onset_table[idx-1][rlo:rhi]):
                continue

            tidx = idx - 1

            lo = rlo // self.__reset_granularity
            hi = rhi // self.__reset_granularity
            while tidx >= 0 and tidx >= idx - lookbehind and np.count_nonzero(self.__lgd_table[tidx,rlo:rhi] >= -0.6) > (rhi-rlo) * 0.75:
                tidx -= 1
            tidx += 1
            for k in range(lo,hi):
                lst = self.__reset_list[k]
                if lst[-1][0] >= tidx - 1 or lst[-1][1] >= tidx:
                    lst[-1] = (lst[-1][0],idx)
                else:
                    lst.append((tidx,idx))

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
