# cython: np_pythran=False

import av, numpy as np, scipy as sp, scipy.signal as ss, scipy.fftpack as sf
cimport numpy as np
import scipy.fftpack as fp
import itertools as it

cdef class Framer:
    def __init__(self, filename, frame_size, hop_size = 0,rate=0,layout=None,dtype=np.float32, pad = False, *args, **kwargs):
        if not hop_size:
            hop_size = frame_size / 4

        self.frame_size = int(frame_size)
        self.hop_size   = int(hop_size)
        self.c = av.open(filename)
        self.s = next(s for s in self.c.streams if s.type is u'audio' or s.type is b'audio')
        self.f = av.AudioFifo()

        if not layout:
            layout = self.s.layout

        self.r = av.AudioResampler(format='fltp',layout=layout,rate=rate or self.s.rate)
        self.d = self.c.demux(self.s)
        while self.f.samples <  self.frame_size * 2:
            for frm in next(self.d).decode():
                frm = self.r.resample(frm)
                if pad:
                    tmp = av.AudioFrame(format=frm.format,layout=frm.layout,samples=self.frame_size//2)
                    tmp.rate = frm.rate
                    for plane in tmp.planes:
                        plane.update('\x00' * plane.buffer_size)
                    tmp.pts = frm.pts - frm.samples / frm.rate / frm.time_base
                    self.f.write(tmp)
                    pad = False
                self.f.write(frm)

    cpdef read(self, frame_size = 0):
        if not frame_size:
            frame_size = self.frame_size
        while self.f.samples < frame_size + self.frame_size:
            for frm in next(self.d).decode():
                self.f.write(self.r.resample(frm))
        ret = self.f.peek(self.frame_size,False)
        self.f.drain(self.hop_size)
        if ret:
            ret.pts = self.next_sample - ret.samples
        return ret

    def __next__(self):
        return self.read()

    def __iter__(self):
        return self

    cpdef seek(self, pts):
        self.c.seek(float(pts))
        self.d.close()
        self.d = self.c.demux(self.s)
        self.f.drain(0)
        while self.f.samples <  self.frame_size * 2:
            for frm in next(self.d).decode():
                self.f.write(self.r.resample(frm))

    def seek_sample(self, pts):
        self.seek(float(pts/(self.f.rate)))

    def __len__(self):
        return int(self.frames)

    @property
    def samples(self):
        if self.s.frames and self.s.frame_size:
            return int(self.s.frame_size * self.s.frames)
        else:
            return int(self.s.duration * self.s.time_base * self.s.rate)
    @property
    def frames(self):
        return int((self.samples - self.frame_size)//self.hop_size)
    @property
    def duration(self):
        return self.c.duration / av.time_base
    @property
    def next_pts(self):
        return self.f.next_pts
    @property
    def next_sample(self):
        return float(self.next_pts * (self.f.rate * self.f.time_base))
    @property
    def rate(self):
        return self.r.rate
    @property
    def format(self):
        return self.r.format
    @property
    def layout(self):
        return self.r.layout

cdef class NpFramer(Framer):
    def __init__(self,filename,frame_size, hop_size = 0, layout=None,transpose=True, dtype=None, pad = False, *args, **kwargs):
        super(NpFramer,self).__init__(filename,frame_size=frame_size,hop_size=hop_size,layout=layout, pad=pad)
        self.transpose = transpose
        self.dtype = dtype or np.float32

    cpdef read(self,frame_size = 0):
        if self.transpose:
            x = super(NpFramer,self).read(frame_size).to_ndarray().T.astype(self.dtype)
        else:
            x =super(NpFramer,self).read(frame_size).to_ndarray().astype(self.dtype)
        if len(x.shape) > 1 and x.shape[1] == 1:
            x=x.reshape(x.shape[0])
        elif len(x.shape) > 1 and x.shape[0] == 1:
            x=x.reshape(x.shape[1])
        return x

cdef class NpImageFramer(Framer):
    def __init__(self, filename, frame_size, width, hop_size = 0, dtype=None, transpose=True, pad = False, *args, **kwargs):
        if dtype is None:
            dtype = np.float32
        super(NpImageFramer,self).__init__(filename,*args, frame_size=frame_size,hop_size=hop_size,layout='mono', pad = pad, **kwargs)
        self.image = np.ndarray((width,frame_size),dtype=dtype)
    cpdef read(self, frame_size = 0):
        x = super(NpImageFramer,self).read().to_ndarray()
        if len(x.shape) > 1 and x.shape[1] == 1:
            x=x.reshape(x.shape[0])
        elif len(x.shape) > 1 and x.shape[0] == 1:
            x=x.reshape(x.shape[1])
        self.image[0:,::] = self.image[1:,::]
        self.image[-1,::] = x[::]
        return x

cdef class Spectrogram(NpFramer):
    def __init__(self, filename, frame_size, hop_size = 0, pad = False):
        super().__init__(filename,frame_size,hop_size, layout=av.AudioLayout(1),transpose=False, pad=pad)
        self.h = sf.fftshift(ss.hann(self.frame_size))

    cpdef read(self, frame_size = 0):
        x = sf.fftshift(super(Spectrogram,self).read(),axes=-1)
        x *= self.h
        return sf.fft(x)

cdef class NpImageSpectrogram(Framer):
    def __init__(self, filename, frame_size, width, hop_size = 0, dtype=None, transpose=True, pad = False):
        if dtype is None:
            dtype = np.complex64
        super(NpImageSpectrogram,self).__init__(filename,frame_size,hop_size,'mono', pad=pad)
        self.real_size = frame_size//2+1
        self.h = sf.fftshift(ss.hann(self.frame_size))
        self.image = np.ndarray((width,self.real_size),dtype=dtype)

    cpdef read(self, frame_size = 0):
        x = super(NpImageSpectrogram,self).read().to_ndarray()
        if len(x.shape) > 1 and x.shape[1] == 1:
            x=x.reshape(x.shape[0])
        elif len(x.shape) > 1 and x.shape[0] == 1:
            x=x.reshape(x.shape[1])
        x = sf.fft(sf.fftshift(x) * self.h)
        self.image[0:,::] = self.image[1:,::]
        self.image[-1,::] = x[:self.real_size]
        return x

cdef class RMSpectrogram(NpFramer):
    def __init__(self, filename, frame_size,hop_size = 0):
        super().__init__(filename,frame_size,hop_size,av.AudioLayout(1), False)
        self.h = sf.fftshift(ss.hann(self.frame_size))
        self.Th = self.h * sf.fftshift(
            np.linspace(
                -self.frame_size/2
              , self.frame_size/2
              , self.frame_size
              , endpoint=False
                )
            )
        self.unit = sf.fftshift(
                np.linspace(
                    -1
                  ,  1
                  , self.frame_size
                  , endpoint=False
                    )
                )
        self.Dh = sf.ifft(sf.fft(self.h) * self.unit)
        self.unit *= 2 * np.pi / self.frame_size
        self.complex_unit = complex(0,1)**self.unit

    cpdef read(self, frame_size = 0):
        next_pos = self.next_pts + self.frame_size/2
        chunk = sf.fftshift(super(RMSpectrogram,self).read(),axes=-1)
        real_len = self.frame_size // 2 + 1
        X = sf.fft(chunk * self.h)[:real_len]
        X_Dh = sf.fft(chunk * self.Dh)[:real_len]
        X_Th = sf.fft(chunk * self.Th)[:real_len]

        mag  = np.abs(X)
        dt = -((X_Th * X.conj()) / np.square(mag+1e-6)).real
        dw =  ((X_Dh * X.conj()) / np.square(mag+1e-6)).imag
        X *= (self.complex_unit[:real_len])**next_pos
        phase = np.angle(X)
        sel = mag >= max((2**-22),mag.max() * (10 ** -4))
        dt = np.where(sel, dt,0.)
        dw = np.where(sel, dw,0.)
        return X,(mag,phase), dt, dw


cdef class DFCollection:
    def __init__(self, funcs, filename, frame_size = 2048, hop_size = 256):
        self.funcs = funcs
        self.curves = [list() for func in funcs]
        self.spectrogram = Spectrogram(filename,frame_size,hop_size)
    def process(self):
        spectrum = next(self.spectrogram).sum(axis=0)
        for func, curve in zip(self.funcs,self.curves):
            curve.append(func(spectrum))

    def process_all(self):
        for spectrum in map(lambda x:x.sum(axis=0),self.spectrogram):
            for func, curve in zip(self.funcs,self.curves):
                curve.append(func(spectrum))

def HighFrequencyAudioCurve(mag):
    bins = len(mag)//4
    return (np.abs(mag[:bins]) * np.arange(0, bins)).sum() * 8/(bins *bins)

def get_audio(fn, dtype = None):
    c = av.open(fn)
    d = c.demux(next(s for s in c.streams if s.type is 'audio'))
    if dtype is not None:
        ret = np.vstack(f.to_ndarray().astype(dtype) for f in it.chain.from_iterable(p.decode() for p in d))
    else:
        ret = np.vstack(f.to_ndarray() for f in it.chain.from_iterable(p.decode() for p in d))
    d.close()
    return ret

def frequency_domain_window(data, lag = 0):
    coef = 0.25 * (np.e ** complex(0, (np.pi * 2 * lag) / len(data)))
    ret = data * 0.5
    for i in range(data.shape[0]):
        ret[i] += coef * data[(i+1)%len(data)] + coef.conjugate() *data[(i-1)]
    return ret;

def vamp_collect(features):
    import collections
    midval = collections.defaultdict(lambda : collections.defaultdict(list))
    for f in features:
        for key, val in f.items():
            for k,v in val.items():
                midval[key][k].append(v)
    return { key: { k: np.asarray(v) for k,v in val.items()} for key,val in midval.items()}
