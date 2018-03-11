# cython: np_pythran=False

from libcpp cimport vector, map as std_map, unordered_map, deque
from libcpp.string cimport string  as std_string

from rmwarp.framer cimport NpFramer, Framer
#cimport respectrum
from .respectrum cimport ReSpec
from .refft cimport ReFFT
from .basic cimport time_derivative_window
from .basic cimport time_weighted_window
from .kaiserwindow import cxx_kaiser_window
from .interp cimport linear_interp,cubic_hermite,windowed_diff,find_runs,find_tagged_runs,binary_smooth,binary_opening,binary_closing,binary_dilation,binary_erosion
import weakref
import numpy as np
cimport numpy as np
import scipy as sp, scipy.signal as ss

cdef class ReOdf:
    def __init__(self
        , rate       = 44100
        , frame_size = 2048
        , hop_size   = 512
        , shaping    = 2.4
        , d2Phi_threshold  = 0.25
        , lgd_threshold    = None
        , opening = 4
        , closing = 4
            ):
        self.__rate = rate
        self.__frame_size = frame_size
        self.__hop_size   = hop_size
        self.__shaping    = shaping
        self.__d2Phi_threshold  = d2Phi_threshold
        self.__opening = opening
        self.__closing = closing
        if lgd_threshold is None:
            lgd_threshold = 1.
        self.__lgd_threshold  = lgd_threshold
        self.__when       = 0
        window = np.asarray(cxx_kaiser_window(frame_size, shaping).astype(np.float32))
        self.fft = ReFFT(frame_size)
        self.fft.set_window(window)
        self.spec = []
        self.neigh = np.zeros(shape=(64,),dtype=np.float32)

    def process(self, data):
        spec = None
        if len(self.spec) > 2:
            spec = self.spec.pop(0)
        spec = self.fft.process(data, spec, self.__when)
        self.spec.append(spec)
        self.__when += self.__hop_size
        if len(self.spec) > 1:
            spec.unwrap_from(self.spec[-2])

        lgd_table = -spec.dPhi_dw/self.__hop_size
        nz = binary_closing(
            binary_opening(
                ((np.abs(spec.d2Phi_dtdw+1.0) < self.__d2Phi_threshold)
            * (np.abs(lgd_table) < self.__lgd_threshold)),self.__opening),self.__closing)
        for _ in nz.nonzero()[0]:
            off = int(np.floor(lgd_table[_]) + len(self.neigh)//2)
            if 0 <= off and off < len(self.neigh):
                self.neigh[off] += 1

        self.neigh[:-1] = self.neigh[1:]
        self.neigh[-1] = 0
        return self.neigh[0]

    @property
    def rate(self):
        return self.__rate

    @property
    def frame_size(self):
        return self.__frame_size

    @property
    def hop_size(self):
        return self.__hop_size

    @property
    def shaping(self):
        return self.__shaping

    @property
    def when(self):
        return self.__when

cdef class ReTrack:
    def __init__(self
        , framer
        , acf_duration = 6.
        , acf_size = None
        , tempo_count = None
        , tempo_func  = None
        , tempo_table = None
        , tempo_start = None
        , tempo_step  = None
        , tempo_sigma = 8. ** -1
        , cum_decay = 0.9
        , cum_sigma = 5. ** -1
        , kernel_threshold = 1e-3
        ):
        self.framer = framer
        self.odf = ReOdf(
            rate = framer.rate
          , frame_size = framer.frame_size
          , hop_size = framer.hop_size
            )

        if acf_size is None:
            acf_size = int(self.rate * acf_duration / self.hop_size)
        else:
            acf_duration = self.hop_size * acf_size / self.rate

        self.acf_duration = acf_duration
        self.acf_size     = acf_size

        self.odf_buf = np.zeros(shape=(2 * acf_size,),dtype=np.float32)
        self.acf_buf = np.zeros(shape=(2 * acf_size,),dtype=np.float32)

        if tempo_table is not None:
            tempo_table = np.asarray(
                tempo_table
              , dtype=np.float64
                )
        elif tempo_count is not None:
            if tempo_func is not None:
                tempo_table = np.fromiter(map(tempo_func,range(tempo_count)),dtype=np.float64)
            elif tempo_step is not None and tempo_start is not None:
                tempo_table = np.arange(tempo_count,dtype=np.float64) * tempo_step + tempo_start
        if tempo_table is None:
            raise ValueError("tempo_table must happen *some*how...")
        self.tempo_table = tempo_table
        self.tempo_count = len(self.tempo_table)

        self.comb_out  = np.zeros(shape=(self.tempo_count // 4,),dtype=np.float32)
        self.tempo_obs = np.zeros_like(self.tempo_table)
        self.delta_buf = np.ones_like(self.tempo_table)

    def induct_tempo(self):
        pass

    @property
    def rate(self):
        return self.odf.rate

    @property
    def frame_size(self):
        return self.odf.frame_size

    @property
    def hop_size(self):
        return self.odf.hop_size

