import pyximport

pyximport.install(reload_support=True,language_level=3, setup_args={
    'options':{
        'build_ext':{
            'cython_directives':{
                'language_level':3,
                'embedsignature':True,
                'cdivision':True,
                'infer_types':True,
                'always_allow_keywords':True,
                'c_string_encoding':'ascii',
            },
            'cython_cplus':True,

        },
    },
})
import numpy as np, scipy as sp, pyfftw as fftw, scipy.signal as ss
from .basic import time_derivative_window, time_weighted_window
from .rmspectrum import RMSpectrum
from .rmfft import RMFFT
from .windows import kbd_window, xiph_vorbis_window
from .framer import NpFramer, NpImageFramer, NpImageSpectrogram, Spectrogram, vamp_collect, HighFrequencyAudioCurve, get_audio, DFCollection, Framer, av, frequency_domain_window
from .kaiserwindow import cxx_kaiser_window

from .refft import ReFft
from .respectrum import ReSpec
