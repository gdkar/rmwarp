
#cython: language_level=3
#cython: extension_language=c++
import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss
from .rmspectrum import RMSpectrum
from .rmspectrum cimport RMSpectrum
from . import basic

cimport numpy as np
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, weak_ptr, unique_ptr, allocator
from libcpp.cast cimport *

