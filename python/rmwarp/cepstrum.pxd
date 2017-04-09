# -*- coding: utf-8 -*-

import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss, scipy.special as ssp
cimport numpy as np


cpdef real_cepstrum(x, n = *)
cpdef inverse_real_cepstrum(x, n = *)
cpdef minimum_phase_cepstrum(x, n = *)
cpdef smooth_cepstrum(x, bw = *, n = *,)

