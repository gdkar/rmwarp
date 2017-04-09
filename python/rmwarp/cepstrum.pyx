import pyfftw as fftw, numpy as np, scipy as sp, scipy.signal as ss, scipy.special as ssp


cpdef real_cepstrum(x, n = None):
    n = n or len(x)
    return np.fft.ifft(np.log(np.abs(np.fft.fft(x,n=n)))).real

cpdef inverse_real_cepstrum(x, n=None):
    n = n or len(x)
    return np.fft.ifft(np.exp(np.fft.fft(x,n=n))).real

cpdef minimum_phase_cepstrum(x, n = None):
    data_len = len(x)
    n = n or data_len * 8
    ceps = real_cepstrum(x,n=n)
    ceps[1:len(ceps)//2]  *= 2
    ceps[len(ceps)//2+1:] *= 0
    return inverse_real_cepstrum(ceps)[:data_len]

cpdef smooth_cepstrum(x, bw = 1/32,n = None):
    n = n or len(x)
    ceps = real_cepstrum(x, n=n)
    ceps[int(bw * n):-int(bw * n)] = 0
    return inverse_real_cepstrum(ceps,n=n)
