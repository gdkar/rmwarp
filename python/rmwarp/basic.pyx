import pyfftw as fftw, numpy as np, scipy as sp


def time_derivative_window(_win):
    n = len(_win)
    time = fftw.empty_aligned((n,),dtype=np.complex64)
    freq = fftw.empty_aligned((n,),dtype=np.complex64)
    plan = fftw.FFTW(
        input_array = time, output_array = freq
      , direction = 'FFTW_FORWARD'
      , flags = ( 'FFTW_ESTIMATE',)
        )
    plan.input_array[::] = _win[::]
    plan.execute()
    base_idx = n // 2 + 1
    base_mul = np.float32(- ( n - 1 ) * 0.5)
    norm_mul = np.float32(( 2 * np.pi ) * n**-2)
    for i in range(n):
        idx = (i + base_idx) % n
        mul =-(i + base_mul) * norm_mul
        time[idx] = np.complex(freq[idx].imag,freq[idx].real) * mul
    plan.execute()
    return freq.real

def time_weighted_window(_win):
    n = len(_win)
    c = np.float32(-0.5 * (n-1))
    return _win * np.linspace(c, -c, n, True, dtype=np.float32)
