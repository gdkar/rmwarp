import numpy as np, scipy as sp, scipy.fftpack as sf,scipy.signal as ss
from matplotlib import pyplot as plt

def make_response(x, oversample=64):
    N = len(x)
    N_plot = N * oversample
    x_plot = np.zeros((N_plot,),dtype=np.float32)
    x_plot[:(N)//2] = sf.fftshift(x)[:(N)//2]
    x_plot[(N)//2 - N:] = sf.fftshift(x)[(N)//2 - N:]
    X_plot = sf.fft(x_plot)/(oversample*np.pi)
    X_mag = np.abs(X_plot)
    X_log  = np.log(X_plot)
    ax0 = plt.subplot(311)
    plt.grid()
    plt.plot((X_log.real) * 20/ np.log(10))
    ax1 = plt.subplot(312, sharex=ax0)
    plt.grid()
    plt.plot((X_log.imag))
    ax2 = plt.subplot(313, sharex=ax0)
    plt.grid()
    plt.plot(X_mag.real)
