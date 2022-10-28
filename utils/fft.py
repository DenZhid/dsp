import numpy as np
from scipy.fft import fft


def custom_fft(duration, sig, fd):
    fftl = __nextpow2(duration)
    y = np.abs(fft(sig, fftl))
    y = 2 * y / duration
    y[0] = y[0] / 2
    F = np.arange(0, fd / 2 - 1 / fftl, fd / fftl)
    return y, F


def __nextpow2(a):
    b = 1
    while b < a:
        b *= 2
    return b
