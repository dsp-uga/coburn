from numpy as np
from . import Transform

class Frequency(Transform):
    def __call__(self, images):
        return images.map_as_series(func=self.fourier)
    def fourier(self, arr):
        n = 128
        sp = np.fft.fft(arr, n=n)
        # freq = np.fft.fftfreq(n=n, d=.02)
        return sp
