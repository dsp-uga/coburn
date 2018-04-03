from .Transform import Transform
import numpy as np
from numpy.fft import fft

class Frequency(Transform):
    def __call__(self, images):
        return images.map_as_series(func=fft).toarray()
class Histogram(Transform):
    def __call__(self,images):
        return images.map_as_series(func=self.hist)
    def hist(self, arr):
        bins, idx = np.histogram(arr)
        return bins
