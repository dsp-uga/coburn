import numpy as np
from .Transform import Transform

class Frequency(Transform):
    def __call__(self, images):
        return images.toarray()
