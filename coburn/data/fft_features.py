import numpy as np
from . import Transform

class Frequency(Transform):
    def __call__(self, images):
        return images.toarray()
