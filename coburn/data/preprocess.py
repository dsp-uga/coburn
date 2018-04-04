"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from .Transform import Transform
from thunder.images.images import Images
import numpy as np
import cv2


class Mean(Transform):
    """
    Computes the mean of a series of images along the time axis
    """

    def __call__(self, images):
        return images.mean()


class Variance(Transform):
    """
    Computes the variance of a series of images along the time axis
    """

    def __call__(self, images):
        return images.var()


class Gaussian(Transform):
    """
    Computes the gaussian smoothing of images along the time axis
    """

    def __init__(self, sigma, size):
        self.size = size
        self.sigma = sigma

    def __call__(self, images):
        return images.gaussian_filter(sigma=self.sigma, order=self.size)


class Deviation(Transform):
    """
    Computes the standard deviation of images along the time axis
    """

    def __call__(self, images):
        return images.std()


class Subtract(Transform):
    """
    This will subtract given value from all the images

    """

    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        return images.subtract(val=self.size)


class UniformFilter(Transform):
    """
    Applies uniform filter to all the images
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        return images.uniform_filter(size=self.size)


class MedianFilter(Transform):
    """
    Applies a median filter with the specified kernel size to every image in the series
    """
    def __init__(self, size):
        self.size = size or 2

    def __call__(self, images):
        return images.median_filter(size=self.size)


class OpticalFlow(Transform):
    """
    Computes optical flow between each sequential pair of images.
    return shape has an extra dimension of size 2, and the first dimension is
    of size one less (e.g. 100 becomes 99, i.e. shape[0]-=1).
    """
    def __call__(self, images):
        images = np.array(images)
        flows = np.empty((0, images.shape[1], images.shape[2], 2))
        i = 0
        while(i < images.shape[0]-1):
            flow = cv2.calcOpticalFlowFarneback(
                images[i], images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows = np.append(flows, [flow], axis=0)
            i += 1
        return Images(flows)

