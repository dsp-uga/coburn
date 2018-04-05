"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from skimage.transform import resize as sk_resize
import thunder as td
import numpy as np
from .Transform import Transform


class UniformResize(Transform):
    """
    Resizes the series of images to be of uniform size
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, images):
        images = images.toarray()
        resized = np.zeros((len(images), self.height, self.width))
        for idx in range(0, len(images)):
            img = images[idx]
            resized_img = sk_resize(img, (self.height, self.width))
            resized[idx] = resized_img

        return td.images.fromarray(resized)


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


class ToArray(Transform):
    """
    Converts a thunder.Images object to a numpy ndarray with dimensions [H x W x T]
    where H is the height, W is the width, and T is the time or number of channels
    """
    def __call__(self, images):
        # handle the special case when the array is 2D:
        images = images.toarray()
        if len(images.shape) == 2:
            images = images[:, :, np.newaxis]
            return images

        return images.swapaxes(0, 2).swapaxes(0, 1)  # move the non-spatial axis to the correct position


class MaskToSegMap(Transform):
    """
    Converts an m x n PNG mask to a segmentation map.  Mask should be a numpy array with shape (m, n)
    A segmentation map will be m x n x 3.
    segmap[row, col, i] will be 1 if the mask has class i at location (row, col)
    """
    def __call__(self, mask):
        shape = mask.shape
        segmap = np.empty((shape[0], shape[1], 3))

        segmap[mask == 2] = [0, 0, 1]
        segmap[mask == 1] = [0, 1, 0]
        segmap[mask == 0] = [1, 0, 0]
        return segmap


class ResizeMask(Transform):
    """
    Resizes a PNG mask to be the specified size
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, mask):
        return sk_resize(mask, (self.height, self.width))
