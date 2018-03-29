"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from .Transform import Transform


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

    def __call__(self, series):
        images=series.toimages()
        return images.gaussian_filter(sigma=2,order=0).values[0];


class Median(Transform):
    """
    Computes the median filter of smoothing the images along the time axis
    """

    def __call__(self, series):
        images = series.toimages()
        return images.median_filter(size=2).values[0];


class Deviation(Transform):

    """
    Computes the standard deviation of images along the time axis
    """

    def __call__(self, series):
        images = series.toimages()
        return images.std().values[0];


class Subtract(Transform):
    """
    This will subtract given value from all the images

    """
    def __call__(self, series):
        images = series.toimages()

        return images.subtract(2).values[0];

class UniformFilter(Transform):
    """
    Applies uniform filter to all the images

    """

    def __call__(self, series):
        images = series.toimages()
        return images.uniform_filter(2).values[0];