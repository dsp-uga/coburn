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

    def __init__(self, sigma, size):
        self.size = size;
        self.sigma = sigma

    def __call__(self, images):
        return images.gaussian_filter(sigma=self.sigma, order=self.size);


class Median(Transform):
    """
    Computes the median filter of smoothing the images along the time axis
    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
        return images.median_filter(size=self.size);


class Deviation(Transform):
    """
    Computes the standard deviation of images along the time axis
    """

    def __call__(self, images):
        return images.std();


class Subtract(Transform):
    """
    This will subtract given value from all the images

    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
        return images.subtract(val=self.size);


class UniformFilter(Transform):
    """
    Applies uniform filter to all the images

    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
<<<<<<< HEAD
        return images.var()
=======
        return images.uniform_filter(size=self.size);
>>>>>>> vibodh/master
