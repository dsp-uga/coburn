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