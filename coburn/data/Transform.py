"""
Abstract base class for Transforms that can be applied to thunder.Series objects
"""

from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, images):
        """
        This method should be overridden in child classes.  It should take a thunder.Images object and, for maximum
        composability, return a thunder.Images object
        :param data: thunder.Images
        :return: object resulting from applying the transform to the images
        """
        pass
