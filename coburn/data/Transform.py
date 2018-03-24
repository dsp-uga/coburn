"""
Abstract base class for Transforms that can be applied to thunder.Series objects
"""

from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, data):
        """
        This method should be overridden in child classes.  It should take a thunder.Series object and, for maximum
        composability, return a thunder.Series object
        :param series: thunder.Series
        :return: object resulting from applying the transform to the series
        """
        pass
