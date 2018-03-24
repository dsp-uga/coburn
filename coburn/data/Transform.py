"""
Abstract base class for Transforms that can be applied to thunder.Series objects
"""

from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, series):
        """
        This method should be overridden in child classes.  It takes a thunder.Series object and should return an
        :param series:
        :return:
        """
        pass
