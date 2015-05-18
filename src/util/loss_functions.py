# -*- coding: utf-8 -*-


"""
Loss functions.
"""

from numpy import log2
from abc import ABCMeta, abstractmethod

class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        """
        Calculate error
        ----------
        target : list
        output : list
            target and output must be the same length!
        Returns
        -------
        float :
            the mean squared error value
        """
        # Here you have to calculate the MeanSquaredError
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        result = map(lambda t, o: (t - o) * (t - o), target, output)
        return (0.5 * sum(result) / len(result))

class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        # Here you have to calculate the SumSquaredError
        # MSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        result = map(lambda t, o: 0.5 * (t - o) * (t - o), target, output)
        return sum(result)


class CrossEntropyError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        result = map(lambda t, o: t * log2(o) + (1 - t) * log2(1 - o), target, output)
        return -1.0 * sum(result)
