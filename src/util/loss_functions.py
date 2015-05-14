# -*- coding: utf-8 -*-


"""
Loss functions.
"""


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
        # Here you have to calculate the MeanSquaredError
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        sum = 0
        for targetValue, outputValue in zip(target, output):
            sum += (targetValue - outputValue) * (targetValue - outputValue)
        return (sum / len(target))

class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        # Here you have to calculate the SumSquaredError
        # MSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        pass


class CrossEntropyError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        pass
