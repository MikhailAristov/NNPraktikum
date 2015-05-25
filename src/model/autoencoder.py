# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class Autoencoder:
    """
    Abstract class of a classifier
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, trainingSet, validationSet):
        # train procedures of the classifier
        pass
