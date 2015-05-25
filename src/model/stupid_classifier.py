# -*- coding: utf-8 -*-

from random import choice
from model.classifier import Classifier

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

class StupidClassifier(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    classifier.

    It also serves as a baseline to compare with other
    classifying method later on.
    """

    def __init__(self, train, valid, test):
        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

    def train(self):
        # Do nothing
        pass

    def classify(self, testInstance):
        # Randomly pick a label from the test set
        return choice(self.testSet.label)

    def evaluate(self):
        return list(map(self.classify, self.testSet.input))
