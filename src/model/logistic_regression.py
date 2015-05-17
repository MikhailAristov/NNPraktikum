# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from util.activation_functions import Activation
from model.classifier import Classifier

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    errorThreshold : positive float
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

        # Stop learning if the MSE drops below this value
        self.errorThreshold = 0.01

    def train(self):
        """Train the Logistic Regression"""
        # Selt MSE as the error function
        from util.loss_functions import MeanSquaredError
        loss = MeanSquaredError()

        learned = False
        iteration = 0

        while not learned:
            iteration += 1
            # Attempt to classify each symbol and update the weights accordingly
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):
                output = self.fire(input)
                delta = (label - output) * output * (1 - output)
                self.updateWeights(input, delta)

            # Calculate the total error function with current weights
            error = loss.calculateError(self.validationSet.label, map(self.fire, self.validationSet.input))

            # Exit if error threshold or max epochs reached
            if error <= self.errorThreshold or iteration >= self.epochs:
                learned = True
            logging.info("Iteration: %i; Error: %.4f", iteration, error)

    def updateWeights(self, input, delta):
        """Update input weights.

        Parameters
        ----------
        input : list of floats
        delta : float
        """
        for index in xrange(len(input)):
            self.weight[index] += self.learningRate * delta * input[index]

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return (self.fire(testInstance) >= 0.5)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
