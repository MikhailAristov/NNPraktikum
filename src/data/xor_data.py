# -*- coding: utf-8 -*-

from numpy.random import shuffle
from data.custom_data_set import CustomDataSet
from numpy import array

class XORData(object):
    """
    Small set (5000 instances) of shuffled XOR data to recognize

    Parameters
    ----------
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, numTrain=3000, numValid=1000, numTest=1000):
        
        # Generate data
        trainingData = []
        while len(trainingData) < (numTrain + numValid + numTest):
            for a, b in zip([-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]):
                if (a + b) == 0:
                    label = 1.0
                else:
                    label = -1.0
                trainingData.append([label, a, b])
        shuffle(trainingData)
        
        # Assign data to sets
        self.trainingSet = CustomDataSet(array(trainingData[:numTrain]))
        self.validationSet = CustomDataSet(array(trainingData[numTrain:-numTest]))
        self.testSet = CustomDataSet(array(trainingData[-numTest:]))
        
