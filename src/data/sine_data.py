# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.custom_data_set import CustomDataSet

class SineData(object):
    """
    Small subset (5000 instances) of sin() data to recognize

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
            for i in xrange(200):
                x = i * np.pi / 100 - np.pi
                trainingData.append([np.sin(x), x])
        shuffle(trainingData)
        
        # Assign data to sets
        self.trainingSet = CustomDataSet(np.array(trainingData[:numTrain]))
        self.validationSet = CustomDataSet(np.array(trainingData[numTrain:-numTest]))
        self.testSet = CustomDataSet(np.array(trainingData[-numTest:]))
        
