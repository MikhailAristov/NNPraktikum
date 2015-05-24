# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.classifier import Classifier
from model.layer import Layer

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class MultilayerPerceptron(Classifier):
    """
    A multilayer digit recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    hiddenLayers : list
        A list of positive integers, indicating, in order, how many neurons each hidden layer should contain
    outputDim : positive int
        How many neurons the output layer should contain
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    learningRate : float
    epochs : positive int
    layers : list of layer objects
    size : positive int
        total number of layers
    errorDeltaThreshold : float
        how small the error function descrease can get before the learning is stopped early
    """

    def __init__(self, train, valid, test, hiddenLayers, outputDim, learningRate=0.1, epochs=50):
        self.learningRate = learningRate
        self.epochs = epochs
        self.errorDeltaThreshold = 0.3 # 0.0001

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        # Initialize layers
        self.layers = []
        self.size = len(hiddenLayers) + 1
        
        # Create hidden layers
        previousLayerSize = self.trainingSet.input.shape[1]
        for layerSize in hiddenLayers:
            self.layers.append(Layer(nIn = previousLayerSize, nOut = layerSize, activation='sigmoid', maxInitWeight=0.02))
            previousLayerSize = layerSize
        # Create output layer
        self.layers.append(Layer(nIn = previousLayerSize, nOut = outputDim, activation='softmax'))
        
        # Cross-link each layer except the output (which obviously has no downstream) to the respective downstream layer
        for i in xrange(self.size - 1):
            self.layers[i].setDownstream(self.layers[i+1])

    def train(self):
        """Train the Multilayer perceptron"""
        from util.loss_functions import CrossEntropyError
        loss = CrossEntropyError()

        learned = False
        iteration = 0
        prevError = 1000000
        currentLearningRate = self.learningRate
            
        # Preprocessing for performance reasons
        validationOutputs = map(self.convertLabelToFlags, self.validationSet.label)
        
        while not learned:
            iteration += 1
            currentLearningRate -= self.learningRate * (iteration - 1) / self.epochs
            batchSize = 1 if iteration == 1 else len(self.trainingSet.label) / 10
            
            currentBatchCount = 0
            # Update the weights from each input in the training set
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):
                self.fire(input)
                self.learn(self.convertLabelToFlags(label), currentLearningRate)
                
                currentBatchCount += 1
                if currentBatchCount >= batchSize:
                    self.updateAllWeights()
                    currentBatchCount = 0
            self.updateAllWeights()

            # Calculate the total error function in the validation set with current weights
            error = sum(map(loss.calculateError, validationOutputs, map(self.fire, self.validationSet.input)))
            
            # Exit if error stops decreasing significantly or starts increasing again or max epochs reached
            if iteration >= self.epochs or (prevError - error) < self.errorDeltaThreshold:
                learned = True
            prevError = error # Update the error value for the next iteration
    
            logging.info("Iteration: %2i; Error: %9.4f", iteration, error)
            
    def convertLabelToFlags(self, label):
        """Convert a label to an MLP output array

        Parameters
        ----------
        label : string

        Returns
        -------
        list of floats
            all zeroes, except at position = label
        """
        result = [0.0] * 10
        result[int(label)] = 1.0
        return result
        
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        string
            the most probable label, given MLP output
        """
        return np.argmax(self.fire(testInstance))

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
        return list(map(self.classify, test))

    def fire(self, input):
        """Calculate the raw MLP result of an input

        Parameters
        ----------
        input : list of floats

        Returns
        -------
        list of floats
        """
        prevLayerOutput = input
        for layer in self.layers:
            # Pass calculated layer output with the input plus bias
            prevLayerOutput = layer.forward(np.append(prevLayerOutput, 1))
        return prevLayerOutput
    
    def learn(self, targetOutput, learningRate):
        """Backpropagate the error through all layers.
        This method does not return anything and only updates the cumulativeWeightsUpdate attribute of each layer.
        updateAllWeights() has to be called to actually update the weights in the layers.

        Parameters
        ----------
        targetOutput : list of floats
        """
        # Go from the TOP of the layer stack and backpropagate the error
        downstreamSigmas = None
        for layer in reversed(self.layers):
            if downstreamSigmas is None: # i.e. this is the output (highest) layer
                downstreamSigmas = layer.backward(learningRate, targetOutput=targetOutput)
            else: # i.e. this is a hidden layer
                downstreamSigmas = layer.backward(learningRate, downstreamSigmas=downstreamSigmas)
                
    def updateAllWeights(self):
        """
        Updates all weights in all layers of the MLP, using the data stored in the cumulativeWeightsUpdate attribute of each layer.
        """
        for layer in self.layers:
            layer.updateWeights()