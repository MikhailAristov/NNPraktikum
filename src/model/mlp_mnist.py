# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.classifier import Classifier
from model.layer import Layer
from copy import deepcopy

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
    layerWeights : list
        A list of of weights (one item = a complete weight matrix of a layer)
    randomLayers : list
        A list of positive integers, indicating, in order, how many neurons each hidden layer (on top of the predefined layers) should contain
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
    bestConfig : copy of layers
    """

    def __init__(self, train, valid, test, outputDim, layerWeights=[], randomLayers=[], outputLayer=None, learningRate=0.1, epochs=150):
        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize layers
        self.layers = []
        self.size = len(layerWeights) + len(randomLayers) + 1
        self.bestConfig = None

        # Create hidden layers
        previousLayerSize = self.trainingSet.input.shape[1]
        # Append fixed layers, if any
        for predefinedLayerWeight in layerWeights:
            if len(predefinedLayerWeight[0]) == previousLayerSize + 1: # +1 for bias
                self.layers.append(Layer(nIn = previousLayerSize, nOut = len(predefinedLayerWeight), activation='sigmoid', weights=predefinedLayerWeight, frozen=False))
                previousLayerSize = len(predefinedLayerWeight)
            else:
                raise ValueError("Layer size mismatch!")
        # Append random layers, if any
        for layerSize in randomLayers:
            self.layers.append(Layer(nIn = previousLayerSize, nOut = layerSize, activation='sigmoid', maxInitWeight=0.0005))
            previousLayerSize = layerSize
        # Create output layer
        if outputLayer is not None:
            if len(outputLayer[0]) == previousLayerSize + 1 and len(outputLayer) == outputDim:
                self.layers.append(Layer(nIn = previousLayerSize, nOut = outputDim, weights=outputLayer, activation='softmax'))
            else:
                raise ValueError("Layer size mismatch!")
        else:
            self.layers.append(Layer(nIn = previousLayerSize, nOut = outputDim, activation='softmax'))
        
        # Cross-link each layer except the output (which obviously has no downstream) to the respective downstream layer
        for i in xrange(self.size - 1):
            self.layers[i].setDownstream(self.layers[i+1])

    def train(self, errorThreshold=550.0, errorDeltaThreshold=0.01, miniBatchSize=100, weightDecay=0.0001, momentum=0.7, dropout=0.05):
        """Train the Multilayer perceptron
        Parameters
        ----------
        errorThreshold : positive float
            how small the error function can get before the learning is stopped early
        errorDeltaThreshold : float
            how small the error function decrease can get before the learning is stopped early
        miniBatchSize : positive int
        weightDecay : positive float
        momentum : positive float
        dropout : positive float
            portion of the input to randomly set to zeroes in each layer
        """
        from util.loss_functions import CrossEntropyError
        loss = CrossEntropyError()

        learned = False
        strikes = 0
        iteration = 0
        prevError = 1000000
        minError = prevError
        currentMomentum = momentum

        # Preprocessing for performance reasons
        validationOutputs = map(self.convertLabelToFlags, self.validationSet.label)
        labeledInputs = zip(self.trainingSet.input, self.trainingSet.label)

        while not learned:
            iteration += 1
            currentBatchCount = 1
            currentMomentum = currentMomentum * 1.1 # Increase the momentum with every iteration
            # Update the weights from each input in the training set
            for input, label in labeledInputs:
                # Calculate the outputs (stored within the layer objects themselves) and backpropagate the errors
                self.fire(input, dropout)
                self.learn(self.convertLabelToFlags(label))
                # Minibatch handling
                if iteration <= 1 or currentBatchCount >= miniBatchSize: # The first iteration is always stochastic gradient descent, so it converges faster
                    self.updateAllWeights(self.learningRate, weightDecay, momentum if iteration > 1 else 0.0)
                    currentBatchCount = 0
                else:
                    currentBatchCount += 1
            # Apply the last weight update in case the last batch was too short
            self.updateAllWeights(self.learningRate, weightDecay, momentum if iteration > 1 else 0.0)

            # Calculate the total error function in the validation set with current weights
            error = sum(map(loss.calculateError, validationOutputs, map(self.fire, self.validationSet.input)))
            errorDelta = prevError - error
            # Save current configuration if it is the best so far
            if error < minError:
                self.bestConfig = deepcopy(self.layers)
                minError = error
            # Exit if error stops decreasing significantly or starts increasing again or max epochs reached
            if iteration >= self.epochs:
                learned = True
            elif (error < errorThreshold and errorDelta < 2.0 * errorDeltaThreshold) or errorDelta < errorDeltaThreshold:
                if strikes < 3 and dropout > 0.0: # Use the three-strikes system only if random noise is applied, otherwise stop learning immediately
                    strikes += 1
                else:
                    learned = True
            else:
                strikes = 0 if strikes == 0 else strikes - 1
            prevError = error # Update the error value for the next iteration

            logging.info("Iteration: %2i; Error: %9.4f", iteration, error)
        
        # Restore best config from backup
        self.layers = self.bestConfig

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

    def fire(self, input, dropout=0.0):
        """Calculate the raw MLP result of an input

        Parameters
        ----------
        input : list of floats
        dropout : positive float
            portion of the input to randomly set to zeroes in each layer

        Returns
        -------
        list of floats
        """
        prevLayerOutput = input
        for layer in self.layers:
            # Pass calculated layer output with the input plus bias
            prevLayerOutput = layer.forward(np.append(prevLayerOutput, 1), dropout)
        return prevLayerOutput

    def learn(self, targetOutput):
        """Backpropagate the error through all layers.
        This method does not return anything and only updates the cumulativeWeightsUpdate attribute of each layer.
        updateAllWeights() has to be called to actually update the weights in the layers.

        Parameters
        ----------
        targetOutput : list of floats
        """
        # Only backpropagate if the error is big enough
        
        # Go from the TOP of the layer stack and backpropagate the error
        downstreamDeltas = None
        for layer in reversed(self.layers):
            if downstreamDeltas is None: # i.e. this is the output (highest) layer
                downstreamDeltas = layer.backward(targetOutput=targetOutput, errorFunction='CrossEntropyError')
            elif not layer.frozen: # i.e. this is a hidden layer
                downstreamDeltas = layer.backward(downstreamDeltas=downstreamDeltas)
            else: # Stop calculation upon encountering the first frozen layer
                break

    def updateAllWeights(self, learningRate, weightDecay, momentum):
        """
        Updates all weights in all layers of the MLP, using the data stored in the cumulativeWeightsUpdate attribute of each layer.
        
        Parameters
        ----------
        learningRate : positive float
        weightDecay : positive float
        momentum : positive float
        """
        for layer in self.layers:
            layer.updateWeights(learningRate, weightDecay, momentum)
