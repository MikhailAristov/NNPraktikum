# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from model.autoencoder import Autoencoder
from model.layer import Layer

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class DenoisingAutoencoder(Autoencoder):
    """
    A denoising autoencoder.

    Parameters
    ----------
    train : list
    valid : list
    test : list
    layerWeights : list
        A list of of weights (one item = a complete weight matrix of a layer)
    randomLayers : list
        A list of positive integers, indicating, in order, how many neurons each hidden layer (on top of the predefined layers) should contain
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
    """

    def __init__(self, train, valid, test, layerWeights=[], randomLayers=[], learningRate=0.1, epochs=100):
        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize layers
        self.layers = []
        self.size = len(layerWeights) + len(randomLayers) + 1

        # Create hidden layers
        previousLayerSize = self.trainingSet.input.shape[1]
        maxInitLayerWeight = 4.0 / 7840.0
        # Append fixed layers, if any
        for predefinedLayerWeight in layerWeights:
            if len(predefinedLayerWeight[0]) == previousLayerSize + 1: # +1 for bias
                self.layers.append(Layer(nIn = previousLayerSize, nOut = len(predefinedLayerWeight), activation='sigmoid', weights=predefinedLayerWeight))
                previousLayerSize = len(predefinedLayerWeight)
            else:
                raise ValueError("Layer size mismatch!")
        # Append random layers, if any
        for layerSize in randomLayers:
            # Input layer receives 784 inputs averaging at 0.5; their weighted sum must still be within sigmoid's effective range
            self.layers.append(Layer(nIn = previousLayerSize, nOut = layerSize, activation='sigmoid', maxInitWeight=maxInitLayerWeight))
            previousLayerSize = layerSize
        # Create output layer
        self.layers.append(Layer(nIn = previousLayerSize, nOut = self.trainingSet.input.shape[1], activation='sigmoid', maxInitWeight=maxInitLayerWeight))

        # Cross-link each layer except the output (which obviously has no downstream) to the respective downstream layer
        for i in xrange(self.size - 1):
            self.layers[i].setDownstream(self.layers[i+1])

    def train(self, maskSize=0.1, errorThreshold=0.0, errorDeltaThreshold=0.01, miniBatchSize=100, weightDecay=0.0, momentum=0.0):
        """Train the denoiser autoencoder
        Parameters
        ----------
        maskSize : positive float
            portion of the input to be randomly masked (e.g. 0.1 for 10%)
        errorThreshold : positive float
            how small the error function can get before the learning is stopped early
        errorDeltaThreshold : float
            how small the error function decrease can get before the learning is stopped early
        miniBatchSize : positive int
        weightDecay : positive float
        momentum : positive float
        """
        from util.loss_functions import MeanSquaredError
        loss = MeanSquaredError()

        learned = False
        iteration = 0
        prevError = 1000000

        while not learned:
            iteration += 1
            currentBatchCount = 1
            # Update the weights from each input in the training set
            for input in self.trainingSet.input:
                # Calculate the outputs (stored within the layer objects themselves) and backpropagate the errors
                self.fire(self.mask(input, maskSize))
                self.learn(input)
                
                # Minibatch handling
                if iteration <= 1 or currentBatchCount >= miniBatchSize: # The first iteration is always stochastic gradient descent, so it converges faster
                    self.updateAllWeights(self.learningRate, weightDecay, momentum)
                    currentBatchCount = 0
                else:
                    currentBatchCount += 1
            # Apply the last weight update in case the last batch was too short
            self.updateAllWeights(self.learningRate, weightDecay, momentum)

            # Calculate the total error function in the validation set with current weights
            error = sum(map(loss.calculateError, self.validationSet.input, map(self.fire, self.validationSet.input)))
            errorDelta = prevError - error
            # Exit if error stops decreasing significantly or starts increasing again or max epochs reached
            if iteration >= self.epochs or (error < errorThreshold and errorDelta < 2.0 * errorDeltaThreshold) or errorDelta < errorDeltaThreshold:
                learned = True
            prevError = error # Update the error value for the next iteration

            logging.info("Iteration: %2i; Error: %9.4f", iteration, error)
    
    def mask(self, input, portion):
        # Calculate the exact number of input features to mask
        maskedCount = int(portion * len(input))
        # Randomly select this many features
        maskedFeatures = np.random.choice(range(len(input)),maskedCount)
        # Set the values of masked features to zero
        maskedInput = np.array(input)
        for i in maskedFeatures:
            maskedInput[i] = 0.0
        return maskedInput
    
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

    def learn(self, targetOutput):
        """Backpropagate the error through all layers.
        This method does not return anything and only updates the cumulativeWeightsUpdate attribute of each layer.
        updateAllWeights() has to be called to actually update the weights in the layers.

        Parameters
        ----------
        targetOutput : list of floats
        """
        # Go from the TOP of the layer stack and backpropagate the error
        downstreamDeltas = None
        for layer in reversed(self.layers):
            if downstreamDeltas is None: # i.e. this is the output (highest) layer
                downstreamDeltas = layer.backward(targetOutput=targetOutput, errorFunction = 'MeanSquaredError')
            else: # i.e. this is a hidden layer
                downstreamDeltas = layer.backward(downstreamDeltas=downstreamDeltas)

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
