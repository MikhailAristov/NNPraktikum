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
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, hiddenLayers, outputDim, learningRate=0.01, epochs=50):
        self.learningRate = learningRate
        self.epochs = epochs
        self.errorDeltaThreshold = 1

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        # Initialize layers
        self.layers = []
        # Create hidden layers
        previousLayerSize = self.trainingSet.input.shape[1]
        for layerSize in hiddenLayers:
            self.layers.append(Layer(nIn = previousLayerSize, nOut = layerSize, activation='sigmoid', maxInitWeight=0.02))
            previousLayerSize = layerSize
        # Create output layer
        self.layers.append(Layer(nIn = previousLayerSize, nOut = outputDim, activation='softmax'))

    def train(self):
        """Train the Multilayer perceptron"""
        from util.loss_functions import CrossEntropyError
        loss = CrossEntropyError()

        learned = False
        iteration = 0
        prevError = 1000000
        dynamicLearningRate = False
        currentLearningRate = self.learningRate
        validationOutputs = map(self.convertLabelToFlags, self.validationSet.label)
        
        while not learned:
            iteration += 1
            
            # Update learning rate, if necessary
            if dynamicLearningRate:
                currentLearningRate *= 0.4
            
            # Update the weights from each input in the training set
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):
                self.trainSingleInput(input, self.convertLabelToFlags(label), currentLearningRate)

            # Calculate the total error function in the validation set with current weights
            error = sum(map(loss.calculateError, validationOutputs, map(self.fire, self.validationSet.input)))
            
            # Exit if error stops decreasing significantly or starts increasing again or max epochs reached
            if iteration >= self.epochs:
                learned = True
            elif (prevError - error) < self.errorDeltaThreshold:
                if not dynamicLearningRate:
                    logging.info("Dynamic learning rate activated")
                    dynamicLearningRate = True
                else:
                    learned = True
            prevError = error # Update the error value for the next iteration
    
            logging.info("Iteration: %i; Error: %.4f", iteration, error)
            
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
            # Pass calculate layer output with the input plus bias
            prevLayerOutput = layer.forward(np.append(prevLayerOutput, 1))
        return prevLayerOutput
    
    def trainSingleInput(self, input, target, learningRate):
        """Trains the MLP with a specific input and output.

        Parameters
        ----------
        input : list of floats
        target : list of floats
        learningRate : float
        """
        # First, calculate output of every layer; this is functionally identical to the fire() method, except it saves each layer's output separately
        layerOutputs = []
        layerOutputDerivatives = []
        for layer in self.layers:
            # For the first layer, take the input parameter as input, for all higher levels, take the respective previous layers output
            currentInput = input if len(layerOutputs) == 0 else layerOutputs[-1]
            # Calculate the current layer's output and derivatives and add them to the respective arrays (append 1 to input for the bias)
            layerOutputs.append(layer.forward(np.append(currentInput, 1)))
            #print layer.forward(np.append(currentInput, 1))
            layerOutputDerivatives.append(layer.computeDerivative(np.append(currentInput, 1)))
        # Now we have all the layer outputs in format layerOutputs[layerNumberFromTheBottom][neuronNumber], and likewise the derivatives
        
        # Then, go from the TOP of the output stack and backpropagate the error
        downstreamSigmas = None
        for l in xrange(len(layerOutputs)-1, -1, -1):
            # Current layer's input is the previous layer's output plus bias, unless it's the first hidden layer, in which case it takes the input
            currentInput = np.matrix(np.append(input if l == 0 else layerOutputs[l-1], 1))
            # Update the weights
            if downstreamSigmas is None: # i.e. this is the output layer
                # Calculate layer sigmas (a list with as many elements as there are neurons in the output layer)
                sigmas = target - layerOutputs[l]
            else: # i.e. this is one of the hidden layers
                # Downstream factors := downstream sigmas x downstream weights (in this network design, the downstream is the complete layer above)
                # Note that the last weight of each downstream neuron is the bias weight, which doesn't influence this layer's weights, so it is filtered out
                downstreamFactors = np.dot(downstreamSigmas, self.layers[l+1].weights[:,:-1])
                # Multiply this layer's derivative functions with the downstream's correction factors to get this layer's sigmas
                sigmas = np.multiply(downstreamFactors, layerOutputDerivatives[l])
            # Calculate the layer weights delta (a matrix: number of neurons x number of their inputs)
            # TODO: Fix the double transposition
            weightDeltas = learningRate * np.dot(currentInput.transpose(), np.matrix(sigmas)).transpose()
            # Apply weight deltas
            self.layers[l].updateWeights(weightDeltas)
            # Save the sigmas for the next step
            downstreamSigmas = sigmas
