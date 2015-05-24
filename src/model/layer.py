import time
import numpy as np

from util.activation_functions import Activation

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

class Layer(object):
    """
    A layer of perceptrons
    
    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    maxInitWeight: float: highest initial (random) weight in the layer
    
    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    cumulativeWeightsUpdate : ndarray (same dimensions as weights)
        update values for the weight matrix
    learningIterationCounter : positive int
        how many times the cumulativeWeightsUpdate has been updated since the last weights update
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    downstream: layer
        pointer at the downstream layer (for backpropagation)
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    transposedWeights : ndarray
        transposed weight matrix (optimization)
    lastInput : ndarray
        last input to the layer, stored for backpropagation
    lastOutput : ndarray
        last output of the layer, stored for backpropagation
    lastOutputDerivatives : ndarray
        derivatives of the last output, stored for backpropagation
    """

    def __init__(self, nIn, nOut, weights=None, activation='sigmoid', maxInitWeight=1.0):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.derivative = Activation.getDerivative(self.activationString)
        self.downstream = None

        self.nIn = nIn
        self.nOut = nOut        

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1), high=maxInitWeight)
        else:
            self.weights = weights
        # Prepare the update matrix
        self.cumulativeWeightsUpdate = np.ndarray(self.weights.shape)
        self.cumulativeWeightsUpdate.fill(0.0)
        self.learningIterationCounter = 0.0

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape
        # Some performance optimization
        self.transposedWeights = self.weights.transpose()
        # Runtime variables
        self.lastInput = None
        self.lastOutput = None
        self.lastOutputDerivatives = None

    def forward(self, input):
        """
        Compute forward step over the input using its weights
        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer
        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # Save the last input
        self.lastInput = np.matrix(input)
        # Calculate the net output of the neuron
        netOutput = np.dot(input, self.transposedWeights)
        # Apply the activation function
        self.lastOutput = self.activation(netOutput)
        # Apply the derivative activation function for gradient descent
        self.lastOutputDerivatives = self.activation(netOutput)
        return self.lastOutput

    def backward(self, learningRate, targetOutput=None, downstreamSigmas=None):
        """
        TODO: Documentation
        """
        # Calculate the sigmas for the weight update, either based on expected output, or on the downstream sigmas
        if self.downstream is None and targetOutput is not None: # i.e. this is the output layer
            sigmas = targetOutput - self.lastOutput # TODO: Generalize, not just cross entropy derivative
        elif self.downstream is not None and downstreamSigmas is not None: # i.e. this is one of the hidden layers
            # Downstream factors := downstream sigmas x downstream weights (in this network design, the downstream is the complete layer above)
            # Note that the last weight of each downstream neuron is the bias weight, which doesn't influence this layer's weights, so it is filtered out
            downstreamFactors = np.dot(downstreamSigmas, self.downstream.weights[:,:-1])
            # Multiply this layer's derivative functions with the downstream's correction factors to get this layer's sigmas
            sigmas = np.multiply(downstreamFactors, self.lastOutputDerivatives)
        else:
            raise ValueError("Target output must be specified for the output layer and downstream sigmas, for all hidden layers!")
        
        # Update the weights update matrix (actual weights are updated in by the updateWeights() method (to enable batch learning)
        self.cumulativeWeightsUpdate += learningRate * np.dot(np.matrix(sigmas).transpose(), self.lastInput)
        self.learningIterationCounter += 1.0
        # Return the current layer's sigmas to be propagated upstream
        return sigmas
    
    def updateWeights(self):
        """
        Update weights with the current cumulative update values.
        If called immediately after backward(), amounts to stochastic gradient descent, otherwise enables (mini-)batch GD
        """
        if self.learningIterationCounter > 0.0:
            self.weights += 1.0 / self.learningIterationCounter * self.cumulativeWeightsUpdate
            self.cumulativeWeightsUpdate.fill(0.0) # Reset the update values
            self.learningIterationCounter = 0.0  # Reset the updates counter
            self.transposedWeights = self.weights.transpose() # Performance
    
    def setDownstream(self, layer):
        """
        Set the pointer to the downstream layer.
        Parameters
        ----------
        deltas : layer
            pointer to the downstream layer
        """
        if type(layer) is type(self):
            self.downstream = layer
        else:
            raise TypeError("Downstream must be a layer!")
        