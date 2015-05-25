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
        # Storage variables for backpropagation
        self.lastInput = None
        self.lastOutput = None
        self.lastOutputDerivatives = None
        self.lastWeightUpdate = np.ndarray(self.weights.shape)
        self.lastWeightUpdate.fill(0.0)

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
        self.lastOutputDerivatives = self.derivative(netOutput)
        return self.lastOutput

    def backward(self, targetOutput=None, downstreamDeltas=None):
        """
        Backpropagates the error from downstream through the layer and stores the cumulative weight updates.
        Parameters
        ----------
        targetOutput : ndarray
            a numpy array (1,nOut) containing the target out of the layer (output layer ONLY!)
        downstreamDeltas : ndarray
            a numpy array (1,nOut) containing the deltas out of the downstream layer (hidden layers ONLY!)
        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the deltas of the current layer, to be passed upstream
        """
        # Calculate the deltas for the weight update, either based on expected output, or on the downstream deltas
        if self.downstream is None and targetOutput is not None: # i.e. this is the output layer
            deltas = targetOutput - self.lastOutput # TODO: Generalize, not just cross entropy derivative
        elif self.downstream is not None and downstreamDeltas is not None: # i.e. this is one of the hidden layers
            # Downstream factors := downstream deltas x downstream weights (in this network design, the downstream is the complete layer above)
            # Note that the last weight of each downstream neuron is the bias weight, which doesn't influence this layer's weights, so it is filtered out
            downstreamFactors = np.dot(downstreamDeltas, self.downstream.weights[:,:-1])
            # Multiply this layer's derivative functions with the downstream's correction factors to get this layer's deltas
            deltas = np.multiply(downstreamFactors, self.lastOutputDerivatives)
        else:
            raise ValueError("Target output must be specified for the output layer and downstream deltas, for all hidden layers!")

        # Update the weights update matrix (actual weights are updated in by the updateWeights() method (to enable batch learning)
        self.cumulativeWeightsUpdate += np.dot(np.matrix(deltas).transpose(), self.lastInput) # Tensor product!
        self.learningIterationCounter += 1.0
        # Return the current layer's deltas to be propagated upstream
        return deltas

    def updateWeights(self, learningRate, weightDecay, momentum):
        """
        Update weights with the current cumulative update values.
        If called immediately after backward(), amounts to stochastic gradient descent, otherwise enables (mini-)batch GD
        Parameters
        ----------
        learningRate : positive float
        weightDecay : positive float
        momentum : positive float
        """
        if self.learningIterationCounter > 0.0:
            # Calculate the weight update
            weightUpdate = (learningRate / self.learningIterationCounter) * self.cumulativeWeightsUpdate - learningRate * weightDecay * self.weights + momentum * self.lastWeightUpdate
            # Apply the weight update
            self.weights += weightUpdate
            # Initialize/update runtime variables
            self.cumulativeWeightsUpdate.fill(0.0) # Reset the update values
            self.learningIterationCounter = 0.0  # Reset the updates counter
            self.transposedWeights = self.weights.transpose() # Performance
            self.lastWeightUpdate = weightUpdate

    def setDownstream(self, layer):
        """
        Set the pointer to the downstream layer.
        Parameters
        ----------
        layer : layer
            pointer to the downstream layer
        """
        if type(layer) is type(self):
            self.downstream = layer
        else:
            raise TypeError("Downstream must be a layer!")
