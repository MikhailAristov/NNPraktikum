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
    W: weight matrix with the shape of (nOut, nIn + 1), 1 for the bias
    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None, activation='sigmoid'):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.derivative = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))
        else:
            self.weights = weights

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape
        # Some performance optimization
        self.transposedWeights = self.weights.transpose()

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
        # Change the following line to calculate the net output
        # Here is just an example to ensure you have correct shape of output
        #netOutput = np.full(shape=(1, self.nOut), 1)
        netOutput = np.dot(input, self.transposedWeights)
        return map(self.activation, netOutput)

    def computeDerivative(self, input):
        """
        Compute the derivative
        Parameters
        ----------
        input : ndarray
            a numpy array containing the input of the layer
        Returns
        -------
        ndarray :
            a numpy array containing the derivatives on the input
        """
        # Here you have to compute the derivative values
        netOutput = np.dot(input, self.transposedWeights)        
        return map(self.derivative, netOutput)
    
    def updateWeights(self, deltas):
        """
        Update weights from a matrix of deltas.
        Parameters
        ----------
        deltas : ndarray
            a 2D numpy array of the same dimensions as the self.weights
        """
        self.weights += deltas
        self.transposedWeights = self.weights.transpose()
            