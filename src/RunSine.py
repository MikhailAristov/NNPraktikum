#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.sine_data import SineData
from model.mlp_sine import MultiLayerPerceptronSine
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

def main():
    data = SineData()
    myMLPerceptron = MultiLayerPerceptronSine(data.trainingSet,
                                              data.validationSet,
                                              data.testSet,
                                              hiddenLayers=[8],
                                              outputDim=1,
                                              epochs=150)
    
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nMultilayer perceptron has been training..")
    myMLPerceptron.train()
    print("Done..")
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    MLPPred = myMLPerceptron.evaluate()
    
    # Report the result
    print("=========================")
    
    # ...as a plot image
    x = np.arange(-np.pi, np.pi, np.pi/100);
    y = np.sin(x)
    plt.plot(x, y) # the reference sine curve
    plt.plot(data.testSet.input[:,0], MLPPred, 'r+') # the MLP approximation
    plt.savefig("../data/result_sin_mlp_8hl_150e", ext="png", close=True, verbose=True)

if __name__ == '__main__':
    main()
