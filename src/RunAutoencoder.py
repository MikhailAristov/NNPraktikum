#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.denoiser import DenoisingAutoencoder

import numpy as np

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)
    myDenoiser = DenoisingAutoencoder(data.trainingSet,
                                      data.validationSet,
                                      data.testSet,
                                      randomLayers=[40]
                                      )
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nDenoising outencoder has been training..")
    myDenoiser.train()
    print("Done..")
    
    # Output
    out = myDenoiser.fire(data.testSet.input[0])
    fx = np.array(zip(data.testSet.input[0], out))
    np.savetxt("../data/result.csv", fx, delimiter=",")

if __name__ == '__main__':
    main()
