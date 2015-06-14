#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.denoiser import DenoisingAutoencoder

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)
    # The encoding layer should have as many neurons, as there are, on average, non-zero inputs in each training data set
    featuresLayerSize = int(data.nonZeroDataPoints / 5000) # = 151
    myDenoiser = DenoisingAutoencoder(data.trainingSet,
                                      data.validationSet,
                                      data.testSet,
                                      randomLayers=[featuresLayerSize]
                                      )
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nDenoising autoencoder has been training..")
    myDenoiser.train()
    print("Done..")
    
    myDenoiser.layers[0].saveToFile("../data/features_layer_" + str(featuresLayerSize) + ".gz")

if __name__ == '__main__':
    main()
