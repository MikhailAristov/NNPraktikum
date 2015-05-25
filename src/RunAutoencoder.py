#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.denoiser import DenoisingAutoencoder

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)
    myDenoiser = DenoisingAutoencoder(data.trainingSet,
                                      data.validationSet,
                                      data.testSet,
                                      randomLayers=[80]
                                      )
    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nDenoising autoencoder has been training..")
    myDenoiser.train()
    print("Done..")
    
    myDenoiser.layers[0].saveToFile("../data/features_layer_80.gz")

if __name__ == '__main__':
    main()
