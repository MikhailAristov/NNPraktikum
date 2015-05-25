#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_classifier import StupidClassifier
from model.mlp_mnist import MultilayerPerceptron
from report.evaluator import Evaluator
from model.layer import Layer

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)
    myStupidClassifier = StupidClassifier(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           layerWeights=[Layer.loadFromFile("../data/features_layer_80.gz")],
                                           randomLayers=[40],
                                           outputDim=10)
    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nMultilayer perceptron has been training..")
    myMLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    mlpPred = myMLPClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid classifier:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)
    evaluator.printClassificationResult(data.testSet, stupidPred, ['null', 'eins', 'zwei','drei','vier','fuenf','sechs','sieben','acht','neun'])

    print("\nResult of the MLP classifier:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, mlpPred)
    evaluator.printClassificationResult(data.testSet, mlpPred, ['null', 'eins', 'zwei','drei','vier','fuenf','sechs','sieben','acht','neun'])

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
