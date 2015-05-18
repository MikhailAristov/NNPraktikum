#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.mlp_mnist import MultilayerPerceptron
from report.evaluator import Evaluator

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000, oneHot=False)
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           hiddenLayers=[28],
                                           outputDim=10)
    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMultilayer perceptron has been training..")
    myMLPClassifier.train()
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()

    #print np.array(zip(data.testSet.label, mlpPred))

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the MLP recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, mlpPred)
    evaluator.printClassificationResult(data.testSet, mlpPred, ['null', 'eins', 'zwei','drei','vier','fuenf','sechs','sieben','acht','neun'])

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
