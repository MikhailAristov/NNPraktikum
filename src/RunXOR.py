#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.xor_data import XORData
from model.stupid_recognizer import StupidRecognizer
from model.mlp_xor import MultiLayerPerceptronXOR
from report.evaluator import Evaluator

__author__ = "Mikhail Aristov"
__email__ = "mikhail.aristov@student.kit.edu"

def main():
    data = XORData()
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myMLPerceptron = MultiLayerPerceptronXOR(data.trainingSet,
                                             data.validationSet,
                                             data.testSet,
                                             hiddenLayers=[2],
                                             outputDim=1)
    
    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")
    
    print("\nMultilayer perceptron has been training..")
    myMLPerceptron.train()
    print("Done..")
    
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = map(lambda b: 1.0 if b else -1.0, myStupidClassifier.evaluate()) # The default stupid recognizer returns True/False, but we need 1.0/-1.0
    MLPPred = myMLPerceptron.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("Result of the MLP recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, MLPPred)

    # eval.printConfusionMatrix(data.testSet, pred)
    # eval.printClassificationResult(data.testSet, pred, target_names)

if __name__ == '__main__':
    main()
