import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet    

class MNISTSeven(object):
    '''
    Small subset (5000 instances) of MNIST data to recognize the digit 7
    '''
    
    #dataPath = "data/mnist_seven.csv"
    

    def __init__(self, dataPath,numTrain=3000,numValid=1000,numTest=1000):
        
        self.trainingSet=[]
        self.validationSet=[]
        self.testSet=[]
        
        self.load(dataPath, numTrain, numValid, numTest)
        
    
    def load(self,dataPath,numTrain, numValid, numTest):
            
        print "Loading data from " + dataPath + "..."
            
        data = np.genfromtxt(dataPath, delimiter = ",", dtype = "uint8")
        
        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:numTrain+numValid], data[numTrain+numValid:]
        shuffle(train)
        
        train, valid = train[:numTrain], train[numTrain:]
        
        self.trainingSet, self.validationSet, self.testSet = DataSet(train), DataSet(valid), DataSet(test)
        
        print "Data loaded." 

        