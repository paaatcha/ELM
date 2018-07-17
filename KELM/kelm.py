# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This class implements the Kernel Extreme Learning Machine (ELM) according to:

[1] Huang, Guang-Bin, et al. "Extreme learning machine for regression and multiclass 
classification." IEEE Transactions on Systems, Man, and Cybernetics, 
Part B (Cybernetics) 42.2 (2012): 513-529.

All the code is very commented to ease the undertanding.

If you find some bug, please e-mail me =)

'''

import numpy as np
import sys

# Insert the utils path
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')


from utils import sigmoid, cont_error
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel


class KELM:
    inTrain = None
    outTrain = None
    kernelType = None
    
    # The constructor method. If you intend to train de ELM, you must fill all parameters.
    # If you already have the weights and wanna only execute the net, just fill W and beta.
    def __init__ (self, inTrain=None, outTrain=None, kernelType = 'rbf'):

        
        if kernelType != 'rbf' and kernelType != 'pol' and kernelType != 'sig':
            print 'ERROR: This kernelType does not exist' 
            raise Exception('ELM initialize error') 
        else:           
            self.kernelType = kernelType
        
        if inTrain is not None and outTrain is not None:           
            self.inTrain = inTrain
            self.outTrain = outTrain            
        else:            
            print 'ERROR: you need to set inTrain and outTrain' 
            raise Exception('ELM initialize error')   
            

    # This method trains and tests the ELM. If you wanna check the training error, set aval=True
    def train_and_test (self, dataTest, realOutput=None, aval=False, reg=0.01, deg=3, gamm=None, coef=1):  
                
        if self.kernelType == 'rbf':
            K = rbf_kernel(self.inTrain, self.inTrain, gamm)
            Ktest = rbf_kernel(dataTest, self.inTrain, gamm)
        elif self.kernelType == 'pol':
            K = polynomial_kernel(self.inTrain, self.inTrain, deg, gamm, coef)
            Ktest = polynomial_kernel(dataTest, self.inTrain, deg, gamm, coef)
        elif self.kernelType == 'sig':
            K = sigmoid_kernel(self.inTrain, self.inTrain, gamm, coef)
            Ktest = sigmoid_kernel(dataTest, self.inTrain, gamm, coef)
 
        I = np.eye(self.inTrain.shape[0])
        outNet = np.dot (np.dot(Ktest, np.linalg.inv(K + reg*I)), self.outTrain)
        
        if aval:        
            miss = float(cont_error (realOutput, outNet))
            si = float(outNet.shape[0])
            acc = (1-miss/si)*100
            print 'Miss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%'       
            return outNet, acc
            
        return outNet, None