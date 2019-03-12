# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This class implements the Extreme Learning Machine (ELM) according to:

[1] Huang, G.B.; Zhu, Q.Y.; Siew, C.-K. Extreme learning machine: theory and applications.
Neurocomputing, v. 70, n. 1, p. 489 - 501, 2006.

You can choose initialize the input weights with uniform distribution or using random orthogonal projection
proposed by:

[2] Wenhui W. and Xueyi L. ; The selection of input weights of extreme learning machine: A sample
structure preserving point of view, Neurocomputing, 2017, in press

Using this class you can either train the net or just execute if you already know the ELM's weight.
All the code is very commented to ease the undertanding.

If you find some bug, please e-mail me =)

'''

import numpy as np
import sys

# Insert the path to the RBM implementation
sys.path.insert (0, '../RBM/')
from rbm import RBM
from sklearn.decomposition import PCA

# Function to transform the data to one hot enconding
def one_hot_encoding(ind, N=None):
    ind = np.asarray(ind)
    if ind is None:
        return None
    
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

# Function to cont the errors
def contError (vreal, vclass):
      # Getting the matrix binarized
      vclass = one_hot_encoding (vclass)
      [m,n] = vreal.shape
      #dif = vreal - vclass
      err = abs(vreal - vclass).sum()
      return int(err/2)

# Function to compute the sigmoid
def sigmoid (v):
    return 1/(1+np.exp(-v))


class ELM:
    neurons = None
    inTrain = None
    outTrain = None
    W = None
    beta = None
    P = None
    batchSize = None
    
    # The constructor method. If you intend to train de ELM, you must fill all parameters.
    # If you already have the weights and wanna only execute the net, just fill W and beta.
    def __init__ (self, neurons=20, inTrain=None, outTrain=None, W=None, beta=None, init='uniform', batchSize=None):
        # Setting the neuron's number on the hidden layer        
        self.neurons = neurons
        
        # Here we add 1 into the input's matrices to vectorize the bias computation
        self.inTrain = np.concatenate ((inTrain, np.ones([inTrain.shape[0],1])), axis = 1)
        self.outTrain = outTrain
        self.batchSize = batchSize
        
        if inTrain is not None and outTrain is not None:          
            # If you wanna initialize the weights W, you just set it up as a parameter. If don't,
            # let the W=None and the code will initialize it here with random values
            if W is not None:
                self.W = W
            else:
                # The last row is the hidden layer's bias. Because this we added 1 in the training
                # data above.               
                if init == 'uniform':
                    self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])             
                elif init == 'RO':                    
                    if neurons >= inTrain.shape[1]:
                        self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])
                        self.W,_ = np.linalg.qr(self.W.T)
                        self.W = self.W.T
                    else:   
                        print 'Starting PCA...'
                        A = np.random.uniform(-1,1,[neurons,neurons])
                        A,_ = np.linalg.qr(A.T)
                        A = A.T
                        pca = PCA(n_components=neurons)                    
                        wpca = pca.fit_transform(inTrain.T).T
                        print wpca.shape
                        self.W = np.dot(A,wpca).T                        
                        # including the bias
                        b = np.random.uniform(-1,1,[1,self.W.shape[1]])                        
                        self.W = np.vstack((self.W,b))   
                      
#                        self.W = np.random.uniform(-1,1,[inTrain.shape[1]+1,neurons])
#                        self.W,_ = np.linalg.qr(self.W)
#                        self.W = self.W    
#                        print 'W: ', self.W.shape
        else:            
            # In this case, there is no traning. So, you just to fill the weights W and beta
            if beta is not None and W is not None:
                self.beta = beta
                self.W = W
            else:
                print 'ERROR: you set up the input training as None, but you did no initialize the weights' 
                raise Exception('ELM initialize error')   
                

            
    # This method just trains the ELM. If you wanna check the training error, set aval=True
    def train (self, aval=False):
        # Computing the matrix H
        H = sigmoid(np.dot(self.inTrain,self.W))
                      
        # Computing the weights beta
        self.beta = np.dot(np.linalg.pinv(H),self.outTrain)    
        
        #print '\nCONDITION NUMBER:', np.linalg.cond(self.beta), '\n'
        
        if aval == True:            
            H = sigmoid (np.dot(self.inTrain, self.W))
            outNet = np.dot (H,self.beta)
            miss = float(cont_error (self.outTrain, outNet))
            si = float(self.outTrain.shape[0])
            print 'Miss classification on the training: ', miss, ' of ', si, ' - Accuracy: ', (1-miss/si)*100, '%'
            
    def ostrain (self, nInit=5, epc=1, reg=None, aval=False):
        nSam = self.inTrain.shape[0]
        I = np.identity(self.batchSize)
        Ix = np.identity(self.neurons)
        
        # Initialization phase        
        X0 = self.inTrain[0:self.batchSize*nInit]
        Y0 = self.outTrain[0:self.batchSize*nInit]
        H = sigmoid (np.dot(X0, self.W))
        
        if reg is not None:
            P_prev = np.linalg.inv(np.dot(H.T,H) - Ix*reg)
        else:
            P_prev = np.linalg.inv(np.dot(H.T,H))
            
        beta_prev = np.dot(np.dot(P_prev,H.T),Y0)        
               
        for e in range(epc): 
            print '\n### epc {} of {} ####'.format(e,epc)
            # Sequential phase
            for offset in xrange(0,nSam,self.batchSize):
                end = offset + self.batchSize
                
                if end > nSam:
                    break
                
                xBatch, yBatch = self.inTrain[offset:end], self.outTrain[offset:end] 
                
                H = sigmoid (np.dot(xBatch, self.W))
                
                paux1 = np.linalg.inv ((I + np.dot(np.dot(H,P_prev),H.T)))
                paux2 = np.dot(np.dot(P_prev,H.T), paux1) 
                P = P_prev - np.dot(np.dot(paux2,H),P_prev)           
    #            P = P_prev - np.dot (np.dot( np.dot(P_prev,H.T), np.linalg.inv ( (I + np.dot(np.dot(H,P_prev),H.T)) ) ), np.dot(H,P_prev))
                
                beta = beta_prev + np.dot (np.dot(P, H.T), (yBatch - np.dot(H,beta_prev) ))
                
                P_prev = P
                beta_prev = beta
                
                if aval:                
                    outNet = np.dot (H,beta)
                    miss = float(cont_error (yBatch, outNet))
                    si = float(yBatch.shape[0])
                    print 'Miss classification on batch {}/{}: {} of {} - Accuracy: {} %'.format(end,nSam, miss, si, (1-miss/si)*100)          
            
        self.beta = beta
        
        
    def os_rbm_train (self, nInit=5, epc=1, reg=None, aval=False, rbmType='GBRBM',lr=0.001, wc=0.0002, momInit=0.5, momFinal=0.9, cdIter=1, rbmVerbose=True):
        
        rbmNet = RBM (numVis=self.inTrain.shape[1]-1, numHid=self.neurons, rbmType=rbmType)          
        
        nSam = self.inTrain.shape[0]
        I = np.identity(self.batchSize)
        Ix = np.identity(self.neurons)
        
        # Initialization phase        
        X0 = self.inTrain[0:self.batchSize*nInit]
        Y0 = self.outTrain[0:self.batchSize*nInit]        
                
        rbmNet.train_batch (X0[:,0:-1], lr=lr, wc=wc, mom=momInit, cdIter=cdIter, batchSize=self.batchSize, verbose=rbmVerbose, tol=10e-5)        
        self.W = rbmNet.getInputWeights ()        
        
        H = sigmoid (np.dot(X0, self.W))
        if reg is not None:
            P_prev = np.linalg.inv(np.dot(H.T,H) - Ix*reg)
        else:
            P_prev = np.linalg.inv(np.dot(H.T,H))
            
        beta_prev = np.dot(np.dot(P_prev,H.T),Y0)
        
        for e in range(epc):
            print '\n### epc {} of {} ####'.format(e,epc)
            # Sequential phase
            for offset in xrange(0,nSam,self.batchSize):
                end = offset + self.batchSize            
                if end > nSam:
                    break
                
                xBatch, yBatch = self.inTrain[offset:end], self.outTrain[offset:end]
                
                rbmNet.train_batch (xBatch[:,0:-1], lr=lr, wc=wc, mom=momFinal, cdIter=cdIter, batchSize=self.batchSize, verbose=rbmVerbose, tol=10e-5)        
                self.W = rbmNet.getInputWeights () 
                
                H = sigmoid (np.dot(xBatch, self.W))            
                
                paux1 = np.linalg.inv ((I + np.dot(np.dot(H,P_prev),H.T)))
                paux2 = np.dot(np.dot(P_prev,H.T), paux1) 
                P = P_prev - np.dot(np.dot(paux2,H),P_prev)           
                
                beta = beta_prev + np.dot (np.dot(P, H.T), (yBatch - np.dot(H,beta_prev) ))
                
                P_prev = P
                beta_prev = beta
                
                if aval:                
                    outNet = np.dot (H,beta)
                    miss = float(cont_error (yBatch, outNet))
                    si = float(yBatch.shape[0])
                    print 'Miss classification on batch {}/{}: {} of {} - Accuracy: {} %'.format(end,nSam, miss, si, (1-miss/si)*100)     
                    
            
        self.beta = beta        
            
            

    # This method executes the ELM, according to the weights and the data passed as parameter
    def getResultByBatch (self, data, batchSize=100, realOutput=None, aval=False, verbose=False):
        # including 1 because the bias
        dataTest = np.concatenate ((data, np.ones([data.shape[0],1])), axis = 1)       
        nSam = dataTest.shape[0]
        allNetOut = list()
        allAcc = list()
        
        for offset in xrange(0,nSam,batchSize):
            if verbose:
                print 'Testing batch {} of {}'.format(offset,nSam/batchSize)
                
            end = offset + batchSize            
            if end > nSam:
                end = nSam         
            
            if aval:
                xBatch, yBatch = dataTest[offset:end], realOutput[offset:end]        
            else:
                xBatch = dataTest[offset:end]       
        
            # Getting the H matrix
            H = sigmoid (np.dot(xBatch, self.W))            
            netOutput = np.dot (H,self.beta)

            if aval:        
                miss = float(cont_error (yBatch, netOutput))
                si = float(netOutput.shape[0])
                acc = (1-miss/si)*100
                if verbose:
                    print '\nMiss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%'       
                #return netOutput, acc
                allNetOut.append(netOutput)
                allAcc.append(acc)
        
        allAcc = np.asarray(allAcc)
        return netOutput, allAcc.mean()        
            
    # This method executes the ELM, according to the weights and the data passed as parameter
    def getResult (self, data, realOutput=None, aval=False):
        data = np.asarray(data)
        # including 1 because the bias
        dataTest = np.concatenate ((data, np.ones([data.shape[0],1])), axis = 1)       
        
        # Getting the H matrix
        H = sigmoid (np.dot(dataTest, self.W))
        netOutput = np.dot (H,self.beta)

        if aval:        
            miss = float(cont_error (realOutput, netOutput))
            si = float(netOutput.shape[0])
            acc = (1-miss/si)*100
            print '\nMiss classification on the test: ', miss, ' of ', si, ' - Accuracy: ',acc , '%'       
            return netOutput, acc
            
        return netOutput, None
        
    # This method saves the trained weights as a .csv file
    def saveELM (self, nameFile='T1'):
        np.savetxt('weightW'+nameFile+'.csv', self.W)
        np.savetxt('weightBeta'+nameFile+'.csv', self.beta)
        
    # This method computes the norm for input and output weights
    def getNorm (self, verbose=True):
        wNorm = np.linalg.norm(self.W)
        betaNorm = np.linalg.norm(self.beta)
        if verbose:
            print 'The norm of W: ', wNorm
            print 'The norm of beta: ', betaNorm
        return wNorm, betaNorm
        



     

      







