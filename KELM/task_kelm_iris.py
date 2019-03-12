# -*- coding: utf-8 -*-
"""

Author: André Pacheco
E-mail: pacheco.comp@gmail.com

"""

import sys
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/machine-learning/machine-learning/utils/')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/DataClass')
#sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/RBM')
sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/KELM')

import numpy as np
#from elm import *
#from elm_tensorflow import *
from kelm import *
from utils import *
from data import data
#from rbm import *
import time
import gc


# loading the data set
print 'Loading the dataset...'
irisAll = np.genfromtxt('/home/patcha/Datasets/Iris/iris.csv', delimiter=',')

acc = list()
tim = list()

for i in range(30):
    iris = data (dataset=irisAll, percTrain=0.7, percVal=0, percTest=0.3, normType='mean')
    print iris
      
    init = time.time()    
    elmNet = KELM (inTrain=iris.trainIn, outTrain=iris.trainOut, kernelType ='pol')
    res, a = elmNet.train_and_test(iris.testIn, iris.testOut, aval=True)    
    end = time.time()    
    acc.append(a)
    tim.append(end-init)  
        
    del(iris)
    del(elmNet)
    gc.collect()
    
    
    
acc = np.asarray(acc)
tim = np.asarray(tim)

print '\nMedia: ', acc.mean(), '\nStd: ', acc.std()
print '\nTime: ', tim.mean(), '\nStd: ', tim.std()











