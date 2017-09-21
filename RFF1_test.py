'''
Created on Sep 15, 2017

@author: michael
'''
#import data
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from kerpy.Kernel import Kernel
from kerpy.GaussianKernel import GaussianKernel
import os
import multiprocessing as mp

import pickle
pickle_in1 = open("ridge error","rb")
pickle_in2 = open("rff error","rb")
pickle_in3 = open("ridge_rff error","rb")
 
mse_pres = pickle.load(pickle_in1)
mse_rffs = pickle.load(pickle_in2)
mse_pre_rffs = pickle.load(pickle_in3)
 
print mse_pres
print mse_rffs
print mse_pre_rffs
 
 
DD = np.arange(10,500,10) 
 
     
fig = plt.figure()
   
ax1 = fig.add_subplot(2,1,1) 
axes = plt.gca()
plt.plot(DD,np.repeat(np.mean(mse_pres),len(DD)),c='b')
plt.plot(DD, mse_rffs)
plt.xlabel("Number of Features")
plt.ylabel("The Mean Square Error")
plt.title("Mean Square Errors for Ridge Regression and RFF")
   
ax2 = fig.add_subplot(2,1,2)
plt.plot(DD,mse_pre_rffs,c='b')
plt.xlabel("Number of Features")
plt.ylabel("The Error Difference")
plt.title("The Error Difference between Ridge and RFF")
plt.show()  