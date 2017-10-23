'''
Created on Sep 15, 2017

@author: michael
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from kerpy.Kernel import Kernel
from kerpy.GaussianKernel import GaussianKernel



#import data
import pickle


pickle_in1 = open("ridge error changing n","rb")
pickle_in2 = open("rff 10 error changing n", "rb")
pickle_in3 = open("rff 50 error changing n", "rb")
pickle_in4 = open("rff 100 error changing n", "rb")
 
ridge_results = pickle.load(pickle_in1)
rff_50_results = pickle.load(pickle_in2)
rff_100_results = pickle.load(pickle_in3)
rff_200_results = pickle.load(pickle_in4)
nn = np.arange(100,1000,10)


plt.plot(nn,ridge_results,c='b')
plt.plot(nn, rff_50_results)
plt.plot(nn, rff_100_results)
plt.plot(nn, rff_200_results)
plt.xlabel("Number of Data")
plt.ylabel("The Mean Square Error")
plt.title("Mean Square Errors for Ridge Regression and RFF")
plt.show() 


