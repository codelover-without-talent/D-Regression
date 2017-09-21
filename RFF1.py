from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from kerpy.Kernel import Kernel
from kerpy.GaussianKernel import GaussianKernel
import os
import multiprocessing as mp

 

  
  
###########################################################################
#Generate sample data
### the data generating function 
# def dat_gen(ns,nt):
#     xs = 5 * rng.rand(ns, 1) 
#     ys = np.sin(xs).ravel()
#    
#     #Add noise to targets
#     ys[::2] += 3 * (0.5 - rng.rand(xs.shape[0]//2))
#     #y = np.reshape(y,(nn,1))    
#     xt = np.linspace(0, 5, nt)[:, None]
#     yt = np.sin(xt).ravel()
#     return xs,ys,xt,yt

def dat_gen(ns,nt):
    xs = 5 * np.random.rand(ns, 1) 
    ys = np.sin(xs).ravel()
   
    #Add noise to targets
    ys[::2] += 3 * (0.5 - np.random.rand(xs.shape[0]//2))
    #y = np.reshape(y,(nn,1))    
    xt = np.linspace(0, 5, nt)[:, None]
    yt = np.sin(xt).ravel()
    return xs,ys,xt,yt

##############################################################################
#computing error
 
### define function for computing ridge regression error 
def err_pre(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1):
    kernel = GaussianKernel(float(sgma))
    aa,y_pre,err_pre0 = kernel.ridge_regress(xs,ys,lamda,Xtst=xt,ytst=yt)
    return y_pre,err_pre0

### define function for rff error
def err_rff(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1,D = 50):
    kernel = GaussianKernel(float(sgma))
    kernel.rff_generate(D)
    bb,y_rff,err_rff0 = kernel.ridge_regress_rff(xs,ys,lamda,Xtst=xt,ytst=yt)
    return y_rff,err_rff0
#################################################################################
def err_computing(nlp,nn,ni,D_ftr,sgma0,lamda0): 
### the error for ridge regression
    err_pres = np.zeros(nlp)

### the error for the rff
    err_rffs = np.zeros(nlp)
 
### the error difference for the two regression
    err_pre_rffs = np.zeros(nlp)


    for num in range(nlp):
### generating data
        xs,ys,xt,yt =dat_gen(nn,ni)

### run xvalidation
        kernel = GaussianKernel(sgma0)
        lamda_pre, width_pre = kernel.xvalidate(xs,ys,method="ridge_regress")
        kernel.rff_generate(D_ftr)
        lamda_rff,width_rff = kernel.xvalidate(xs,ys,method="ridge_regress_rff")
   
### perform ridge regression
        y_pre, err_pre0 = err_pre(xs,ys,xt,yt,width_pre,lamda_pre)
        err_pres[num] = err_pre0
### perform random fourier features
        y_rff, err_rff0 = err_rff(xs,ys,xt,yt,width_rff,lamda_rff,D_ftr)
        err_rffs[num] = err_rff0
### comparing the difference between two predictions        
        err_pre_rffs[num] = np.linalg.norm(y_pre-y_rff)**2
  
 
### the mean square error for ridge regression
    mse_pre = np.mean(err_pres)
 
### the mean square error for rff
    mse_rff = np.mean(err_rffs)
 
### the mean square error for the difference between ridge and rff
    mse_pre_rff = np.mean(err_pre_rffs)
    
    results = np.array([mse_pre,mse_rff,mse_pre_rff])
    return results



def err_processes(processes,nlp,nn,ni,D_ft,sgma0,lamda0):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(err_computing,args=(nlp,nn,ni,D_ftr,sgma0,lamda0)) for D_ftr in D_ft]
    results = [p.get() for p in results]
    return results


### parameters
### number of training data
n_tr = 1000
 
### number of testing data
n_tt = 500



 
### number of simulations
n_sm = 10
 
### feature numbers
DD = np.arange(10,500,10)
 
### initial width
width0 = 1.0
 
### initial regularization parameter
lmda0 = 0.1
 

 
### number of processes
pross = 10
 
err_results = []
err_results = np.array(err_processes(pross,n_sm,n_tr,n_tt,DD,width0,lmda0))

 
mse_pres = err_results[:,0]
mse_rff = err_results[:,1]
mse_pre_rff = err_results[:,2]
  
####################################################################
### save the computation
#os.chdir("/home/michael/stats/ecwork/RFF1/RFF1/results")
pickle_out1 = open("ridge error","wb")
pickle_out2 = open ("rff error", "wb")
pickle_out3 = open("ridge_rff error","wb")
   
pickle.dump(mse_pres, pickle_out1)
pickle.dump(mse_rff,pickle_out2)
pickle.dump(mse_pre_rff,pickle_out3)
   
pickle_out1.close()
pickle_out2.close()
pickle_out3.close()



