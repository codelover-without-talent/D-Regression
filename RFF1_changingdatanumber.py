from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from kerpy.Kernel import Kernel
from kerpy.GaussianKernel import GaussianKernel

 

  
###########################################################################
#the data generating function 
def dat_gen(ns,nt):
    xs = 5 * np.random.rand(ns, 1) 
    ys = np.sin(xs).ravel()
  
    #Add noise to targets
    ys[::2] += 3 * (0.5 - np.random.rand(xs.shape[0]//2))
    #y = np.reshape(y,(nn,1))    
    xt = np.linspace(0, 5, nt)[:, None]
    yt = np.sin(xt).ravel()
    return xs,ys,xt,yt

# x_tr,y_tr,x_tt,y_tt = dat_gen(100,100)
# 
# print x_tr
# define the err computing function for ridge
def err_pre(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1):
    kernel = GaussianKernel(float(sgma))
    aa,y_pre,err_pre0 = kernel.ridge_regress(xs,ys,lamda,xt,yt)
    return err_pre0

### define the err computing function for rff
def err_rff(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1,D = 50):
    kernel = GaussianKernel(float(sgma))
    kernel.rff_generate(D)
    bb,y_rff,err_rff0 = kernel.ridge_regress_rff(xs,ys,lamda,xt,yt)
    return err_rff0


def ridge_error(nlp,nn,ni,sgma = 1.0):
    mse_pres = np.zeros(nlp)
    for ii in np.arange(nlp):
        x_tr,y_tr,x_tt,y_tt = dat_gen(nn,ni)
        kernel = GaussianKernel(float(sgma))
        lamda_pre,width_pre = kernel.xvalidate(x_tr,y_tr,method = "ridge_regress")
        
        mse_pres[ii] = err_pre(x_tr,y_tr,x_tt,y_tt,width_pre,lamda_pre)
    mse_pre = np.mean(mse_pres)
    return mse_pre

def rff_error(nlp,nn,ni,D,sgma = 1.0):
    mse_rffs = np.zeros(nlp)
    for ii in np.arange(nlp):
        x_tr,y_tr,x_tt,y_tt =dat_gen(nn,ni)
        kernel = GaussianKernel(float(sgma))
        kernel.rff_generate(D)
        lamda_rff,width_rff = kernel.xvalidate(x_tr,y_tr,method="ridge_regress_rff")
        
        mse_rffs[ii] = err_rff(x_tr,y_tr,x_tt,y_tt,width_rff,lamda_rff,D)
    
    mse_rff = np.mean(mse_rffs)
    return mse_rff


def ridge_err_processes(processes,nlp,nn,ni,sgma):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(ridge_error,args=(nlp,nn0,ni,sgma)) for nn0 in nn]
    results = [p.get() for p in results]
    return results

def rff_err_processes(processes,nlp,nn,ni,D,sgma):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(rff_error,args=(nlp,nn0,ni,D,sgma)) for nn0 in nn]
    results = [p.get() for p in results]
    return results

processes0 = 10
nlp0 = 10
nn = np.arange(100,1000,10)
ni0 = 100
sgma0 = 1.0

# 
ridge_results = ridge_err_processes(processes0,nlp0,nn,ni0,sgma0)    
 
rff_10_results = rff_err_processes(processes0,nlp0,nn,ni0,10,sgma0)
 
rff_50_results = rff_err_processes(processes0,nlp0,nn,ni0,50,sgma0)
 
rff_100_results = rff_err_processes(processes0,nlp0,nn,ni0,100,sgma0)
 





    
# ###############################################################
#save computation
pickle_out1 = open("ridge error changing n","wb")
pickle_out2 = open("rff 10 error changing n", "wb")
pickle_out3 = open("rff 50 error changing n", "wb")
pickle_out4 = open("rff 100 error changing n", "wb")
  
pickle.dump(ridge_results,pickle_out1)
pickle.dump(rff_10_results,pickle_out2)
pickle.dump(rff_50_results,pickle_out3)
pickle.dump(rff_100_results,pickle_out4)
  
  
pickle_out1.close()
pickle_out2.close()
pickle_out3.close()   
pickle_out4.close() 
