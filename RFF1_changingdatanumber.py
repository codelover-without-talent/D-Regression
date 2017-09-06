from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from kerpy.Kernel import Kernel
from kerpy.GaussianKernel import GaussianKernel

 
rng = np.random.RandomState(0)
  
  
###########################################################################
#the data generating function 
def dat_gen(ns,nt):
    xs = 5 * rng.rand(ns, 1) 
    ys = np.sin(xs).ravel()
  
    #Add noise to targets
    ys[::2] += 3 * (0.5 - rng.rand(xs.shape[0]/2))
    #y = np.reshape(y,(nn,1))    
    xt = np.linspace(0, 5, nt)[:, None]
    yt = np.sin(xt).ravel()
    return xs,ys,xt,yt
# define the err computing funciton for ridge
def err_pre(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1):
    kernel = GaussianKernel(float(sgma))
    aa,y_pre,err_pre0 = kernel.ridge_regress(xs,ys,lamda,Xtst=xt,ytst=yt)
    return err_pre0

### define the err computing funciton for rff
def err_rff(xs,ys,xt,yt,sgma = 1.0,lamda = 0.1,D = 50):
    kernel = GaussianKernel(float(sgma))
    kernel.rff_generate(D)
    bb,y_rff,err_rff0 = kernel.ridge_regress_rff(xs,ys,lamda,Xtst=xt,ytst=yt)
    return err_rff0

#################################################################################
nlp = 100
sgma0 = 1.0
lamda0 = 0.1

# number of training data
nn = np.arange(100,5000,100)

# number of testing data
ni = 1000


xxs, yys, xxt, yyt = dat_gen(100,1000)
kernel = GaussianKernel(sgma0)
lamda_pre,width_pre = kernel.xvalidate(xxs, yys, method = "ridge_regress") 

kernel = GaussianKernel(sgma0)       
DD = 20 * 2 * int(round(np.log(1000)/2))
kernel.rff_generate(DD)
lamda_rff,width_rff = kernel.xvalidate(xxs,yys,method="ridge_regress_rff")



err_pres = np.zeros((nlp,len(nn)))
err_rffs = np.zeros((nlp,len(nn)))

for num in range(nlp):
    for ii in range(len(nn)):
        #generating data
        xs, ys, xt, yt = dat_gen(nn[ii],ni)
        err_pres[num,ii] = err_pre(xs,ys,xt,yt,width_pre,lamda_pre)
        
        err_rffs[num,ii] = err_rff(xs,ys,xt,yt,width_rff,lamda_rff,D = DD)

mse_pre = np.mean(err_pres,0)
mse_rff = np.mean(err_rffs,0)        
    
# ###############################################################
pickle_out1 = open("ridge error changing n","wb")
pickle_out2 = open("rff error changing n", "wb")
 
pickle.dump(err_pres,pickle_out1)
pickle.dump(err_rffs,pickle_out2)
 
 
pickle_out1.close()
pickle_out2.close()
   
 
# ###############################################################
# #import data
# pickle_in1 = open("ridge error changing n","rb")
# pickle_in2 = open("rff error changing n", "rb")
# 
# err_pres = pickle.load(pickle_in1)
# err_rffs = pickle.load(pickle_in2)

plt.plot(nn,mse_pre,c='b')
plt.plot(nn, mse_rff)
plt.xlabel("Number of Data")
plt.ylabel("The Mean Square Error")
plt.title("Mean Square Errors for Ridge Regression and RFF")
plt.show() 




