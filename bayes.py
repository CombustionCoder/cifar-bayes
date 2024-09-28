# Author: Jukka Hirvonen
# This script classifies CIFAR-10 images using bayesian probability.
# It produces a comparison of naive and non-naive bayesian algorithms.

import pickle
import time

start_time = time.time() # for calculating runtime
print("Start time: " + time.ctime(start_time))
print("-----")

import numpy as np
from skimage import transform
from scipy import stats

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def data_open():
    datadict = unpickle('c:/s/ml/c10/cifar-10-batches-py/data_batch_1')
    datadict2 = unpickle('c:/s/ml/c10/cifar-10-batches-py/data_batch_2')
    datadict3 = unpickle('c:/s/ml/c10/cifar-10-batches-py/data_batch_3')
    datadict4 = unpickle('c:/s/ml/c10/cifar-10-batches-py/data_batch_4')
    datadict5 = unpickle('c:/s/ml/c10/cifar-10-batches-py/data_batch_5')
    testdict = unpickle('c:/s/ml/c10/cifar-10-batches-py//test_batch')
    
    X = datadict["data"]
    Y = datadict["labels"]
    X2 = datadict2["data"]
    Y2 = datadict2["labels"]
    X3 = datadict3["data"]
    Y3 = datadict3["labels"]
    X4 = datadict4["data"]
    Y4 = datadict4["labels"]
    X5 = datadict5["data"]
    Y5 = datadict5["labels"]
    X = np.concatenate((X,X2,X3,X4,X5), axis=0)
    Y = np.concatenate((Y,Y2,Y3,Y4,Y5), axis=0)
    
    Tx = testdict["data"]
    Ty = testdict["labels"]
    return(X,Y,Tx,Ty)

def cifar10_resizer(pics,size): # resize to size x size pixels
    numpics=pics.shape[0]
    resized=transform.resize(pics,(numpics,size,size), preserve_range=True)
    return(resized)

def cifar_10_bayes_learn(Xp,Yp):
    rgb=Xp.reshape((sampleSize,siz*siz*3))
    mur=np.array([])
    for i in range(10): # mean
        wheres=np.where(y==i)
        mr=np.mean(rgb[wheres], axis=0)
        mur=np.hstack((mur,mr))
    mur=mur.reshape((10,siz*siz*3))
    mur=mur.transpose(1,0)
    
    covariance=np.zeros((siz*siz*3,siz*siz*3,1))
    for i in range(10): # covariance
        wheres=np.where(y==i)
        cut=rgb[wheres]
        cov=np.cov(cut, rowvar=False)
        covariance=np.dstack((covariance,cov))
    covariance=covariance[:,:,1:]
    
    p=np.histogram(Yp) # priors
    p=p[0]
    p=p/sum(p)
    return(covariance,mur,p)

def cifar_10_classifier_bayes(tt,mu,covariance,p):
    tx=tt.reshape((10000,siz*siz*3), order='C')
    p_reds=np.array([])
    for i in range(1,tt_rows+1):
        p_red=np.array([])
        for j in range(1,mu_cols+1):
            pr = stats.multivariate_normal.pdf(tx[i-1,:],mu[:,j-1],covariance[:,:, j-1])*p[j-1]
            p_red=np.append(p_red, pr)
        p_reds=np.append(p_reds, p_red)
    p_reds=p_reds.reshape(tt_rows,10)    
    bayes=p_reds
    res1=np.argmax(bayes, axis=1)
    return(res1)

def cifar_10_naivebayes_learn(Xp,Yp):
    rgb=Xp.reshape((sampleSize,siz*siz*3), order='C')
    mur=np.array([])
    for i in range(10): # mean
        wheres=np.where(y==i)
        mr=np.mean(rgb[wheres], axis=0)
        mur=np.hstack((mur,mr))
    
    mur=mur.reshape((10,siz*siz,3))
   
    sigr=np.array([])
    for i in range(10): # var
        wheres=np.where(y==i)
        sr=np.var(rgb[wheres], axis=0)
        sigr=np.hstack((sigr,sr))
    sigr=sigr.reshape((10,siz*siz,3))
  

    p=np.histogram(Yp) # priors
    p=p[0]
    p=p/sum(p)
    return(sigr,mur,p)

def cifar_10_classifier_naivebayes(tt,mu,sigma2s,p):
    tt_red=tt[:,:,:,0]
    tt_green=tt[:,:,:,1]
    tt_blue=tt[:,:,:,2]
    tt_red=tt_red.reshape((10000,siz*siz))
    tt_green=tt_green.reshape((10000,siz*siz))
    tt_blue=tt_blue.reshape((10000,siz*siz))

    
    # red
    red=np.ones([10000,10])
    for pixel in range(siz*siz):
        p_reds=np.array([])
        for i in range(1,tt_rows+1): # sample
            p_red=np.array([])
            for j in range(1,11): # class
                pr = (1/np.sqrt(2*np.pi*sigma2s[j-1,pixel,0])*np.exp(-1/(2*sigma2s[j-1,pixel,0])*(tt_red[i-1,pixel]-mu[j-1,pixel,0])**2))*tt_red[i-1,pixel]*p[j-1]
                p_red=np.append(p_red, pr)
            p_reds=np.append(p_reds, p_red)
        p_reds=p_reds.reshape(tt_rows,10)
        red=np.dstack((red,p_reds))
    red=red[:,:,1:]

    
    # green
    green=np.ones([10000,10])
    for pixel in range(siz*siz):
        p_greens=np.array([])
        for i in range(1,tt_rows+1): # sample
            p_green=np.array([])
            for j in range(1,11): # class
                pr = (1/np.sqrt(2*np.pi*sigma2s[j-1,pixel,1])*np.exp(-1/(2*sigma2s[j-1,pixel,1])*(tt_green[i-1,pixel]-mu[j-1,pixel,1])**2))*tt_green[i-1,pixel]*p[j-1]
                p_green=np.append(p_green, pr)
            p_greens=np.append(p_greens, p_green)
        p_greens=p_greens.reshape(tt_rows,10)
        green=np.dstack((green,p_greens))
    green=green[:,:,1:]
    
    # blue
    blue=np.ones([10000,10])
    for pixel in range(siz*siz):
        p_blues=np.array([])
        for i in range(1,tt_rows+1): # sample
            p_blue=np.array([])
            for j in range(1,11): # class
                pr = (1/np.sqrt(2*np.pi*sigma2s[j-1,pixel,2])*np.exp(-1/(2*sigma2s[j-1,pixel,2])*(tt_blue[i-1,pixel]-mu[j-1,pixel,2])**2))*tt_blue[i-1,pixel]*p[j-1]
                p_blue=np.append(p_blue, pr)
            p_blues=np.append(p_blues, p_blue)
        p_blues=p_blues.reshape(tt_rows,10)
        blue=np.dstack((blue,p_blues))
    blue=blue[:,:,1:]
    bayes=np.array([])

    bayes=red*green*blue # multiply probabilities from colors together
    bayes=np.prod(bayes,axis=2) # multiplying probabilities along axis

    res1=np.argmax(bayes, axis=1)

    return(res1)

def class_ac(pred,gt):
    score = 0
    for i in range(len(ty)):
        if pred[i] == gt[i]:
            score += 1
    acc = score / min(len(pred),len(gt))
    print("Training samples " + str(sampleSize) + ", image size " +str(siz) +" x "+str(siz))
    print("Correctly classified " + str(score) + " out of " + str(min(len(pred),len(gt))) + " test samples = " + str(round((acc*100),2)) + "%")

# execution segment below ---------------------

X,Y,Tx,Ty=data_open()

print("Naive bayes classifier:")
# sample size limiter for testing 1-50000
for i in range(9):
    sampleSize = 50000
    y = Y[:sampleSize] # training labels
    tt=Tx[:sampleSize,:] # test samples
    ty=Ty[:sampleSize] # test labels
    xx=X[:sampleSize,:] # training samples
    
    xx = xx.reshape(sampleSize, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    tt = tt.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    
    sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9)
    siz=sizes[i]    
    xx=cifar10_resizer(xx,siz)
    tt=cifar10_resizer(tt,siz)
    
    sigma2s, mu, p=cifar_10_naivebayes_learn(xx,y)
    
    tt_rows, tt_2, tt_3, tt_4 = tt.shape
    
    result=cifar_10_classifier_naivebayes(tt,mu,sigma2s,p)
    class_ac(result, ty)


print("-----")
print("Non-naive bayes classifier:")
# sample size limiter for testing 1-50000
for i in range(9):
    sampleSize = 50000
    y = Y[:sampleSize] # training labels
    tt=Tx[:sampleSize,:] # test samples
    ty=Ty[:sampleSize] # test labels
    xx=X[:sampleSize,:] # training samples
    
    xx = xx.reshape(sampleSize, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    tt = tt.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    
    sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9)
    siz=sizes[i]    
    xx=cifar10_resizer(xx,siz)
    tt=cifar10_resizer(tt,siz)
    
    covariance, mu, p=cifar_10_bayes_learn(xx,y)
    
    tt_rows, tt_2, tt_3, tt_4 = tt.shape
    mu_rows, mu_cols = mu.shape
    
    result=cifar_10_classifier_bayes(tt,mu,covariance,p)
    class_ac(result, ty)


# lets see how long it took
print("-----")
print("End time: " + time.ctime(time.time()))
print("Runtime: " + str(time.time()-start_time) +" seconds")

    