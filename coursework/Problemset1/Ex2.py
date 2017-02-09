# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:59:40 2017

@author: roger
Median of Mean estimator
"""
import sklearn
import math as mat
import numpy as np
import itertools as itt
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random as rand
import functools as ft

#Given a sample (data) and a precision (delta). Computes de MoM estimator with 
#K=[8*log(1/delta)] divisions.
def MoM(data,delta=0.1):
    K=int(8*mat.log(1/delta)) #compute num of divisions.
    rand.shuffle(data) #shuffle the data.
    sp_data=np.array_split(data,K) #split the data into K sets. 
    means=list(map(np.mean,sp_data)) #computes the mean in each set.
    return np.median(means) #computes the median of the means.


#distr: Normal(mu,var), Binomial(n,p), Poisson(lbd), Gamma(k,theta), 
#Pareto(a), T-Student(nu), Weibull, Lognorm
#parm1 and parm2: parameters of the distribution
#n: number of random numbers in each sample
#draws: number of times it is going to be performed
#delta: precision
def perf_MoM(distr,parm1,parm2,n,draws,delta):
    #replicate the parameters a number of times indicated by the draws.    
    p1=[parm1]*draws
    p2=[parm2]*draws
    num=[n]*draws
    prec=delta
    if distr=="Normal": #Generates 'draws' iid Normal samples of 'num' elements
        data=list(map(rand.normal,p1,p2,num)) 
        exp_value=[parm1]*draws
    elif distr=="Binomial": #Generates 'draws' iid Binomial samples of 'num' elements
        data=list(map(rand.binomial,p1,p2,num))
        exp_value=[parm1*parm2]*draws
    elif distr=="Poisson": #Generates 'draws' iid Poisson samples of 'num' elements
        data=list(map(rand.poisson,p1,num))
        exp_value=[parm1]*draws
    elif distr=="Gamma": #Generates 'draws' iid Gamma samples of 'num' elements
        data=list(map(rand.gamma,p1,p2,num))
        exp_value=[parm1*parm2]*draws
    elif distr=="Pareto": #Generates 'draws' iid Pareto samples of 'num' elements
        data=list(map(rand.pareto,p1,num))
        exp_value=[parm1/(parm1-1)-1]*draws
    elif distr=="T-Student": #Generates 'draws' iid T-student samples of 'num' elements
        data=list(map(rand.standard_t,p1,num))
        exp_value=[0]*draws
    elif distr=="Weibull": #Generates 'draws' iid Weibull samples of 'num' elements
        data=list(map(op.mul,list(map(rand.weibull,p1,num)),p2))
        exp_value=[parm2*(mat.gamma(1+1/parm1))]*draws
    elif distr=="Lognorm": #Generates 'draws' iid Lognorm samples of 'num' elements
        data=list(map(rand.lognormal,p1,p2,num))
        exp_value=[mat.exp(parm1+parm2/2)]*draws
    MoM_est=list(map(ft.partial(MoM,delta=prec),data)) #Compute the 'draws' estimators using MoM
    mean_est=list(map(np.mean,data)) #Compute the 'draws' estimators using sample mean
    dif_mean=list(map(abs,map(op.sub,exp_value,mean_est))) #Compute the error of each sample mean
    dif_MoM=list(map(abs,map(op.sub,exp_value,MoM_est))) #Compute the error of each MoM
    #eval performance using worst case    
    wc_mean=max(dif_mean) 
    wc_MoM=max(dif_MoM)
    #eval performance using best case
    bc_mean=min(dif_mean)
    bc_MoM=min(dif_MoM)    
    #eval performance using average of error
    ave_mean=sum(dif_mean)/len(dif_mean)
    ave_MoM=sum(dif_MoM)/len(dif_MoM)
    #compute the mean of the estimators over the 'draws' samples.
    ave_mean_MoM=np.mean(MoM_est)
    ave_mean_mean=np.mean(mean_est)
    #plt.hist(MoM_est,color="blue")
    #plt.hist(mean_est,color="green")
    compare=pd.DataFrame([['','Exp value','Worst case dev','Best case dev',
    'Average dev','ave_mean'],
                ['Mean',exp_value[0],wc_mean,bc_mean,ave_mean,ave_mean_mean],
                ['MoM',exp_value[0],wc_MoM,bc_MoM,ave_MoM,ave_mean_MoM]])
    return compare


n=100
parm1=0
parm2=1000
draws=1000
delta=0.1


8*np.log(1/delta)


test=perf_MoM("Normal",parm1,parm2,n,draws,delta)

p1=[parm1]*draws
p2=[parm2]*draws
num=[n]*draws

pr=[prec]*n

data=list(map(rand.pareto,p1,num))
a=list(zip(map(rand.pareto,p1,num),pr))
exp_value=[1.5/(1.5-1)-1]*1000



