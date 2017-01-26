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
import seaborn as sns
from numpy import random as rand



def MoM(data,delta=0.05):
    K=int(8*mat.log(1/delta))
    rand.shuffle(data)
    sp_data=np.array_split(data,K)
    means=list(map(np.mean,sp_data))
    return np.median(means)


#distr: Normal(mu,var), Binomial(n,p), Poisson(lbd), Gamma(k,theta), 
#Pareto(a), T-Student(nu), Weibull, Lognorm
#parm1 and parm2: parameters of the distribution
#n: number of random numbers in each sample
#draws: number of times it is going to be performed
#delta: precision
def perf_MoM(distr,parm1,parm2,n,draws,delta):
    p1=[parm1]*draws
    p2=[parm2]*draws
    num=[n]*draws
    if distr=="Normal":
        data=list(map(rand.normal,p1,p2,num))
        exp_value=[parm1]*draws
    elif distr=="Binomial":
        data=list(map(rand.binomial,p1,p2,num))
        exp_value=[parm1*n]*draws
    elif distr=="Poisson":
        data=list(map(rand.poisson,p1,num))
        exp_value=[parm1]*draws
    elif distr=="Gamma":
        data=list(map(rand.gamma,p1,p2,num))
        exp_value=[parm1*parm2]*draws
    elif distr=="Pareto":
        data=list(map(rand.pareto,p1,num))
        exp_value=[parm1/(parm1-1)-1]*draws
    elif distr=="T-Student":
        data=list(map(rand.standard_t,p1,num))
        exp_value=[0]*draws
    elif distr=="Weibull":
        data=list(map(op.mul,list(map(rand.weibull,p1,num)),p2))
        exp_value=[parm2*(mat.gamma(1+1/parm1))]*draws
    elif distr=="Lognorm":
        data=list(map(rand.lognormal,p1,p2,num))
        exp_value=[mat.exp(parm1+parm2/2)]*draws
    MoM_est=list(map(MoM,data))
    mean_est=list(map(np.mean,data))
    dif_mean=list(map(abs,map(op.sub,exp_value,mean_est)))
    dif_MoM=list(map(abs,map(op.sub,exp_value,MoM_est))) 
    wc_mean=max(dif_mean)
    wc_MoM=max(dif_MoM)
    bc_mean=min(dif_mean)
    bc_MoM=min(dif_MoM)    
    ave_mean=sum(dif_mean)/len(dif_mean)
    ave_MoM=sum(dif_MoM)/len(dif_MoM)
    ave_mean_MoM=sum(MoM_est)/len(MoM_est)
    ave_mean_mean=sum(mean_est)/len(mean_est)
    #plt.hist(MoM_est,color="blue")
    #plt.hist(mean_est,color="green")
    compare=pd.DataFrame([['','Exp value','Worst case dev','Best case dev',
    'Average dev','ave_mean_MoM'],
                ['Mean',exp_value,wc_mean,bc_mean,ave_mean,ave_mean_MoM],
                ['MoM',exp_value,wc_MoM,bc_MoM,ave_MoM,ave_mean_mean]])
    return compare


test=perf_MoM("Pareto",20,1,100000,1000,0.01)





