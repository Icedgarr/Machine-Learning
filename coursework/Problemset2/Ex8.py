# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 23:22:30 2017

@author: roger
"""

import sklearn as sk
import math as mat
import numpy as np
from numpy import random as rand
from sklearn import neighbors

def risk_knn(n,d,k,m):

    #Generate X of uniform distribution
    train_X=rand.uniform(low=0,high=1,size=(n,d))
    
    #Generates Y from X as a Bernoulli with p=x[0] in each case
    train_p=train_X[:,0]
    train_Y=rand.binomial([1]*len(train_p),train_p,size=len(train_p))
    
    #Zip data
    #train_data=list(zip(train_X,train_Y))
    
    #Generate the KNN function
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    
    #Train the models
    knn.fit(train_X,train_Y)
    
    #Generate test data
    test_X=rand.uniform(low=0,high=1,size=(m,d))
    test_p=test_X[:,0]
    test_Y=rand.binomial([1]*len(test_p),test_p,size=len(test_p))
    
    #Predict values
    knn_pred=knn.predict(test_X)
    
    #Compute error frequency
    knn_check=test_Y==knn_pred
    knn_wrong=m-sum(knn_check)
    return knn_wrong/m
    

n=[10000]*5
m=[1000000]*5
k=[1,3,5,7,9]

#Estimate risk knn for different d
d=[1]*5
risks1=list(map(risk_knn,n,d,k,m))
risks1

d=[2]*5
risks2=list(map(risk_knn,n,d,k,m))
risks2

d=[3]*5
risks3=list(map(risk_knn,n,d,k,m))
risks3

d=[4]*5
risks4=list(map(risk_knn,n,d,k,m))
risks4

d=[5]*5
risks5=list(map(risk_knn,n,d,k,m))
risks5

d=[10]*5
risks10=list(map(risk_knn,n,d,k,m))
risks10

d=[15]*5
risks15=list(map(risk_knn,n,d,k,m))
risks15

d=[25]*5
risks25=list(map(risk_knn,n,d,k,m))
risks25

d=[50]*5
risks50=list(map(risk_knn,n,d,k,m))
risks50

