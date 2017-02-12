# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 23:22:30 2017

@author: roger
"""

import sklearn as sk
import numpy as np
from numpy import random as rand
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns

def risk_knn(n,d,k,m):

    train_X=rand.uniform(low=0,high=1,size=(n,d)) #Generate X of uniform distribution    
    train_p=train_X[:,0] #Generates Y from X as a Bernoulli with p=x[0] in each case
    train_Y=rand.binomial([1]*len(train_p),train_p,size=len(train_p))
    
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)  #Generate the KNN function
    knn.fit(train_X,train_Y) #Train the models
    
    test_X=rand.uniform(low=0,high=1,size=(m,d)) #Generate test data
    test_p=test_X[:,0]
    test_Y=rand.binomial([1]*len(test_p),test_p,size=len(test_p))
    
    knn_pred=knn.predict(test_X)     #Predict values
    
    knn_check=test_Y==knn_pred #Compute error frequency
    knn_wrong=m-sum(knn_check)
    return knn_wrong/m

n=[10000]*5
m=[10000]*5
k=[1,3,5,7,9]

#Estimate risk knn for different d
i=1
all_risks=[0]*6
for d in [1,5,10,50,100,1000]:

    s=[d]*5
    risks=list(map(risk_knn,n,s,k,m))
    all_risks[i-1]=risks
    #plt.subplot(3,2,i)    
    plt.plot(k, risks)
    #plt.ylim([0.15,0.5])
    plt.title("KNN's Risk vs. 'K'NN. N= %.0f D=%.0f" %(n[0],d)) 
    i+=1
plt.legend(['d=1','d=5', 'd=10', 'd=50', 'd=100', 'd=1000' ],loc='upper right')
plt.xlabel('Number of Neighbors')
plt.ylabel('Risk')
plt.title("KNN's Risk vs Number of Neighbours for N= %.0f" %n[0])


