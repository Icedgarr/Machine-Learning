# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:22:07 2017

@author: roger
"""


import sklearn as sk
import numpy as np
from numpy import random as rand
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math as mat
import pylab



#Generate the data
def gen_data(n,d):
    Y=rand.binomial(1,1/2,n)

    X=np.zeros((n,d))
    cov=np.identity(d)
    for i in range(0,n):
        if Y[i]==0:
            vmean=[0]*d
        else:
            if d<=2:
                vmean=[1]*d
            else:
                vmean=[1,1]+[0]*(d-2)    
        X[i]=rand.multivariate_normal(vmean,cov,1)
    return X,Y



#Decision tree
m=10000
ns = [50,100,500,1000,5000,10000,50000]
logns = [mat.log(x,10) for x in ns]    
ds = [3, 5, 10, 20, 30]
ks=[2,5,10,20,50,100,500]
j=0
for k in ks:

    scores_list = []

    for d in ds:
        
        accuracy = np.zeros(len(ns))        
        X_test,Y_test=gen_data(m,d)
                
        for i in range(len(ns)):
            
            n=ns[i]    
            X_train,Y_train=gen_data(n,d)
            dec_tree = DecisionTreeClassifier(max_leaf_nodes=k)
            dec_tree.fit(X_train,Y_train)
            Y_pred = dec_tree.predict(X_test)
            accuracy[i] = 1-sum(abs(Y_test - Y_pred))/m
            
        scores_list.append(accuracy)
    
    plt.figure(j)
    j+=1
    for z in range(len(ds)):
    
        plt.plot(logns, scores_list[z], label = 'D = %s'%ds[z])
    
    plt.xlabel('log(N)')
    plt.ylabel('accuracy')
    plt.legend(loc="upperright")   
    plt.title('Acc. of the dec. tree with K='+str(k)+' terminal nodes')
    
    pylab.savefig('/home/roger/Desktop/BGSE/14D005 Machine Learning/problemsets/Problemset4/tree_acc_k'+str(k)+'.pdf')
    


#Bagging

def bagging_dec_tree(train_set,test_set,max_leaf_nodes,times_bag,size_bag):
    dec_tree=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    X_test=test_set[:,:(-1)]
    Y_test=test_set[:,(-1)]
    Y_array=Y_test
    for i in range(times_bag):
        bag=train_set[rand.randint(len(train_set),size=size_bag)]
        bag_X=bag[:,:(-1)]
        bag_Y=bag[:,(-1)]
        dec_tree.fit(bag_X,bag_Y)
        Y_array=np.column_stack((Y_array,dec_tree.predict(X_test)))
    
    Y_pred=np.median(Y_array[:,1:],axis=1)
    return Y_pred


m=10000
ns = [50,100,500,1000,5000,10000,50000]
logns = [mat.log(x,10) for x in ns]    
ds = [3, 5, 10, 20, 30]
ks=[2,5,10,20,50,100,500]
num_bags=100
j=0
for k in ks:

    scores_list = []

    for d in ds:
        
        accuracy = np.zeros(len(ns))        
        X_test,Y_test=gen_data(m,d)
        test_set=np.column_stack((X_test,Y_test))
                
        for i in range(len(ns)):
            
            n=ns[i]    
            X_train,Y_train=gen_data(n,d)
            train_set=np.column_stack((X_train,Y_train))
            Y_pred = bagging_dec_tree(train_set,test_set,k,num_bags,mat.floor(0.2*n))
            accuracy[i] = 1-sum(abs(Y_test - Y_pred))/m
            
        scores_list.append(accuracy)
    
    plt.figure(j)
    j+=1
    for z in range(len(ds)):
    
        plt.plot(logns, scores_list[z], label = 'D = %s'%ds[z])
    
    plt.xlabel('log(N)')
    plt.ylabel('accuracy')
    plt.legend(loc="upperright")   
    plt.title('Acc. of the dec. tree with K='+str(k)+' terminal nodes')
    
    pylab.savefig('/home/roger/Desktop/BGSE/14D005 Machine Learning/problemsets/Problemset4/bag_acc_k'+str(k)+'.pdf')
    


#random sub-space
def rsubspace_dec_tree(train_set,test_set,max_leaf_nodes,times_rspace):
    dec_tree=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    X_test=test_set[:,:(-1)]
    Y_test=test_set[:,(-1)]
    train_X=train_set[:,:(-1)]
    train_Y=train_set[:,(-1)]                        
    Y_array=Y_test
    for i in range(times_rspace):
        subid=rand.choice(len(train_X[0]),2,replace=False)        
        subsp_train=train_X[:,subid]
        dec_tree.fit(subsp_train,train_Y)
        subsp_test=X_test[:,subid]
        Y_array=np.column_stack((Y_array,dec_tree.predict(subsp_test)))
    
    Y_pred=np.median(Y_array[:,1:],axis=1)
    return Y_pred



m=10000
ns = [50,100,500,1000,5000,10000,50000]
logns = [mat.log(x,10) for x in ns]    
ds = [3, 5, 10, 20, 30]
ks=[2,5,10,20,50,100,500]
num_subsp=100
j=0
for k in ks:

    scores_list = []

    for d in ds:
        
        accuracy = np.zeros(len(ns))        
        X_test,Y_test=gen_data(m,d)
        test_set=np.column_stack((X_test,Y_test))
                
        for i in range(len(ns)):
            
            n=ns[i]    
            X_train,Y_train=gen_data(n,d)
            train_set=np.column_stack((X_train,Y_train))
            Y_pred = rsubspace_dec_tree(train_set,test_set,k,num_subsp)
            accuracy[i] = 1-sum(abs(Y_test - Y_pred))/m
            
        scores_list.append(accuracy)
    
    plt.figure(j)
    j+=1
    for z in range(len(ds)):
    
        plt.plot(logns, scores_list[z], label = 'D = %s'%ds[z])
    
    plt.xlabel('log(N)')
    plt.ylabel('accuracy')
    plt.legend(loc="upperright")   
    plt.title('Acc. of the dec. tree with K='+str(k)+' terminal nodes')
    
    pylab.savefig('/home/roger/Desktop/BGSE/14D005 Machine Learning/problemsets/Problemset4/subsp_acc_k'+str(k)+'.pdf')
    

