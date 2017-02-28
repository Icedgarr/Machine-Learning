# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:48:17 2017

@author: roger
"""


import sklearn as sk
import numpy as np
from numpy import random as rand
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math as mat



def gen_data(n,d):
    X=rand.uniform(low=-2**(1/d),high=2**(1/d),size=(n,d))
    
    Y=[2]*n
    for i in range(0,n):
        if any(abs(n)>1 for n in X[i])==True:
            Y[i]=0
        else:
            Y[i]=1
    return(X,Y)


def cube_classifier(X,Y):
    data=list(zip(X,Y))
    data=pd.DataFrame(data)
    data.columns=['X','Y']
    
    data1=data[data['Y']==1]
    a=np.amax(list(abs(data1['X'])))
    return(a)



def rectangle_classifier(X,Y):
    data=np.column_stack([X,Y])
    data1=data[data[:,d]==1]

    a=np.amin(data1[:,0:d],axis=0)
    b=np.amax(data1[:,0:d],axis=0)
    return(list(zip(a,b)))


def check_classcube(X_test,Y_test,class_cube):
    m=len(X_test)    
    Y_emp_cube=[]
    for i in range(0,m):
        if any(abs(n)>class_cube for n in X_test[i])==True:
            Y_emp_cube.append(0)
        else:
            Y_emp_cube.append(1)
    
    check_cube=np.array(Y_test)!=np.array(Y_emp_cube)
    wrong_cube=sum(check_cube)
    freq_cube=wrong_cube/m
    return(freq_cube)
    
def check_classrect(X_test,Y_test,class_rectangle):    
    m=len(X_test)  
    Y_emp_rect=[]
    for i in range(0,m):
        if any((X_test[i][j]<class_rectangle[j][0] or X_test[i][j]>class_rectangle[j][1]) 
                for j in range(0,d))==True:
            Y_emp_rect.append(0)
        else:
            Y_emp_rect.append(1)
        
    check_rect=np.array(Y_test)!=np.array(Y_emp_rect)
    wrong_rect=sum(check_rect)
    freq_rect=wrong_rect/m

    return(freq_rect)


m=100000

for d in [3,5,10,50,100]:

    s=[d]*6
    n=[10,50,100,1000,5000,10000]
    lenn=len(n)    
    check_cube=[0]*lenn
    check_rect=[0]*lenn
    X_test, Y_test=gen_data(m,d)    
    
    for i in range(0,10):
        
        train=list(map(gen_data,n,s))
        X_train=[x[0] for x in train]
        Y_train=[x[1] for x in train]
        class_cube=list(map(cube_classifier,X_train,Y_train))
        class_rect=list(map(rectangle_classifier,X_train,Y_train))
        aux_cube=list(map(check_classcube,[X_test]*lenn,[Y_test]*lenn,class_cube))
        check_cube=list(map(np.add,check_cube,aux_cube))
        aux_rect=list(map(check_classrect,[X_test]*lenn,[Y_test]*lenn,class_rect))
        check_rect=list(map(np.add,check_rect,aux_rect))
    
    check_cube=list(map(np.divide,check_cube,[10]*lenn))
    check_rect=list(map(np.divide,check_rect,[10]*lenn))
    #plt.subplot(3,2,i)    
    plt.figure(0)    
    plt.plot(list(map(mat.log,n,[10]*lenn)), check_cube)
    plt.figure(1)    
    plt.plot(list(map(mat.log,n,[10]*lenn)), check_rect)
    #plt.ylim([0.15,0.5])

plt.figure(0)
plt.legend(['d=3','d=5', 'd=10', 'd=50', 'd=100' ],loc='upper right')
plt.xlabel('log(n)')
plt.ylabel('Risk')
plt.title("Cube classifier's Risk vs log n train points")


plt.figure(1)
plt.legend(['d=3','d=5', 'd=10', 'd=50', 'd=100', 'd=1000' ],loc='upper right')
plt.xlabel('log(n)')
plt.ylabel('Risk')
plt.title("Rectangle classifier's Risk vs log n train points")



