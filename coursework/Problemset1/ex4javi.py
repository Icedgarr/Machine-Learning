# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:35:07 2017

@author: javi
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt


random.seed(333)
D=10
d=2

basis = np.identity(D) #We generate an identity matrix to obtain the basis vectors

W = np.random.normal(0, (1/d), (d,D)) #Generate a random matrix
W


proj_mat = np.dot(W, basis) #Project the basis vectors

proj_mat[0,:] = (proj_mat[0,:] - np.mean(proj_mat[0,:]))/np.std(proj_mat[0,:]) #Rescale the vectors
proj_mat[1,:] = (proj_mat[1,:] - np.mean(proj_mat[1,:]))/np.std(proj_mat[1,:])
proj_mat


RAND_1 = np.random.normal(loc=0, scale=1, size=(2, 10)) #Compare with multiple matrices
RAND_2 = np.random.normal(loc=0, scale=1, size=(2, 100))
RAND_3 = np.random.normal(loc=0, scale=1, size=(2, 1000))
RAND_4 = np.random.normal(loc=0, scale=1, size=(2, 10000))
RAND_5 = np.random.normal(loc=0, scale=1, size=(2, 100000))


plt.subplot(3,2,1)
plt.scatter(x=proj_mat[0,:], y=proj_mat[1,:])
plt.title("Projected basis N:10")
plt.subplot(3,2,2)
plt.scatter(x=RAND_1[0,:], y=RAND_1[1,:])
plt.title("Random matrix N:10")
plt.subplot(3,2,3)
plt.scatter(x=RAND_2[0,:], y=RAND_2[1,:])
plt.title("Random matrix N:100")
plt.subplot(3,2,4)
plt.scatter(x=RAND_3[0,:], y=RAND_3[1,:])
plt.title("Random matrix N:1000")
plt.subplot(3,2,5)
plt.scatter(x=RAND_4[0,:], y=RAND_4[1,:])
plt.title("Random matrix N:10000")
plt.subplot(3,2,6)
plt.scatter(x=RAND_5[0,:], y=RAND_5[1,:])
plt.title("Random matrix N:100000")

plt.tight_layout()

### Hypercubes
n=2

for i in range(1,7):
    
    vmax=[1]*(n*i)
    vmin=[-1]*(n*i)

    hc=np.transpose(np.asarray(list(itt.product(*zip(vmin,vmax)))))
    W2=np.random.normal(0,(1/d),(2,(n*i)))
    proj_hc = np.dot(W2,hc)
    proj_hc[0,:] = (proj_hc[0,:] - np.mean(proj_hc[0,:]))/np.std(proj_hc[0,:]) #Rescale the vectors
    proj_hc[1,:] = (proj_hc[1,:] - np.mean(proj_hc[1,:]))/np.std(proj_hc[1,:])
    plt.subplot(3,2,i)
    plt.scatter(proj_hc[0],proj_hc[1])
    plt.title("R^ %.0f" %(i*n))



