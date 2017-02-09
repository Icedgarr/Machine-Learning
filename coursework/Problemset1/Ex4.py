# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:18:36 2017

@author: roger
"""

import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
from numpy import random as rand




def proj_basis(d,D): #Projects the basis of a D-dim space to a d-dim space
    W=rand.normal(0,1/d,(d,D)) #Generate a random matrix to project D-dim vectors to d-dim space 
    basis=np.identity(D) #Generate the basis of a D-dim space
    proj_vect=np.dot(W,basis) #Project the basis
    proj_vect[0]=proj_vect[0]-np.mean(proj_vect[0]) #center first component
    proj_vect[1]=proj_vect[1]-np.mean(proj_vect[1]) #center second component
    std_dev=np.sqrt(np.var(proj_vect[0,])) #compute the std dev of the first component 
    proj_vect=proj_vect/std_dev #rescale by first component
    return proj_vect

d=2
i=0
rng=[10,50,100,150,200,500,1000,10000]
for D in rng: #Plot the proj basis for a rng of dimensions D into a d-dim space
    i=i+1    
    proj_vect=proj_basis(d,D)
    rnd_vect_plane=rand.normal(0,1,(2,D)) #generate random normals
    plt.subplot(4,2,i) #more than one plot
    plt.scatter(proj_vect[0],proj_vect[1]) 
    plt.scatter(rnd_vect_plane[0],rnd_vect_plane[1],color="red")
    plt.title("N=%d"%D) #change title




#hypercube


def proj_hypercube(d,D):

    vmax=[1]*D
    vmin=[-1]*D
    
    hypercube=np.transpose(np.asarray(list(itt.product(*zip(vmin,vmax))))) #generates the vertices
    W=rand.normal(0,1/d,(d,D)) #Generates the projection matrix

    proj_hyp_cube=np.dot(W,hypercube) #Projects

    proj_hyp_cube[0]=proj_hyp_cube[0]-np.mean(proj_hyp_cube[0])
    proj_hyp_cube[1]=proj_hyp_cube[1]-np.mean(proj_hyp_cube[1])
    std_dev=np.sqrt(np.var(proj_hyp_cube[1,]))
    proj_hyp_cube=proj_hyp_cube/std_dev

    return proj_hyp_cube

d=2

rng=[2,3,4,5,6,10]
i=0
for D in rng: #projects the hypercubes from different dimensions to a 2-dim subspace
    i=i+1
    proj_hyp_cube=proj_hypercube(d,D)
    
    plt.subplot(3,2,i) #more than one plot
    plt.scatter(proj_hyp_cube[0],proj_hyp_cube[1])
    plt.title("D=%d"%D) #change title













