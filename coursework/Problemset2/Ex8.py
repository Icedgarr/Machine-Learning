# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 23:22:30 2017

@author: roger
"""

import sklearn
import math as mat
import numpy as np
from numpy import random as rand

n=10000

d=5


#Generate X of uniform distribution
X=rand.uniform(low=0,high=1,size=(n,d))

#Generates Y from X as a Bernoulli with p=x[0] in each case
p=X[:,0]
Y=rand.binomial([1]*len(p),p,size=len(p))

#Zip data
data=list(zip(X,Y))





