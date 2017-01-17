# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 13:48:48 2017

@author: roger
Ridge and Lasso Regression example using scikit learn
"""

import sklearn

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#import the dataset of diabetes inside sklearn package
diabetes=datasets.load_diabetes()

diabetes.target
diabetes.data

diabetes_X=diabetes.data
diabetes_y=diabetes.target

#split into train and test samples, fit the model on training data and predict test data
diabetes_X_train=diabetes_X[:-200]
diabetes_X_test=diabetes_X[-200:]

diabetes_y_train=diabetes_y[:-200]
diabetes_y_test=diabetes_y[-200:]

regr.fit(diabetes_X_train,diabetes_y_train)

print('Coefficients: \n', regr.coef_)
#coef= [ -53.74632032 -240.71698284  514.94482034  240.30473842 -859.49807984
#       431.46764515  222.19158067  271.02036832  785.38147559  157.83390623]
print("Mean squared error %.2f" % np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
#MSE=2908.62

#alpha is the lambda parameter of Ridge
rreg=Ridge(alpha=0.001)

rreg.fit(diabetes_X_train,diabetes_y_train)

print("Coefficients: \n" , rreg.coef_)
#ceof=[ -52.54693017 -240.63732987  515.38839799  240.48130998 -687.47539796
#       292.32721937  145.81437495  253.83816834  723.7304183   157.66918051]
#there is shrinkage in some of the coefficients
print("Mean squared error: %.2f"%np.mean((rreg.predict(diabetes_X_test)-diabetes_y_test)**2))
#MSE=2907.67 (lower than LR)


lreg=Lasso(alpha=0.1)

lreg.fit(diabetes_X_train,diabetes_y_train)

print('Coefficients: \n', lreg.coef_)
#coef=[  -0.         -168.88552677  491.49817854  177.79443884 -123.08230347
#     -0.         -165.70780148    0.          552.96649892  121.54425931]
#some of the coefficients went to 0.
print("Mean squared error: %.2f"%np.mean((lreg.predict(diabetes_X_test)-diabetes_y_test)**2))
#MSE=2904.05 (even better)

#Usually we would test for different alphas in order to get the best lambda according
# to a certain test (like BIC, AIC, MSE..).




