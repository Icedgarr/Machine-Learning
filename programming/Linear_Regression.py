# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:49:59 2017

@author: roger
Linear Regression example using scikit learn
"""
import sklearn


from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the dataset of diabetes inside sklearn package
diabetes=datasets.load_diabetes()

diabetes.target
diabetes.data

diabetes_X=diabetes.data
diabetes_y=diabetes.target


#Generate the object regr of the class LinearRegression.
regr=linear_model.LinearRegression()

#regr now is an object that have atributes and functions embeded on it. We can call
#functions as fit, predict, score, etc. When we do the fit it stores the results 
#regarding that fit, overwritting the previous fit.

regr.fit(diabetes_X,diabetes_y)
#print the relevant features of the fit
print('Coefficients: \n', regr.coef_)
#MSE
print("Mean squared error %.2f" % np.mean((regr.predict(diabetes_X)-diabetes_y)**2))

#score is the R^2
print('Variance score %.2f' % regr.score(diabetes_X,diabetes_y))

#split into train and test samples and redo the fit
diabetes_X_train=diabetes_X[:-200]
diabetes_X_test=diabetes_X[-200:]

diabetes_y_train=diabetes_y[:-200]
diabetes_y_test=diabetes_y[-200:]


regr.fit(diabetes_X_train,diabetes_y_train)

print('Coefficients: \n', regr.coef_)

print("Mean squared error %.2f" % np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))

print('Variance score %.2f' % regr.score(diabetes_X_test,diabetes_y_test))

#select only the 3rd column to plot it
diabetes_X1=np.transpose(np.asmatrix(diabetes.data[:np.newaxis,2]))
#fit the model
regr.fit(diabetes_X1,diabetes_y)
#plot the points
plt.scatter(diabetes_X1,diabetes_y,color='black')
#plot the regression given by X1
plt.plot(diabetes_X1,regr.predict(diabetes_X1),color='blue')





