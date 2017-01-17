# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:10:05 2017

@author: roger
Logistic Regression using scikit learn
"""

import sklearn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#load digits data from the datasets
digits=load_digits()

images_and_labels=list(zip(digits.images,digits.target))
for index,(image,label) in enumerate(images_and_labels[:9]):
    pyplot.subplot(2,9, index+1)
    pyplot.axis('off')
    pyplot.imshow(image,cmap=pyplot.cm.gray_r)
    pyplot.title("Training: %i" %label)

X=digits.data
Y=digits.target

#C is for the regularizations that go in front of the loss function and alpha go in
#front of the regularizer. They are like inverse. (High C is like low alpha)
logistic_classifier=linear_model.LogisticRegression(C=100.0)

logistic_classifier.fit(X,Y)

print("Logistic regression for character recognition: \n%s\n" % metrics.classification_report(Y,logistic_classifier.predict(X)))
#the precision tells us how many of the classified as 1 are real 1. The recall tells
#us how many of the real 1 are classified as 1.



#cross validation takes randomly folds with the same representation of each variables
#in each fold.
scores=cross_val_score(logistic_classifier,X,Y,cv=5)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


