# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:10:05 2017

@author: roger
Logistic Regression using scikit learn
"""

import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import arff
import scipy

from sklearn import datasets, linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.metrics import confusion_matrix

#load digits data from the datasets
digits=load_digits()

images_and_labels=list(zip(digits.images,digits.target))
for index,(image,label) in enumerate(images_and_labels[:9]):
    plt.subplot(2,9, index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r)
    plt.title("Training: %i" %label)

X=digits.data
Y=digits.target
#we split into train and test data. Fit on the train part and predict the test part.
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.2,random_state=0)


#C is for the regularizations that go in front of the loss function and alpha go in
#front of the regularizer. They are like inverse. (High C is like low alpha)
logistic_classifier=linear_model.LogisticRegression(C=100.0)

logistic_classifier.fit(X_train,Y_train)

print("Logistic regression for character recognition: \n%s\n" % metrics.classification_report(Y_test,logistic_classifier.predict(X_test)))
#the precision tells us how many of the classified as 1 are real 1. The recall tells
#us how many of the real 1 are classified as 1.



#cross validation takes randomly folds with the same representation of each variables
#in each fold.
scores=cross_val_score(logistic_classifier,X,Y,cv=5)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

#Application to phishing data,

with open('/home/roger/Desktop/BGSE/14D005 Machine Learning/Machine-Learning/programming/Training Dataset.arff') as fh:
          phisingdata=arff.load(fh)

phising=np.array(phisingdata['data'])

phising.shape

phising=phising.astype(int)

target=phising[:,30]
phising=phising[:,0:29]

#random_state is the seed.
phising_train,phising_test,target_train,target_test=train_test_split(phising,target,test_size=0.2,random_state=0)

logistic_binary=linear_model.LogisticRegression(C=100.0)

logistic_binary.fit(phising_train,target_train)


fpr,tpr,thresholds=metrics.roc_curve(target_test,logistic_binary.predict(phising_test),pos_label=1)

auc=metrics.auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color="red",lw=2,label="ROC curve (area=%0.2f)" %auc)
plt.plot([0,1],[0,1],color="navy",lw=2,linestyle='--')

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

plt.legend(loc='lower right')

mat=confusion_matrix(target_test,logistic_binary.predict(phising_test))

sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=['-1','1'],yticklabels=['-1','1'])





