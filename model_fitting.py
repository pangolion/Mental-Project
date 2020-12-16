# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:31:48 2020

@author: mamba
"""

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


data = pd.read_csv(r"/Users/caolihe/Desktop/康奈尔/Term 1/ORIE 4741/project/final/project_final_correct.csv")
data = data.drop(['Unnamed: 0'],axis=1)

data['0.1.1.1'] = preprocessing.scale(data['0.1.1.1'])
for i in range(124,174):
    data.iloc[:,i] = preprocessing.scale(data.iloc[:,i])
#shuffle data
data.sample(frac=1)

#normalization for numerical data

    
# seperate train and test set
num = int(1524*0.8)
train = data.iloc[:num,:]
test = data.iloc[num:,:]

y_train = train["response"]
y_test = test["response"]
x_train = train.iloc[:,0:-1]
x_test = test.iloc[:,0:-1]


#fit model
#just use the default value first, 
#later replace with the parameters chosen by cross-validation
#logistic
# C = 0.1, penalty = l1,solver = liblinear
log_model_l1 = LogisticRegression(penalty = 'l1',C=0.1,solver='liblinear',max_iter = 10000).fit(x_train,y_train)
print("the score of logistic model is %f" %log_model_l1.score(x_test, y_test))
# C = 0.01, penalty = l2, solver = newton-cg
log_model_l2_1 = LogisticRegression(penalty = 'l2',C=0.01,solver='newton-cg',max_iter=10000).fit(x_train,y_train)
print("the score of logistic model is %f" %log_model_l2_1.score(x_test, y_test))

# C = 10, penalty = l2, solver = saga
log_model_l2_2 = LogisticRegression(penalty = 'l2',C=10,solver='saga',max_iter=10000).fit(x_train,y_train)
print("the score of logistic model is %f" %log_model_l2_2.score(x_test, y_test))

# C = 0.1, penalty = elasticnet, solver = saga
r = np.random.uniform()
log_model_elasticnet = LogisticRegression(penalty = 'elasticnet',C=0.1,l1_ratio = r,solver='saga',max_iter = 10000).fit(x_train,y_train)
print("the score of logistic model is %f" %log_model_elasticnet.score(x_test, y_test))

#knn,n_neighbors=69,p=1
KNN_1 = KNeighborsClassifier(n_neighbors=69,p=1).fit(x_train,y_train)
print("the score of knn is %f" %KNN_1.score(x_test, y_test))

#knn,n_neighbors=50,p=2
KNN_2 = KNeighborsClassifier(n_neighbors=50,p=2).fit(x_train,y_train)
print("the score of knn is %f" %KNN_2.score(x_test, y_test))

# SVM, kernel = rbf, C =10 
SVM_rbf = svm.SVC(kernel = 'rbf',C=10).fit(x_train,y_train)
print("the score of SVM is %f" %SVM_rbf.score(x_test, y_test))

# SVM, kernel = poly, C =10/C=1, degree = 3
SVM_poly = svm.SVC(kernel = 'poly',C=1).fit(x_train,y_train)
print("the score of SVM is %f" %SVM_poly.score(x_test, y_test))

#tree,criterion = entropy, max_depth = 13
decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=13).fit(x_train,y_train)
print("the score of decision tree is %f" %decision_tree.score(x_test, y_test))
