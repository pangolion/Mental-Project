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
data = pd.read_csv(r"C:\Users\mamba\Desktop\4741\Project\data_year.csv")
stat = data["year"].value_counts()
dataset = data.loc[data['year'] >= 2017]
dataset = dataset.dropna(axis="columns",how="all")
x = []
y = []
for i in range(0,len(dataset)):
    for j in range(0,77):
        item = dataset.iloc[i,j]
        if item ==-1:
            x.append(i)
            y.append(j)
missing_index = pd.DataFrame()
missing_index["x"] = x
missing_index["y"] = y
missing_value_stat = missing_index["y"].value_counts()
dataset = dataset.drop(columns = [dataset.iloc[:,71].name])
dataset = dataset.drop(columns = [dataset.iloc[:,67].name])
dataset = dataset.drop(columns = [dataset.iloc[:,66].name])
dataset = dataset.drop(columns = [dataset.iloc[:,20].name])
feature = pd.read_csv(r"C:\Users\mamba\Desktop\4741\Project\feature.csv")
numerical = []
boolean = []
highmissing = []
text = []
label = []
ordinal = []

for i in range(0, len(feature)):
    f = feature.iloc[i]
    if f["Unnamed: 4"] == "high missing":
       highmissing.append(i)
    elif f["Unnamed: 4"] == "delet":
        highmissing.append(i)
highmissing.append(36)
highmissing.append(37)
    
for i in range(0, len(feature)):
    f = feature.iloc[i]
    if f["type"] == "numerical":
        if i not in highmissing:
            numerical.append(i)
    elif f["type"] == "boolean":
        
        if i not in highmissing:
            boolean.append(i)
    elif f["type"] == "ordinal":
        if i not in highmissing:
            ordinal.append(i)
    elif f["type"] == "label":
        if i not in highmissing:
            label.append(i)
value =  pd.DataFrame()
bo6 = []
boo_6  = dataset.iloc[:,6]
stat_6 = boo_6.value_counts()
for i in range(0,len(boo_6)):
    b = boo_6.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    else:
        r = 0
    bo6.append(r)
value["bo6"] = bo6   

bo11 = []
boo11  = dataset.iloc[:,11]
stat_11 = boo11.value_counts()
y_11 = 439/(len(boo11)) 
n_11 = 64/(len(boo11))
el_11 =804/(len(boo11))
for i in range(0,len(boo11)):
    b = boo11.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <n_11:
            r = -1
        elif t>n_11 and t<y_11:
            r = 1
        else:
            r = 0     
    else:
        r = 0
    bo11.append(r)
value["bo11"] = bo11   

bo12 = []
boo_12  = dataset.iloc[:,12]
stat_12 = boo_12.value_counts()
for i in range(0,len(boo_12)):
    b = boo_12.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    else:
        r = 0
    bo12.append(r)
value["bo12"] = bo12  

bo14 = []
boo_14  = dataset.iloc[:,14]
stat_14 = boo_14.value_counts()
y_14 = 561/(len(boo11)) 
n_14 = 616/(len(boo11))
for i in range(0,len(boo11)):
    b = boo_14.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t<y_14:
            r = 1
        else:
            r = -1    
    bo14.append(r)
value["bo14"] = bo14 

  
bo15 = []
boo_15  = dataset.iloc[:,15]
stat_15 = boo_15.value_counts()
y_15 = 369/(len(boo11)) 
n_15 = 812/(len(boo11))
el_15 =126/(len(boo11))
for i in range(0,len(boo_15)):
    b = boo_15.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <el_15:
            r = 0
        elif t>el_15 and t<y_15:
            r = 1
        else:
            r = -1     
    else:
        r = 0
    bo15.append(r)
value["bo15"] = bo15

bo16 = []
boo_16  = dataset.iloc[:,16]
stat_16 = boo_16.value_counts()
y_16 = 406/(len(boo11)) 
n_16 = 553/(len(boo11))
el_16 =348/(len(boo11))
for i in range(0,len(boo_15)):
    b = boo_16.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <el_16:
            r = 0
        elif t>el_16 and t<y_16:
            r = 1
        else:
            r = -1     
    else:
        r = 0
    bo16.append(r)
value["bo16"] = bo16

bo18 = []
boo_18  = dataset.iloc[:,18]
stat_18 = boo_18.value_counts()
y_18 = 407/(len(boo11)) 
n_18 = 326/(len(boo11))
el_18 =574/(len(boo11))
for i in range(0,len(boo11)):
    b = boo_18.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <n_18:
            r = -1
        elif t>n_18 and t<y_18:
            r = 1
        else:
            r = 0     
    else:
        r = 0
    bo18.append(r)
value["bo18"] = bo18   

bo19 = []
boo_19  = dataset.iloc[:,19]
stat_19 = boo_19.value_counts()
y_19 = 504/(len(boo11)) 
n_19 = 371/(len(boo11))
el_19 =432/(len(boo11))
for i in range(0,len(boo_15)):
    b = boo_19.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <n_19:
            r = -1
        elif t>n_19 and t<el_19:
            r = 0
        else:
            r = 1     
    else:
        r = 0
    bo19.append(r)
value["bo19"] = bo19

bo28 = []
boo_28  = dataset.iloc[:,28]
stat_28 = boo_28.value_counts()
for i in range(0,len(boo_6)):
    b = boo_28.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    else:
        r = 0
    bo28.append(r)
value["bo28"] = bo28  

 
bo30 = []
boo_30  = dataset.iloc[:,30]
stat_30 = boo_30.value_counts()
y_30 = 236/(len(boo11)) 
n_30 = 734/(len(boo11))
el_30 =228/(len(boo11))
for i in range(0,len(boo_15)):
    b = boo_30.iloc[i]
    if b == "Yes":
        r = 1
    elif b == "No":
        r = -1
    elif b == "-1":
        t = random.random()
        if t <el_30:
            r = 0
        elif t>el_30 and t<y_30:
            r = 1
        else:
            r = -1     
    else:
        r = 0
    bo30.append(r)
value["bo30"] = bo30

for i in label:
    column = dataset.iloc[:,i]
    exec("c_%d = pd.get_dummies(column)"%i)
    exec("c_%d = c_%d.values"%(i,i))
    exec("c_%d = pd.DataFrame(c_%d)"%(i,i))
    exec("value = pd.concat([value, c_%d],axis=1)"%i)

bo2 =[]
for i in range(0,len(dataset)):
    g = dataset.iloc[i,2]
    if g =="Female":
        r = -1
    elif g =="female":
        r = -1
    elif g=="male":
        r = 1
    elif g == "Male":
        r = 1
    
    else:
        r = 0
    bo2.append(r)
value["bo2"] = bo2
lob = pd.read_csv(r"C:\Users\mamba\Desktop\4741\Project\LOB.csv")
lob["bo2"] = bo2

for i in numerical:
    if (i != 7 & i !=0):
        sum_numerical = 0
        count = 0
        for j in range(1524):
            if dataset.iloc[j,i] != -1:
                sum_numerical += dataset.iloc[j,i]
                count += 1
        numerical_mean = round(sum_numerical/count)
        for k in range(1524):
            if dataset.iloc[j,i] == -1:
                dataset.iloc[j,i] = numerical_mean
 
for i in numerical:
    if i == 0:
        pass
    elif i == 7:
        pass
    else:
        da = dataset.iloc[:,i].values
        dt = pd.DataFrame(da)
        lob = pd.concat([lob, dt],axis = 1)

data = pd.read_csv(r"C:\Users\mamba\Desktop\4741\Project\Text_data.csv")
data = pd.concat([lob, data],axis = 1)
data = data.drop(['Unnamed: 0'],axis=1)
response = dataset.iloc[:,7].values
data["response"] = response

#data.to_csv(r"C:\Users\mamba\Desktop\4741\Project\final_data.csv")
# 开始
data.sample(frac=1)
num = int(1524*0.8)
train = data.iloc[:num,:]
test = data.iloc[num:,:]
y_train = train["response"]
y_test = test["response"]
x_train = train.iloc[:,0:-1]
x_test = test.iloc[:,0:-1]

# cross-validation
    # logistic regression
l1_solver = ['liblinear', 'saga']
l2_solver = ['lbfgs', 'saga','newton-cg']
# elasticnet penalty 只能用 saga solver
elasticnet_solver = np.random.random(5)

C_lr = np.logspace(-2,2,5)
l1 = np.zeros((len(l1_solver),len(C_lr)))
l2 = np.zeros((len(l2_solver),len(C_lr)))
elasticnet = np.zeros((len(elasticnet_solver),len(C_lr)))

for i in l1_solver:
    for j in C_lr:
        lr_clf_l1 = linear_model.LogisticRegression(penalty = 'l1', solver = i)
        scores_l1 = cross_val_score(lr_clf_l1, normalized_X, y, cv = 5)
        scores_l1 = np.mean(scores_l1)
        l1[i,j] = scores_l1
    
for i in l2_solver:
    for j in C_lr:
        lr_clf_l2 = linear_model.LogisticRegression(penalty = 'l2', solver = i)
        scores_l2 = cross_val_score(lr_clf_l2, normalized_X, y, cv = 5)
        scores_l2 = np.mean(scores_l2)
        l2[i,j] = scores_l2
    
for i in elasticnet_solver:
    for j in C_lr:
        lr_clf_elasticnet = linear_model.LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = i)
        scores_elasticnet = cross_val_score(lr_clf_elasticnet, normalized_X, y, cv = 5)
        scores_elasticnet = np.mean(scores_elasticnet)
        elasticnet[i,j] = scores_elasticnet
    
    # decision tree
        #hyper-parameter: max-depth, criterion
num_of_features = len(x_train[1,:])
tree_criterion = ['gini','entropy']
scores_tree = np.zeros((2,num_of_features - 3))
for i in range(len(tree_criterion))
    for j in range(3,num_of_features)):
        dt_clf = tree.DecisionTreeClassifier(criterion = tree_criterion[i],max_depth = j)
        scores = cross_val_score(dt_clf, x_train, y_train, cv = 5)
        scores = np.mean(scores)
        scores_tree[i,j] = scores
print(scores_tree)

    # SVM
        #hyperparameter: kernel type, C, gamma
        #先对 kernel type + C 选参数
kernel_type = ['linear','poly','rbf','sigmoid']
C_range = np.logspace(-2, 2, 5)
kernel_matrix = np.zeros((len(kernel_type), len(C_range)))
for i in range(len(kernel_type)):
    for j in range(len(C_range)):
        svm_clf = svm.SVC(kernel = kernel_type[i], C = C_range[j])
        scores = cross_val_score(svm_clf, normalized_X, y, cv = 5)
        scores = np.mean(scores)
        kernel_matrix[i,j] = scores
print(kernel_matrix)
   
        #对 rbf kernel选参数 C + gamma
C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-9,3,13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    # KNN





#fit model

#logistic
log_model = LogisticRegression().fit(x_train,y_train)
print("the score of logistic model is %f" %log_model.score(x_test, y_test))

#knn
KNN = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
print("the score of knn is %f" %KNN.score(x_test, y_test))

# SVM
SVM = svm.SVC().fit(x_train,y_train)
print("the score of SVM is %f" %SVM.score(x_test, y_test))

#tree
decision_tree = tree.DecisionTreeClassifier().fit(x_train,y_train)
print("the score of decision tree is %f" %decision_tree.score(x_test, y_test))
