# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:15:20 2023

@author: Emmanuel Maseruka
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


data=pd.read_excel("C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/overtrade_ml.xlsx")
data=data.dropna()
x=data[[ 'gru_ml', 'ltsm_ml', 'average','tcn_ml']]
y=data['actual_ml2']

# Splitting train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


#SVM
from sklearn.svm import SVC #support vector classifier
hypothesis_svm = SVC(kernel='rbf', random_state=101)  #RBF is the default kernel used within the sklearn's SVM classification algorithm radial basis kernel
# Cross validation
svm_scores = cross_val_score(hypothesis_svm, x_train, y_train, cv=10, scoring='accuracy')
print ("SVC with rbf kernel -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(svm_scores), np.std(svm_scores)))



#Random forest

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=101).fit(x_train, y_train)
rf_scores = cross_val_score(RF, x_train, y_train, cv=10, scoring= 'accuracy')
print ("Random forest -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(rf_scores), np.std(rf_scores)))



from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

BG = BaggingClassifier(KNeighborsClassifier(3)).fit(x_train, y_train)
BG_scores = cross_val_score(BG, x_train, y_train, cv=10, scoring= 'accuracy')
print ("Bagging -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(BG_scores), np.std(BG_scores)))



from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(max_depth = 1, n_estimators = 3, random_state = 101).fit(x_train, y_train)
GB_scores = cross_val_score(GB, x_train, y_train, cv=10, scoring= 'accuracy')
print ("Gradient boosting-> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(GB_scores), np.std(GB_scores)))


#best algo is simple knn neighbour
from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=3).fit(x_train, y_train)  # You can adjust the number of neighbors
knn_scores =cross_val_score(knn_regressor, x_train, y_train, cv=10, scoring= 'accuracy')
print ("Bagging -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(knn_scores), np.std(knn_scores)))














































