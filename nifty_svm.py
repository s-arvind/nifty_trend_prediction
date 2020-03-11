from sklearn import svm
import pandas as pd
import numpy as np
import math
from pandas.tools.plotting import autocorrelation_plot
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

feature = pd.read_csv("feature.csv", index_col=0, header=0)
index = len(feature.index)-1
for i in range(len(feature.index)-1,0,-1):
	if '2018' in feature.index[i] or '2017' in feature.index[i]: 
		index = i
	else:
		break

test_tocsv = feature[index:]
train_tocsv = feature[:index]

train_tocsv.to_csv('train.csv')
test_tocsv.to_csv('test.csv')

train = pd.read_csv("train.csv", index_col=0, header=0)
test = pd.read_csv("test.csv", index_col=0, header=0)

y_train = train['target'].as_matrix()
y_test = test['target'].as_matrix()
x_train = train.drop('target',axis=1).as_matrix()
x_test = test.drop('target',axis=1).as_matrix()

#####################					Optimization parameters for  SVM 				######################

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,1e-4],
                     'C': [0.1, 1, 10, 100,1000,10000]},
                    {'kernel': ['linear'],'C': [0.1, 1, 10, 100,1000,10000]},
                    {'kernel': ['poly'],'C': [0.1, 1, 10, 100,1000,10000],'degree':[1,2,3]},
                    {'kernel': ['sigmoid'],'C': [0.1, 1, 10, 100,1000,10000]}]

clf = GridSearchCV(svm.SVC(),tuned_parameters,cv=5,scoring = 'accuracy')								# Grid search to optimize SVM (multi class classification for classes 1,-1 and 0
clf.fit(x_train,y_train)
print "Best Params: ", clf.best_params_
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))						

finer_tuning = [{'kernel':['sigmoid'], 'C':[9500,9800,10000,10200,10500]}]
clf_fine = GridSearchCV(svm.SVC(),finer_tuning,cv=5,scoring = 'accuracy')								# Grid search to optimize SVM (multi class classification for classes 1,-1 and 0
clf_fine.fit(x_train,y_train)
print "Best Params: ", clf_fine.best_params_
means = clf_fine.cv_results_['mean_test_score']
stds = clf_fine.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf_fine.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print "classification_report :"		
y_true, y_pred = y_test, clf_fine.predict(x_test)											# Prediction of nifty trend 
print(classification_report(y_true, y_pred))											# classification report / confusion matrix
print (y_pred)
print (y_true)
pd.DataFrame(data=y_true, index=test.index).to_csv('y_true.csv')
pd.DataFrame(data=y_pred, index=test.index).to_csv('y_pred.csv')
print "accurracy : " + str(accuracy_score(y_true, y_pred) * 100) + "%"					# accuracy of algorithm on test data

