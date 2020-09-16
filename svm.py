# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 04:35:15 2020

@author: MOSHOOD OSENI
"""

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df= pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace=True)
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)