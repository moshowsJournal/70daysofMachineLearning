# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:26:44 2020

@author: Folio 1040 PC
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import random
import pandas as pd

#dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

#euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot2[1] - plot2[1])**2)

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance =np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance,group])
        
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
            
    return vote_result,confidence

accuracies = []
for i in range(25):
    
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    test_size = 0.2
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    correct = 0
    total= 0
    for group in test_set:
        for data in test_set:
            for data in test_set[group]:
                vote,confidence = k_nearest_neighbors(train_set,data,k=5)
                if group == vote:
                    correct += 1
                else:
                    #print(confidence)
                total += 1
    #print('Accuracy: ', correct/total)
    accuracies.append(correct/total)
    
print(sum(accuracies)/len(accuracies))
    
#result = k_nearest_neighbors(dataset,new_features,k=3)
#print(result)

