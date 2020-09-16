# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:26:44 2020

@author: Folio 1040 PC
"""
from math import sqrt

plot1 = [1,3]
plot2 = [2,5]

euclidean_distance = sqrt((plot1[0] - plot2[1])**2 - (plot2[0] - plot2[1])**2)

print(euclidean_distance)
