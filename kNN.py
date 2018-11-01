#########################################################################
#!/usr/bin/env
# -*- coding: utf-8 -*-
# File Name: kNN.py
# Created on : 2018-10-14 16:25:08
# Author: Lin Zheng
# Author Email: zlnn520@126.com
# Last Modified: 2018-10-14 16:29:20
#########################################################################


from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

