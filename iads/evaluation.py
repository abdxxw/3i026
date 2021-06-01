# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: evaluation.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------

# ------------------------ A COMPLETER :
def crossvalidation(C, DS, m=10):
    """ Classifieur * tuple[array, array] * int -> tuple[tuple[float,float], tuple[float,float]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    data, label = DS
    T_data = []
    T_label = []
    accA = []
    accT = []
    n = label.size
    for _ in range(0,m):
        
        temp_data = []
        temp_label = []
        for _ in range(0, n//m):
            j = np.random.randint(0,label.size)
            temp_data.append(data[j])
            temp_label.append(label[j])
            data = np.delete(data,j,0) 
            label = np.delete(label,j,0)
        T_data.append(temp_data)
        T_label.append(temp_label)
    
    T_data = np.asarray(T_data)
    T_label = np.asarray(T_label)
    for i in range(0,m):
        C.reset()
        temp_train_data = np.asarray([1]*data.shape[1])
        temp_train_label = np.asarray([])
        for k in range(0,m):
            if(k != i):
                temp_train_data = np.vstack((temp_train_data,T_data[k]))
                temp_train_label = np.concatenate([temp_train_label,T_label[k]])
        temp_train_data = np.delete(temp_train_data,0,0)
        C.train(temp_train_data,temp_train_label)
        accA.append(C.accuracy(temp_train_data,temp_train_label))
        accT.append(C.accuracy(np.asarray(T_data[i]),np.asarray(T_label[i])))
    accA = np.asarray(accA)
    accT = np.asarray(accT)
    return (accA.mean(),accA.std()),(accT.mean(),accT.std())

