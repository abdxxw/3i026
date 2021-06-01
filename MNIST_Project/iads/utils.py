# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o') # 'o' pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])    
    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
        par défaut: binf vaut -1 et bsup vaut 1
    """
    data_desc = np.random.uniform(binf,bsup,(n,p))
    data_label = np.asarray([-1 for i in range(0,int(n/2))] + [+1 for i in range(0,int(n/2))])
    np.random.shuffle(data_label)
    return data_desc, data_label
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    d_des_pos = np.random.multivariate_normal(positive_center, positive_sigma,nb_points)
    d_des_neg = np.random.multivariate_normal(negative_center, negative_sigma,nb_points)
    
    return np.vstack((d_des_neg,d_des_pos)), np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])

# ------------------------ 
def create_XOR(n, var):
    data_desc1, data_label1 = genere_dataset_gaussian(np.array([0,0]),np.array([[var,0],[0,var]]),np.array([1,0]),np.array([[var,0],[0,var]]),n)
    data_desc2, data_label2 = genere_dataset_gaussian(np.array([1,1]),np.array([[var,0],[0,var]]),np.array([0,1]),np.array([[var,0],[0,var]]),n)
    data_desc1 = np.vstack((data_desc1,data_desc2))
    data_label1 = np.concatenate((data_label1,data_label2))
    return data_desc1, data_label1

    


