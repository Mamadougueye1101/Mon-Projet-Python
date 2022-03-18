# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:33:14 2021

@author: Mamadou GUEYE
Projet Python
M1 MAS
"""
import data_etu as de
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=de.generate_data(11509372)

#Pour afficher toutes les colonnes
pd.set_option("display.max.columns", None)

#print(data.info())
#print(data.columns)
#print(data.index)
#print(type(data))
data.shape[0] #nb de lignes
#print(data.shape) #dataframe de n lignes et p colonnes
#print(data.mean()) #calcule les moyennes de toutes ?
#print(data['Y_0'].mean()) #vérification
#print(data['X'].mean())

#Moyenne sur toutes les colonnes
moy=data.mean()

#print(data.mean()>=0.05)
#print(moy)
#moy.index[moy>=0.05]   renvoie Y_9
#indice =data.columns[[data.mean()>=0.05]]

#sort le num de colonne dont la moyenne > 0.05
indice = moy.index[moy>=0.05][0]
#Yk de type serie et plus dataframe
#Yk=data[indice]
# en faisant indice = moy.index[moy>=0.05]

data.describe() #sumamary de R

Yk=data.loc[:,moy>0.05]
#Yk=np.array(Yk)
def moyenne(x):
    Sum=0.
    for i in range(len(x)):
        Sum=Sum+x.iloc[i]
    return Sum/len(x)

#m=moyenne(Yk)

def ecart_type(x) :
    sd=np.sqrt(moyenne((x-moyenne(x))**2))
    return sd;


#.index pour une liste
#p=list(Yk).index(max(Yk))

def stat(x):
    m = moyenne(x)
    sd=ecart_type(x)
    #p=list(x).index(max(x))
    a=data.loc[data[x.name]==max(x),'X']
    data.plot( x='X', y=x.name)
    return m,sd,a

s=stat(Yk)

def monteCarlo(x,n):
    alea=np.random.randint(0, 11509372, n)
    s=0.
    for i in alea:
        s=s+2*x[i]/n
    return s

n=1000000
mC=monteCarlo(Yk, n)
mC
mC-s[0]
data['Z']=Yk/mC
data
Z=data['Z']
z=stat(Z)
def monteCarlo2(x,n):
    alea=np.random.uniform(-1,1, n)
    s=0.
    for i in alea:
        s=s+2*data.loc[data['X']==i,Yk.name]/n
    return s
mC2=monteCarlo2(Yk, n)
