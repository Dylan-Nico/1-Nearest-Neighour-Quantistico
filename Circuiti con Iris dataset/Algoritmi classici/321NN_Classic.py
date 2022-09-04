#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import preprocessing
import numpy as np
import statistics
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from numpy import linalg
import matplotlib.pyplot as plt


# In[4]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
 
# Loading data
irisData = pd.read_csv("Iris/iris.data",header=None,names=["f0","f1","f2","f3","class"])


# In[5]:


irisData


# In[7]:


irisData['class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'],
                        [0, 1, 2], inplace=True)


# In[8]:


irisData


# In[9]:


#Standardise
scaler = StandardScaler()
irisData.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(irisData.loc[:,["f0","f1","f2","f3"]])


# In[10]:


#Normalize
irisData.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(irisData.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[11]:


# random_seed : int : Random number generator seed
# ---- Per cambiare porzione del dataset modificare il valore del seed ----
random_seed = 3
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[12]:


#irisData = irisData[1243:]
#irisData


# In[13]:


#setosa = irisData[irisData["class"]==0]
#versicolor = irisData[irisData["class"]==1]
#virginica = irisData[irisData["class"]==2]


# In[14]:


#dataset con 129 righe (128 quando sceglierò il test)
#versicolor = versicolor.iloc[:-11,:]
#virginica = virginica.iloc[:-10,:]


# In[15]:


#data = pd.concat([setosa,versicolor,virginica])
#data


# In[16]:


irisData = irisData.iloc[rgen.permutation(len(irisData.index))].copy()
irisData = irisData[21:]
irisDataCopy = irisData
irisDataCopy


# In[17]:


from scipy import spatial
#define fidelity 
def fidelity(v0,v1):
    return -pow(abs(1 - spatial.distance.cosine(v0, v1)),2)
    #return np.inner(v0,v1)*np.inner(v0,v1)


# In[18]:


tr = []
predictions = []
ground = []
for v in irisData.index:
    p = 0
    inputVector = irisData.loc[v]
    irisData = irisData.drop(v)
    print(inputVector)
    while not irisData.empty:
        #estrazione del sottoinsieme
        subset = irisData.iloc[:32]
        irisData = irisData.drop(subset.index)
        print("Circuito:",p)
        print(subset)
        knn = KNeighborsClassifier(n_neighbors=1,metric=fidelity,algorithm='brute')
        knn.fit(subset.loc[:,"f0":"f3"].values, subset["class"].values)
        #predict
        pred = knn.predict([inputVector["f0":"f3"].values])
        print("Classe predetta:",pred)
        predictions.append(pred)
        ground.append(inputVector["class"])
        p+=1
    irisData = irisDataCopy #ripristino il dataset prima della prossima iterazione


# In[169]:


#acc = accuracy_score(ground, predictions)
#print("Accuratezza",acc)


# In[19]:


#calcolo accuratezza per ogni test --> 129 statistiche
k = 0
l = 4
leng = len(predictions)
lista = []
while leng>0:
        acc = accuracy_score(ground[k:l], predictions[k:l])
        #print("Accuratezza",acc)
        lista.append(acc)
        k+=4
        l+=4
        leng = leng-4
        
        
#calcolo media accuratezza
print("Accuratezza media:",statistics.mean(lista))


# In[ ]:




