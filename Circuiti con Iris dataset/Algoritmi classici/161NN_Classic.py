#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
 
# Loading data
irisData = pd.read_csv("Iris/iris.data",header=None,names=["f0","f1","f2","f3","class"])


# In[7]:


#irisData


# In[8]:


irisData['class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'],
                        [0, 1, 2], inplace=True)


# In[9]:


#irisData


# In[10]:


#Standardise
scaler = StandardScaler()
irisData.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(irisData.loc[:,["f0","f1","f2","f3"]])


# In[11]:


#Normalize
irisData.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(irisData.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[12]:


# random_seed : int : Random number generator seed
# ---- Per cambiare porzione del dataset modificare il valore del seed ----
random_seed = 3
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[13]:


setosa = irisData[irisData["class"]==0]
versicolor = irisData[irisData["class"]==1]
virginica = irisData[irisData["class"]==2]


# In[14]:


#dataset con 129 righe (128 quando sceglierò il test)
versicolor = versicolor.iloc[:-11,:]
virginica = virginica.iloc[:-10,:]


# In[15]:


data = pd.concat([setosa,versicolor,virginica])
#data


# In[16]:


data = data.iloc[rgen.permutation(len(data.index))].copy()
irisDataCopy = data
data


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
for v in data.index:
    p = 0
    inputVector = data.loc[v]
    data = data.drop(v)
    print(inputVector)
    while not data.empty:
        #estrazione del sottoinsieme
        subset = data.iloc[:16]
        data = data.drop(subset.index)
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
    data = irisDataCopy #ripristino il dataset prima della prossima iterazione


# In[19]:


#acc = accuracy_score(ground, predictions)
#print("Accuratezza",acc)


# In[20]:


#calcolo accuratezza per ogni test --> 129 statistiche
k = 0
l = 8
leng = len(predictions)
lista = []
while leng>0:
        acc = accuracy_score(ground[k:l], predictions[k:l])
        #print("Accuratezza",acc)
        lista.append(acc)
        k+=8
        l+=8
        leng = leng-8
        
        
#calcolo media accuratezza
print("Accuratezza media:",statistics.mean(lista))


# In[ ]:




