#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
 
# Loading data
irisData = pd.read_csv("bank.txt",header=None,names=["f0","f1","f2","f3","class"])


# In[17]:


irisData


# In[18]:


#irisData['class'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'],
                        #[0, 1, 2], inplace=True)


# In[19]:


#irisData


# In[20]:


#Standardise
scaler = StandardScaler()
irisData.loc[:,["f0","f1","f2","f3"]] = scaler.fit_transform(irisData.loc[:,["f0","f1","f2","f3"]])


# In[21]:


#Normalize
irisData.loc[:,["f0","f1","f2","f3"]] = preprocessing.normalize(irisData.loc[:,["f0","f1","f2","f3"]], axis=1)


# In[22]:


# random_seed : int : Random number generator seed
# ---- Per cambiare porzione del dataset modificare il valore del seed ----
random_seed = 3
rgen = np.random.RandomState(random_seed)
def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]


# In[23]:


#irisData = irisData[1243:]
#irisData


# In[24]:


#setosa = irisData[irisData["class"]==0]
#versicolor = irisData[irisData["class"]==1]
#virginica = irisData[irisData["class"]==2]


# In[25]:


#dataset con 129 righe (128 quando sceglierò il test)
#versicolor = versicolor.iloc[:-11,:]
#virginica = virginica.iloc[:-10,:]


# In[26]:


#data = pd.concat([setosa,versicolor,virginica])
#data


# In[27]:


irisData = irisData.iloc[rgen.permutation(len(irisData.index))].copy()
irisData = irisData[347:]
irisDataCopy = irisData
irisDataCopy


# In[28]:


from scipy import spatial
#define fidelity 
def fidelity(v0,v1):
    return -pow(abs(1 - spatial.distance.cosine(v0, v1)),2)
    #return np.inner(v0,v1)*np.inner(v0,v1)


# In[29]:


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
        subset = irisData.iloc[:128]
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


# In[30]:


#acc = accuracy_score(ground, predictions)
#print("Accuratezza",acc)


# In[31]:


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




