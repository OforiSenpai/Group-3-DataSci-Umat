#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import liberies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 


# In[2]:


#Generating our dataset
dataset = make_blobs(n_samples=200,
                     centers=4,
                     n_features=2,
                     cluster_std=1.6,
                     random_state=50)


# In[3]:


print(dataset)


# In[4]:


points = dataset[0]


# In[5]:


print(dataset[0])


# In[6]:


#import K-Means
from sklearn.cluster import KMeans


# In[7]:


#Create a K-Means Object
KMeans = KMeans(n_clusters=4)


# In[8]:


#fit the K-Means  object to the datasets
KMeans.fit(points)


# In[9]:


#shows the unclustered plot
plt.scatter(dataset[0][:,0], dataset[0][:,1])


# In[10]:


clusters = KMeans.cluster_centers_


# In[11]:


#print out the clusters
print(clusters)


# In[12]:


y_km = KMeans.fit_predict(points)


# In[13]:


plt.scatter(points[y_km == 0,0], points[y_km == 0,1], s=80, color='red') 
plt.scatter(points[y_km == 1,0], points[y_km == 1,1], s=80, color='blue')
plt.scatter(points[y_km == 2,0], points[y_km == 2,1], s=80, color='yellow')
plt.scatter(points[y_km == 3,0], points[y_km == 3,1], s=80, color='green')
plt.scatter(clusters[0][0], clusters[0][1], marker='*', s=200, color='black') 
plt.scatter(clusters[1][0], clusters[1][1], marker='*', s=200, color='black')
plt.scatter(clusters[2][0], clusters[2][1], marker='*', s=200, color='black')
plt.scatter(clusters[3][0], clusters[3][1], marker='*', s=200, color='black')

plt.show()


# In[ ]:





# In[ ]:




