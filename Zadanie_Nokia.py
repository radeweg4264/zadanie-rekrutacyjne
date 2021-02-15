#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from tpot import TPOTClassifier 
from sklearn.model_selection import train_test_split
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


data_x = pd.read_csv('dataset/x_train_CA1.csv')
data_y  = pd.read_csv('dataset/y_train_CA1.csv')


# In[3]:


data_y.head(5)


# In[4]:


data_x.head()


# In[5]:


#rename headers in data_train
header_name = ['x_train_CA1']
data_x = pd.read_csv('dataset/x_train_CA1.csv',header=None,skiprows=0,names=header_name)


# In[6]:


data_x.head()


# In[7]:


#rename headers in data_test
header_name = ['y_train']
data_y = pd.read_csv('dataset/y_train_CA1.csv',header=None,skiprows=0,names=header_name)


# In[8]:


data_y['filtered'] = data_y['y_train'].apply(lambda x: 1 if  x >= 0.5 else 0)
data_y['filtered'] = data_y['filtered'].astype(float)
print(data_y.head(20))


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y['filtered'], test_size=0.33, random_state=42)


# In[10]:


tpot = TPOTClassifier(verbosity=2, max_time_mins=1, max_eval_time_mins=0.04, population_size=40)
tpot.fit(x_train, y_train.values.ravel())


# In[11]:


print(tpot.score(x_test,y_test))


# In[12]:


import glob

dfs = []
all_files = glob.glob("dataset/x_train_CA*.csv")
print(all_files)


# In[13]:


df_merged = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_merged, ignore_index=True)
df_merged.to_csv( "merged.csv")


# In[14]:


data_x_test=pd.read_csv('merged.csv')


# In[15]:


header_name = ['x_train']
data_x_test = pd.read_csv('merged.csv',header=None,skiprows=0,names=header_name)


# In[16]:


remove_nan = data_x_test.dropna()
print(remove_nan)


# In[17]:


data_x_test


# In[18]:


dfs = []
all_files = glob.glob("dataset/y_train_CA*.csv")
print(all_files)


# In[19]:


df_merged = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_merged, ignore_index=True)
df_merged.to_csv( "merged2.csv")


# In[20]:


data_y_test=pd.read_csv('merged2.csv')


# In[21]:


header_name = ['y_train']
data_y_test = pd.read_csv('merged2.csv',header=None,skiprows=0,names=header_name)


# In[22]:


data_y_test['filtered'] = data_y_test['y_train'].apply(lambda x: 1 if  x >= 0.5 else 0)
data_y_test['filtered'] = data_y_test['filtered'].astype(float)
print(data_y_test.head(20))


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(data_x_test, data_y_test['filtered'], test_size=0.33, random_state=42)


# In[24]:


tpot = TPOTClassifier(verbosity=2, max_time_mins=3, max_eval_time_mins=0.04, population_size=40)
tpot.fit(x_train, y_train.values.ravel())


# In[25]:


print(tpot.score(x_test,y_test))

