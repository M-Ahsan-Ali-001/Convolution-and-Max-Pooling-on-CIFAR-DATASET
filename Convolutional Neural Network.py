#!/usr/bin/env python
# coding: utf-8

# In[43]:


import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[44]:


#CIFAR DATASET USED

(train ,train_labels),(test, test_labels)=datasets.cifar10.load_data()


# In[45]:


cl_nm=['airplane','automobile','bird' , 'cat' ,'deer' ,'dog'
      ,'frog', 'horse' , 'ship', 'truck']

train=train/255.0
test=test/255.0


# In[46]:


idx=1
plt.figure()
plt.imshow(train[idx],cmap=plt.cm.binary)
plt.xlabel(cl_nm[train_labels[idx][0]])
plt.colorbar()
plt.show()


# In[47]:


train.shape


# In[48]:


test.shape


# In[49]:


from tensorflow import keras
model=keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))



# In[50]:


model.summary()


# In[51]:


model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='relu'))


# In[52]:


model.summary()


# In[57]:


model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
             )
storage=model.fit(train,train_labels ,epochs=10 ,validation_data=(test,test_labels))


# In[58]:


test_loss,test_accu=model.evaluate(test,test_labels, verbose =2)
print('accuracy on unseen data:  ',test_accu)


# In[ ]:




