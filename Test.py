#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


# In[2]:


import Recognition


# In[6]:


net = Recognition.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)


# In[3]:


net = Recognition.Network([784, 30, 10])


# In[4]:


net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# In[4]:


import network2


# In[5]:



net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)


# In[6]:


# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.


# In[7]:


net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)


# In[8]:


# chapter 3 - Regularization (weight decay) example 1 (only 1000 of training data and 30 hidden neurons)


# In[9]:


net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000], 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)


# In[ ]:




