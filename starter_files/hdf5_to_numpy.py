#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import h5py
import numpy as np
# import tensorflow_io as tfio


# In[5]:


data_path = 'Task01_BrainTumour.h5'
hdf5_filename = os.path.join(data_path)

df = h5py.File(hdf5_filename, "r")
imgs_training = df["imgs_train"]
# msks_training = df["msks_train"]
# imgs_testing = df["imgs_test"]
# msks_testing = df["msks_test"]


# In[ ]:


np.save('imgs_train.npy',np.asarray(imgs_training))
# np.save('msks_train.npy',np.asarray(msks_training))


# In[ ]:




