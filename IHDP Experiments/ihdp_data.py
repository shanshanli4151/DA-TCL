#!/usr/bin/env python
# coding: utf-8

# In[1]:


def convert_file(x):
    x = x.values
    x = x.astype(float)
    return x


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


def load_and_format_covariates_ihdp(file_path='/Users/asus/Desktop/data/ihdp_npci_1.csv'):

    data = np.loadtxt(file_path, delimiter=',')
#binfeats代表二分类？对19个二分类
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]

    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    perm = binfeats + contfeats
    x = x[:, perm]
    return x


# In[4]:


def load_all_other_crap(file_path='/Users/asus/Desktop/data/ihdp_npci_1.csv'):
    data = np.loadtxt(file_path, delimiter=',')
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    return t.reshape(-1, 1), y, y_cf, mu_0, mu_1


def main():
    pass


if __name__ == '__main__':
    main()


# In[ ]:




