#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
X = genfromtxt('S21_M4DS_PRJ_02_Optimization_data_X.csv', delimiter=',')
y= genfromtxt('S21_M4DS_PRJ_01_Optimization_data_Y.csv', delimiter=',')
X =  X[:, ~np.isnan(X).any(axis=0)]
r,c = np.shape(X)
a0=np.ones([r,1])
X= np.append(a0, X, axis=1)
print(X)
print(np.shape(X))
print(np.shape(y))


# In[67]:


a_AS1 = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("The matrix a using matrix calcus method\n",a_AS1)
print(np.shape(a_AS1))


# In[68]:


a_AS2 = np.linalg.pinv(X).dot(y)
print("The matrix a using pseudo inverse method\n",a_AS2)
print(np.shape(a_AS2))


# In[69]:


def cost_function(a,X,y):
    m = len(y)
    h = np.dot(X,a)
    cost = 1/(2*m)* np.sum(np.square(h-y))
    return cost


# In[70]:


# input a is from matrix calcus method
cost_function(a_AS1,X,y)


# In[71]:


# input a is from psuedo inverse method
cost_function(a_AS2,X,y)


# In[72]:


def gradient_descent(a,X,y,learning_rate=0.1,iteration=1000):
    a_history = []
    J_history = []
    for i in range(iteration):
        h = np.dot(X,a)
        m= len(y)
        a = a -(1/m)* learning_rate* np.dot(X.T,(h-y))
        a_history =+ a
        J_history.append(cost_function(a,X,y))
        plt.scatter(i,J_history[-1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Cost function', fontsize=16)
    plt.show()
    return a, J_history[-1]


# In[73]:


r,c = np.shape(X)
a=np.zeros([c,1])

gradient_descent(a,X,y)


# In[76]:


def gradient_descent1(a,X,y,learning_rate=0.1,tol=1e-5):
    dJ=1e-1
    J_history=[0]
    i=0
    while dJ> tol:
        h = np.dot(X,a)
        m= len(y)
        a = a -(1/m)* learning_rate* np.dot(X.T,(h-y))
        a_history =+ a
        J_history.append(cost_function(a,X,y))
        dJ=abs(J_history[-2]-J_history[-1])
        plt.scatter(i,J_history[-1])
        i= i+1
    plt.xlabel('d(J)', fontsize=18)
    plt.ylabel('Cost function', fontsize=16)
    return a, J_history[-1]
plt.show()


# In[77]:


gradient_descent1(a,X,y)


# In[ ]:




