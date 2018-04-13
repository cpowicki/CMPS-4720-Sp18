
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


DF = pd.read_csv("Features-2014StatePrimary.csv")
X_train, X_test, y_train, y_test = train_test_split(
    DF.iloc[:, 5:], DF['PartyVoted'], test_size=0.30, random_state=42)


# In[ ]:


SVMmodel = svm.SVC(kernel='rbf')
SVMmodel.fit(X_train,y_train)
print('SVM: ', SVMmodel.score(X_test,y_test))

# SVM: 0.823737195337

