
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('iris.csv', names=['sepal_len_cm','sepal_width_cm','petal_len_cm','petal_width_cm','class'])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], df['class'], test_size=0.30, random_state=42)


# In[3]:


SVMmodel = svm.SVC(kernel='poly')
SVMmodel.fit(X_train,y_train)
KNN = neighbors.KNeighborsClassifier()
KNN.fit(X_train,y_train)


# In[4]:


print('SVM: ', SVMmodel.score(X_test,y_test))
print('KNN: ', KNN.score(X_test,y_test))


# In[5]:


print('SVM: ', accuracy_score(y_test,SVMmodel.predict(X_test)))
print('KNN: ', accuracy_score(y_test,KNN.predict(X_test)))

