
# coding: utf-8

# In[28]:


import torch
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.autograd import Function


# In[29]:


df = pd.read_csv('iris.csv', names=['sepal_len_cm','sepal_width_cm','petal_len_cm','petal_width_cm','class'])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], df['class'], test_size=0.3, random_state=42)



# In[30]:


for i in range(len(y_train)):
    if str(y_train.iloc[i]) == 'Iris-versicolor':
        y_train.iloc[i] = 0.0
    elif str(y_train.iloc[i]) == 'Iris-setosa':
        y_train.iloc[i] = 1.0
    else:
        y_train.iloc[i] = 2.0
        
for i in range(len(y_test)):
    if str(y_test.iloc[i]) == 'Iris-versicolor':
        y_test.iloc[i] = 0.0
    elif str(y_test.iloc[i]) == 'Iris-setosa':
        y_test.iloc[i] = 1.0
    else:
        y_test.iloc[i] = 2.0


# In[31]:


inpt_train = Variable(torch.Tensor(X_train.as_matrix()))
outpt_train = Variable(torch.Tensor(y_train.as_matrix()))

inpt_test = Variable(torch.Tensor(X_test.as_matrix()))


# In[32]:


def thresh_funct(i):
    if abs(2 - float(i)) < 0.5:
        i = 2
    elif abs(1 - float(i)) < 0.5:
        i = 1
    elif abs(float(i)) < 0.5:
        i = 0
    else:
        i = -1
    return i


# In[33]:


def thresh_funct_standard(i):
    if i < 1.0/3.0:
        i = 0.0
    elif i < 2.0/3.0:
        i = 1.0/3.0
    else:
        i = 2.0/3.0
    return i


# In[34]:


model = torch.nn.Sequential( 
    torch.nn.Linear(4, 35), 
    torch.nn.ReLU(), 
    torch.nn.Linear(35, 1)
)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

# Random update
for i in range(3000):
    y_pred = model(inpt_train)
    loss = loss_fn(y_pred,outpt_train)
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

y_pred2 = model(inpt_test)
y_pred2.data.apply_(lambda x: thresh_funct(x))
total = 0
correct = 0
array = y_pred2.data.numpy()
num_fail = 0;
for i in range(len(array)):
    if array[i] == y_test.iloc[i]:
        correct += 1
    if array[i] == -1:
        num_fail += 1
    total += 1
print('Accuracy: %f, Correct: %d, Total: %d' % ((correct/total) * 100, correct, total ))
print('Failures: %d' % (num_fail))

# Printed Results:
# Accuracy: 86.666667, Correct: 39, Total: 45
# Failures: 1
    

