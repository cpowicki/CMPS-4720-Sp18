
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import Perceptron
from math import log
df_train = pd.read_csv('SPECT.train.txt', header=None)
df_test = pd.read_csv('SPECT.test.txt', header=None)


# In[3]:


# Performs summation of xi * wi
def weight_summation(xi, w):
    summation = w[0]
    for i in range(len(xi) - 1):
        summation += w[i + 1] * xi.iloc[i]
    return summation

def predict_thresh(xi , w):
    summation = weight_summation(xi, w)
    if summation >= 0.0:
        return 1.0
    else:
        return 0.0

# Trains weights given a training dataframe, a learning rate, and a number of epochs.
def train_model(training_df, R, epochs):
    weights = [0.0 for i in range(len(training_df.columns))] # initialize all weights to 0. 
    for epoch in range(epochs):
        for index, row in training_df.iterrows():
            pred = predict_thresh(row.iloc[1:], weights)
            error = row.iloc[0] - pred # Error is actual - predicted
            weights[0] = weights[0] + R * error # W0 (the intercept) is not multiplied by xi.
            for i in range(1, len(row)): # Updates the remaining weights according to stochastic GD rule
                weights[i] = weights[i] + R * error * row.iloc[i]
    return weights

# Returns an accuracy measure, the number correctly predicted out of the total. 
def test_model (test_df, weights):
    num_correct = 0
    num_wrong = 0
    for index, row in test_df.iterrows():
        pred = predict_thresh(row.iloc[1:], weights)
        if pred == row.iloc[0]:
            num_correct += 1
        else:
            num_wrong += 1
    return (num_correct / (num_correct + num_wrong)) * 100

# Accuracy training on the 'Test' dataset is much higher than when the model is trained on the 
# 'Training' dataset ... 
print('(inverted test-train) Accuracy: %.2f' % (test_model(df_train, train_model(df_test, 0.1, 56))))
print('Accuracy: %.2f' % (test_model(df_test, train_model(df_train, 0.1, 45))))

