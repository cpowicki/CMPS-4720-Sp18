import pandas as pd
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Includes new features: % Male, % Female, % White, % Black, % Asian, % Hispanic, % Enrolled as College Undergrads, % in each income brack <10k -> 200k, increasing by 5k.

# Perhaps I lost accuracy because I added too many features? I'm gonna need to play with this...

DF = pd.read_csv("Features2016Primary.csv")
X_train, X_test, y_train, y_test = train_test_split(DF.iloc[:, 5:], DF['PartyVoted'], test_size=0.30)


RFC = RandomForestClassifier(max_depth=4)
RFC.fit(X_train,y_train)
print(RFC.score(X_test,y_test))
# 0.710793697541

SVMmodel = svm.SVC(kernel='rbf')
# Adding Cross Validation
scores = cross_val_score(SVMmodel, DF.iloc[:, 5:], DF['PartyVoted'], cv=20)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Accuracy: 0.71 (+/- 0.00)
