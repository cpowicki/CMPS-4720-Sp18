import pandas as pd
import numpy as np
import sys
from sklearn import tree
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression, Perceptron
import graphviz

Filenames = sys.argv[1:]


# Includes new features: % Male, % Female, % White, % Black, % Asian, % Hispanic, % Enrolled as College Undergrads, % in each income brack <10k -> 200k, increasing by 5k.

DF = pd.DataFrame()
for i in Filenames:
    DF2 = pd.read_csv(i,low_memory=False)
    Columns = DF2.columns.tolist()
    Columns.remove("PartyVoted")
    Columns.remove("Population_tract")
    Columns.remove("Population_block")
    Columns.remove("Conc_Addr")
    Columns.remove("County")
    Columns.remove("CTract")
    Columns.remove("CBlock")
    Columns = ["PartyVoted","Population_tract","Population_block"] + Columns
    DF2 = DF2[Columns]
    DF = DF.append(DF2)
    DF = DF[Columns]


Num_Folds = int(len(DF.index) / (len(DF.index) * 0.3))

Imp = Imputer()
DF["Med_Property_Value_tract"] = Imp.fit_transform(DF[["Med_Property_Value_tract"]]).ravel()
DF["Med_Property_Value_block"] = Imp.fit_transform(DF[["Med_Property_Value_block"]]).ravel()


RFC = RandomForestClassifier(max_depth=5)
log = LogisticRegression()
ada = AdaBoostClassifier()

scores = cross_val_score(RFC, DF.iloc[:, 3:], DF['PartyVoted'], cv=Num_Folds)
print("RFC Base Data Set: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(log, DF.iloc[:, 3:], DF['PartyVoted'], cv=Num_Folds)
print("Log Base Data Set: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(ada, DF.iloc[:, 3:], DF['PartyVoted'], cv=Num_Folds)
print("ADA Base Data Set: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



NoTractCols = [i for i in Columns if not i.endswith("tract")]
NoTractCols.remove("Population_block")
NoTractCols.remove("PartyVoted")
NoTractCols = ["PartyVoted","Population_block"] + NoTractCols
DF_NoTract = DF[NoTractCols]

scores = cross_val_score(RFC, DF_NoTract.iloc[:, 2:], DF_NoTract['PartyVoted'], cv=Num_Folds)
print("RFC No Tract: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(log, DF_NoTract.iloc[:, 2:], DF_NoTract['PartyVoted'], cv=Num_Folds)
print("Log No Tract: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(ada, DF_NoTract.iloc[:, 2:], DF_NoTract['PartyVoted'], cv=Num_Folds)
print("ADA No Tract: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

NoBlockCols = [i for i in Columns if not i.endswith("block")]
NoBlockCols.remove("Population_tract")
NoBlockCols.remove("PartyVoted")
NoBlockCols = ["PartyVoted","Population_tract"] + NoBlockCols
NoBlockDF = DF[NoBlockCols]

scores = cross_val_score(RFC, NoBlockDF.iloc[:, 2:], NoBlockDF['PartyVoted'], cv=Num_Folds)
print("RFC No Block: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(log, NoBlockDF.iloc[:, 2:], NoBlockDF['PartyVoted'], cv=Num_Folds)
print("log No Block: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(ada, NoBlockDF.iloc[:, 2:], NoBlockDF['PartyVoted'], cv=Num_Folds)
print("ADA No Block: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#RFC Base Data Set: Accuracy: 0.73 (+/- 0.05)
#Log Base Data Set: Accuracy: 0.75 (+/- 0.00)
#ADA Base Data Set: Accuracy: 0.73 (+/- 0.05)
#RFC No Tract: Accuracy: 0.73 (+/- 0.05)
#Log No Tract: Accuracy: 0.75 (+/- 0.00)
#ADA No Tract: Accuracy: 0.72 (+/- 0.06)
#RFC No Block: Accuracy: 0.73 (+/- 0.05)
#log No Block: Accuracy: 0.75 (+/- 0.00)
#ADA No Block: Accuracy: 0.73 (+/- 0.05)










