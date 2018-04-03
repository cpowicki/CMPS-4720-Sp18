import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

filenames = sys.argv[1:]
MasterDF = pd.DataFrame()
PartyVoted_Series = pd.Series()
MFI_CTract_Series = pd.Series()  # Median Family Income Statistic (Census Tract)
MFI_CBlock_Series = pd.Series()  # Median Family Income Statistic (Census Block)
# ...Series objects for each Additional ACS Data Feature...
for i in filenames:
    nxtDF = pd.read_csv(filename)
    MFI_CTract_Series = MFI_CTract_Series.append(pd.Series(nextDF.loc[:, "CTract_MFI"]))
    MFI_CBlock_Series = MFI_CBlock_Series.append(pd.Series(nextDF.loc[:, "CBlock_MFI"]))
    # ACS_Series.append(nextDF.loc[:,"FeatureName"])
MasterDF[:, "PartyVoted"] = PartyVoted_Series
MasterDF[:, "CTract_MFI"] = MFI_CTract_Series
MasterDF[:, "CTract_MFI"] = MFI_CBlock_Series
#MasterDF[:, "FeatureName"] = ACS_Series
MasterDF.to_csv("Master Dataset")
X_train, X_test, y_train, y_test = train_test_split(
    MasterDF.iloc[:, 1:], MasterDF['PartyVoted'], test_size=0.30, random_state=42)
