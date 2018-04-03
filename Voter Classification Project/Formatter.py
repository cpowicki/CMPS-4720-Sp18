import pandas as pd
import numpy as np
import sys

# This is a little script to standardize the format of all of my separate data files.
# Columns are inconsistently named/indexed across the 12 files I'm using, so before combining them,
# I have to make sure they are formatted in the form:
# "PartyVoted" "Conc_Addr" "CBlock" "CTract" "City" "County" ... ACS Features ...

filename = sys.argv[1]
labels = sys.argv[2::2]
newlabels = sys.argv[3::2]

DF = pd.read_csv(filename)
DF_Formatted = pd.DataFrame(index=DF.index)
for i in range(len(labels)):
    DF_Formatted.loc[:, newlabels[i]] = pd.Series(DF.loc[:, labels[i]], index=DF.index)
DF_Formatted.to_csv("Formatted-" + filename)
