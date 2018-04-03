import pandas as pd
import numpy as np
import requests as req
import json
import sys

CensusKey = '685f8daaa8791c3869eb466279af33ac74e9a3bd'

filename = sys.argv[1]
year = sys.argv[2]
field = sys.argv[3]
field_name = sys.argv[4]


def get_ACS_Ctract(row, year, field):
    r = req.get("https://api.census.gov/data/" + year + "/acs/acs5?get=NAME," + field +
                "&for=tract:" + row['CTract'] + "&in=state:26" + "%20county:" + row['County'] + "&key=" + CensusKey)
    response = r.json()
    return response[field]


def get_ACS_CBlock(row, year, field):
    r = req.get("https://api.census.gov/data/" + year + "/acs/acs5?get=NAME," + field + "&for=block%20group:" +
                row['CBlock'] + "&in=state:26" + "%20county:" + row['County'] + "%20tract:" + row['CTract'] + "&key=" + CensusKey)
    response = r.json()
    return response[field]


# Main
DF = pd.read_csv(filename)
# These two maps prevent us from sending API requests we already have data for.
CTract_Map = {}
CBlock_Map = {}
CTract_Stats = []
CBlock_Stats = []
for row_index, row in DF.iterrows():
    Row_Tract = row['CTract']
    Row_Block = Row_Tract + row['CBlock']

    if Row_Tract in CTract_Map.keys():
        CTract_Stats.append(CTract_Map[Row_Tract])
    else:
        new_stat = get_ACS_Ctract(row, year, field)
        CTract_Stats.append(new_stat)
        CTract_Map[Row_Tract] = new_stat

    if Row_Block in CBlock_Map.keys():
        CBlock_Stats.append(CBlock_Map[Row_Block])
    else:
        new_stat = get_ACS_CBlock(row, year, field)
        CBlock_Stats.append(new_stat)
        CBlock_Map[Row_Block] = new_stat

DF.loc[:, field_name + "_tract"] = pd.Series(CTract_Stats, index=DF.index)
DF.loc[:, field_name + "_block"] = pd.Series(CBlock_Stats, index=DF.index)
DF.to_csv(field_name + "-" + filename)
