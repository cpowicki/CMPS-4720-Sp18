import pandas as pd
import numpy as np
import requests as req
import json
import sys
import time as t


def geocode(address, city):
    r = req.get('https://geocoding.geo.census.gov/geocoder/geographies/address?street=' +
                address + '&city=' + city + '&state=MA&benchmark=4&vintage=4&format=json')
    response = r.json()
    if 'exceptions' in response.keys() or 'result' not in response.keys():
        return False
    elif len(response['result']['addressMatches']) == 0:
        return False
    else:
        return r.json()['result']['addressMatches'][0]


def getCTract(x):
    if type(x) is str or x == False:
        return "None"
    else:
        if 'TRACT' not in x['geographies']['Census Tracts'][0].keys():
            return "None"
        else:
            return x['geographies']['Census Tracts'][0]['TRACT']


def getCounty(x):
    if type(x) is str or x == False:
        return "None"
    else:
        if 'COUNTY' not in x['geographies']['Counties'][0].keys():
            return "None"
        else:
            return x['geographies']['Counties'][0]['COUNTY']


def getCBlock(x):
    if type(x) is str or x == False:
        return "None"
    else:
        if 'BLKGRP' not in x['geographies']['2010 Census Blocks'][0].keys():
            return "None"
        else:
            return x['geographies']['2010 Census Blocks'][0]['BLKGRP']


filename = sys.argv[1]
num_col = int(sys.argv[2])
street_col = int(sys.argv[3])
city_col = int(sys.argv[4])


DF = pd.read_csv(filename)
Addresses = []
CensusTracts = []
CensusBlocks = []
Counties = []
num_failure = 0
num_completed = 0
start = t.time()
for row_index, row in DF.iterrows():
    addr_str = str(row.iloc[num_col]) + " " + str(row.iloc[street_col]).title()
    resp = geocode(addr_str, row.iloc[city_col])
    Addresses.append(addr_str)
    if resp != False:
        tract = getCTract(resp)
        block = getCBlock(resp)
        county = getCounty(resp)
        if tract == "None" or block == "None" or county == "None":
            num_failure += 1
        CensusTracts.append(tract)
        CensusBlocks.append(block)
        Counties.append(county)
    else:
        num_failure += 1
        CensusTracts.append('Failure')
        CensusBlocks.append('Failure')
        Counties.append('Failure')
    num_completed += 1
    elapsed = t.time() - start
    percent_comp = (num_completed / len(DF.index)) * 100
    rate = percent_comp / elapsed
    Estimated_Remaining = ((100 - percent_comp) / rate) / 60
    print("%d Completed. %f Percent Completed Overall. Rate: %f T remaing: %f" %
          (num_completed, percent_comp, rate, Estimated_Remaining))

DF.loc[:, 'Conc_Addr'] = pd.Series(Addresses, index=DF.index)
DF.loc[:, 'County'] = pd.Series(Counties, index=DF.index)
DF.loc[:, 'CTract'] = pd.Series(CensusTracts, index=DF.index)
DF.loc[:, 'CBlock'] = pd.Series(CensusBlocks, index=DF.index)

DF.to_csv('Geocoded-' + filename)
print("Number of Failures: ", num_failure)
