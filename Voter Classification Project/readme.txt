Geocoding.py - For a given CSV file containing addresses and cities, this script adds columns for 2010 Census Tracts and Census Block Groups, populated by a series of API calls to the census.gov website. 

Retrieve_ACS.py - For a given file, field, and vintage (year), this script adds columns for the given field drawn from the American Community Survey (5 - Year Estimates) on the Census Tract and Census Block Group level. 

Formatter.py - Standardizes the names of columns for a given CSV file.

DataReader.py - Reads in a list of CSV files and compiles them into one master CSV file, and splits the master file into testing and training examples for a ML model. 
