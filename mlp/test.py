# Importing the required packages
import numpy as np
import pandas as pd
import pprint

import matplotlib.pyplot as plt
 

dfcities = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\india-cities-states.csv', usecols=['City'])
# Function importing Dataset
def importdata():
   
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')

    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())

    return py_data
 

def main():
   
  # Building Phase
    data = importdata()
    
    X = data
    df = data
    list_places = dfcities.loc[:,'City'] 

    print(list_places)

    df['userIndiaCity'] = df['animalname'].isin(list_places)  #df['userLocation'].apply(get_city_name)
    df['userUsedWord'] = df['animalname'].str.contains("as") 
 
    print(df.head(10))



# Calling main function
if __name__=="__main__":
    main()
