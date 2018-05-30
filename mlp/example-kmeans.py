# Importing the required packages
import numpy as np
import pandas as pd
import pprint

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


# Function importing Dataset
def importdata():
    py_class = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\housing_data.csv', header=None,  names=py_class)
    
    #Clean
    py_data = py_data.fillna(0)

    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())

    return py_data, py_class
 

def main():
   
  # Building Phase
    data, data_feature_names = importdata()
 

    print("Feature names: ")
    print(data_feature_names)
    X = data

    #plt.scatter(X.iloc[:,0:1].values,X.iloc[:,-1:].values, label='True Position')

    kmeans = KMeans(n_clusters=2)  
    kmeans.fit(X)  
    print(kmeans.cluster_centers_)
    print(kmeans.labels_) 

    plt.scatter(X.iloc[:,1:2].values,X.iloc[:,-1:].values, c=kmeans.labels_, cmap='rainbow')
    plt.show()





# Calling main function
if __name__=="__main__":
    main()
