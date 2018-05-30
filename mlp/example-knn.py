# Importing the required packages
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler  

# Function importing Dataset
def importdata():
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')
    py_class = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')

    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())
 
  #  plt.show()

    return py_data

def splitdataset(py_data):

    X= py_data.iloc[:,1:-1].values
    print("\n\nX DATASET ************************************************  \n")
    print(X)

    Y = py_data.iloc[:,-1:].values
    print("\n\nY DATASET ************************************************  \n")
    print(Y)
 
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.4, random_state = 100)

    '''
    Feature Scaling
    Before making any actual predictions, it is always -
    a good practice to scale the features so that all of them can be uniformly evaluated. 

    '''

    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  


    print("\n\nTraining Data has: \n",X_train.shape)
    print("\n\nTesting Data has: \n",X_test.shape)

    return X, Y, X_train, X_test, y_train, y_test


def train_using_knn(X_train, X_test, y_train):

    clf_knn = KNeighborsClassifier(n_neighbors=5)  
    clf_knn.fit(X_train, y_train)  
    
    return clf_knn

# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("\nPredicted values:\n")
    print(y_pred)

    return y_pred

def get_score_knn(X_train, X_test, y_train, y_test):
    knn_score = []

    # Calculating knn_score for K values between 1 and 40
    for i in range(1, 40):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        knn_score.append(accuracy_score(y_test, pred_i))    
         
    return knn_score
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("\nConfusion Matrix: \n\n", confusion_matrix(y_test, y_pred))


    print ("\nAccuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("\nReport :  \n\n",
    classification_report(y_test, y_pred))



def main():
   
  # Building Phase
    data = importdata()
    data_class_names = ['Mammals', 'Birds', 'Reptiles', 'Fish', 'Amphibians','Insects','Invertebrates']   
    data_feature_names = list(data.columns[1:17])
    

    print("Feature names: ")
    print(data_feature_names)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

  
    clf_knn = train_using_knn(X_train, X_test, y_train)
    
    # Operational Phase
    print("RESULTS USING KNN *********** :\n")

    # Prediction using KNN
    y_pred_knn = prediction(X_test, clf_knn)
    cal_accuracy(y_test, y_pred_knn)
    
    cm = confusion_matrix(y_test, y_pred_knn)

    knn_score = get_score_knn(X_train, X_test, y_train, y_test)

    plt.figure(figsize=(12, 6))  
    plt.plot(range(1, 40), knn_score, color='red', linestyle='dashed', marker='o',  
            markerfacecolor='blue', markersize=10)
    plt.title('KNN Score Vs KNN Neighbors')  
    plt.xlabel('# of Neighbors in KNN')  
    plt.ylabel('KNN Accuracy Score') 
    plt.show()
  
# Calling main function
if __name__=="__main__":
    main()
