# Importing the required packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model


# Function importing Dataset
def importdata():
  #  py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')
  #  py_class = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')
    py_class = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\housing_data.csv', header=None,  names=py_class)
    py_data = py_data.fillna(0)
    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())
 
    return py_data, py_class

def splitdataset(py_data):
    

    # Use All features
    X= py_data.iloc[:,0:-1]
    print("\n\nX DATASET ************************************************  \n")
    print(X)

    Y = py_data.iloc[:,-1:]
    print("\n\nY DATASET ************************************************  \n")
    print(Y)
    '''

    # Use only one feature: Legs
    X= py_data.iloc[:,0:1]
    print("\n\nX DATASET ************************************************  \n")
    print(X)

    Y = py_data.iloc[:,-1:]
    print("\n\nY DATASET ************************************************  \n")
    print(Y)
    '''
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)

    print("\n\nTraining Data has: \n",X_train.shape)
    print("\n\nTesting Data has: \n",X_test.shape)

    return X, Y, X_train, X_test, y_train, y_test


def train_using_linearreg(X_train, X_test, y_train):

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    return regr


def train_using_gradientboosterreg(X_train, X_test, y_train):

    # Create linear regression object
    regr = GradientBoostingRegressor()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    return regr

# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("\nPredicted values:\n")
    print(y_pred)

    return y_pred
 
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("\nConfusion Matrix: \n\n", confusion_matrix(y_test, y_pred))


    print ("\nAccuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("\nReport :  \n\n",
    classification_report(y_test, y_pred))

def main():
   
  # Building Phase
    data, feature_names = importdata()
   
 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

  
    clf_lin_regr = train_using_linearreg(X_train, X_test, y_train)
    clf_graboo_regr = train_using_gradientboosterreg(X_train, X_test, y_train)
    # Operational Phase
    print("RESULTS USING KNN *********** :\n")

    # Prediction using Linear Regression
    y_pred = prediction(X_test, clf_lin_regr)

    # Prediction using Gradient Booster Regression.
    y_pred1 = prediction(X_test, clf_graboo_regr)
   
    # The coefficients
    print('Coefficients: \n', clf_lin_regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    
    '''
    for index, feature_name in enumerate(feature_names):
        if (index>=0):
            plt.figure(figsize=(8, 6))
            plt.scatter(data.iloc[:, index], data.iloc[:,-1:])
            plt.ylabel('Price /MEDV', size=25)
            plt.xlabel(feature_name, size=25)
            plt.tight_layout()
    '''
 

    predicted_linear = y_pred
    predicted_booster = y_pred1
    expected = y_test

 
    plt.scatter(expected, predicted_linear, color="green", alpha=0.5, label="With Linear Regression")
    plt.scatter(expected, predicted_booster, color="red", alpha=0.8, label="With Gradient Booster Regression")
    plt.plot(expected, expected, color="blue")
    
    plt.axis('tight')
    plt.xlabel('True MEDV ($1000s)')
    plt.ylabel('Predicted MEDV ($1000s)')   
    plt.legend()
    plt.grid(True)
    plt.show() 

    # Calling main function
if __name__=="__main__":
    main()
