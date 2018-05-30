# Importing the required packages
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz


# Function importing Dataset
def importdata():
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')
    py_class = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')

    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())

    '''
    # Printing the dataset obseravtions
    print(py_data.head())
    print(py_class.head())
   

    corplt = plt
    corr = py_data.iloc[:,1:-1].corr()
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    corplt.figure(figsize=(14,14))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 12},
                cmap = colormap, linewidths=0.1, linecolor='white')
    corplt.title('Correlation of ZOO Features', y=1.05, size=15) 
    '''
  #  plt.show()

    return py_data

def splitdataset(py_data):

    X= py_data.iloc[:,1:-1]
    print("\n\nX DATASET ************************************************  \n")
    print(X.head())

    Y = py_data.iloc[:,-1:]
    print("\n\nY DATASET ************************************************  \n")
    print(Y.head())
 
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 3)

    print("\n\nTraining Data has: \n",X_train.shape)
    print("\n\nTesting Data has: \n",X_test.shape)



    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with Random Forest
def train_using_randomeforest(X_train, X_test, y_train):
 
    # Creating the classifier object
    clf_randomforest = RandomForestClassifier()

    # Performing training
    clf_randomforest.fit(X_train, y_train)
     

    return clf_randomforest

# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with Randomforest
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
    data = importdata()
    data_class_names = ['Mammals', 'Birds', 'Reptiles', 'Fish', 'Amphibians','Insects','Invertebrates']   
    data_feature_names = list(data.columns[1:17])
    

    print("Feature names: ")
    print(data_feature_names)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    clf_randomforest = train_using_randomeforest(X_train, X_test, y_train)
    
    # Operational Phase
    print("RESULTS USING RANDOME FOREST *********** :\n")

    # Prediction using random forest
    y_pred_randomforest = prediction(X_test, clf_randomforest)
    cal_accuracy(y_test, y_pred_randomforest)
 

    
    # VISUAL REPRESENTATION
    with open("C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\dt_train_randomforest.txt", "w") as z:
       z = export_graphviz(clf_randomforest.estimators_[5], 
                        out_file=z, 
                        feature_names=data_feature_names,
                        class_names=data_class_names)    
    
    cm = confusion_matrix(y_test, y_pred_randomforest)
    tick_marks = np.arange(len(data_class_names))
  
    df_cm = pd.DataFrame(cm, data_class_names, data_class_names)

   # plt.figure(figsize = (10,7))
    plt.xticks(tick_marks, data_class_names, rotation=45)
    plt.yticks(tick_marks, data_class_names, rotation=-45 )

    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    
    #plt.show()

  
    # Go to http://webgraphviz.com/ to view the tree   
  
# Calling main function
if __name__=="__main__":
    main()
