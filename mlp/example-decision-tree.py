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
from sklearn import tree
from sklearn.model_selection import GridSearchCV


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
    print(Y)
 
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 3)

    print("\n\nTraining Data has: \n",X_train.shape)
    print("\n\nTesting Data has: \n",X_test.shape)



    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini", splitter="random" ,random_state = 100, max_depth=10, min_samples_leaf=2)

    # Performing training
    clf_gini.fit(X_train, y_train)
     

    return clf_gini
     
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 10, min_samples_leaf =2)

    # Performing training
    clf_entropy.fit(X_train, y_train)


    return clf_entropy

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
    data = importdata()
    data_class_names = ['Mammals', 'Birds', 'Reptiles', 'Fish', 'Amphibians','Insects','Invertebrates']   
    data_feature_names = list(data.columns[1:17])
    

    print("Feature names: ")
    print(data_feature_names)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

  
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

 
    # Operational Phase
    print("RESULTS USING GINI INDEX *********** :\n")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("RESULTS USING ENTROY *********** :\n")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
 

    # VISUAL REPRESENTATION
    with open("C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\dt_train_entropy.txt", "w") as e:
        e = tree.export_graphviz(clf_entropy, 
                        out_file=e, 
                        feature_names=data_feature_names,
                        class_names=data_class_names)    



    with open("C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\dt_train_gini.txt", "w") as f:
        f = tree.export_graphviz(clf_gini, out_file=f,
                        feature_names=data_feature_names,
                        class_names=data_class_names)    

    # Go to http://webgraphviz.com/ to view the tree    
    
    
    cm = confusion_matrix(y_test, y_pred_gini)
    tick_marks = np.arange(len(data_class_names))
  
    df_cm = pd.DataFrame(cm, data_class_names, data_class_names)

   # plt.figure(figsize = (10,7))
    plt.xticks(tick_marks, data_class_names, rotation=45)
    plt.yticks(tick_marks, data_class_names, rotation=-45 )

    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    
    plt.show()
    '''
    dataplt = plt

    lines = [y_test,y_pred_entropy,y_pred_gini]
    colors  = ['r','g','b']
    labels  = ['Test','Entropy','Gini']


    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N))**2  # 0 to 15 point radii    

    dataplt.plot(y_test,y_test, c="red")
    dataplt.scatter(y_test,y_pred_entropy, c="blue", s=area+10, alpha=0.5)
    dataplt.scatter(y_test,y_pred_gini, c="yellow", s=area, alpha=0.5)    
    dataplt.show()

    '''
    
  
# Calling main function
if __name__=="__main__":
    main()
