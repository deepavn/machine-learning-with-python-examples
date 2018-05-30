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
 

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Function importing Dataset
def importdata():
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')
    py_class = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\zoo_data.csv')

    # Printing the dataset shape
    print ("\nDataset Length: ", len(py_data))
    print ("\nDataset Shape: ", py_data.shape)
    
    print(py_data.describe())

    return py_data, py_class

def splitdataset(py_data):

    X= py_data.iloc[:,1:-1]
    print("\n\nX DATASET ************************************************  \n")
    print(X.head())

    Y = py_data.iloc[:,-1:]
    print("\n\nY DATASET ************************************************  \n")
    print(Y)
    
    print("\n\nY BINARIZED DATASET ************************************************  \n")
    Y1 = label_binarize(Y, classes=[1, 2, 3, 4, 5, 6, 7])
    Y = label_binarize(Y, classes=[1, 2, 3, 4, 5, 6, 7])
    print(Y1)
    n_classes = Y.shape[1]
    print(n_classes)
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 3)

    print("\n\nTraining Data has: \n",X_train.shape)
    print("\n\nTesting Data has: \n",X_test.shape)



    return X, Y, X_train, X_test, y_train, y_test, n_classes

def main():
   
    # Building Phase
    data, classes = importdata()
    random_state = np.random.RandomState(0)
    X, Y, X_train, X_test, y_train, y_test, n_classes = splitdataset(data)
 
    print("Classes:\n", classes)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                    random_state=random_state))

    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    print ("y test: \n:", y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
  
    # Compute ROC curve and ROC area for each class    

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresholds = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    ''' 
    plt.figure()

    plt.plot(fpr[2], tpr[2], color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    '''


        # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red','purple','brown'])
    classes = ['Mammals', 'Birds', 'Reptiles', 'Fish', 'Amphibians','Insects','Invertebrates']    
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

  
# Calling main function
if __name__=="__main__":
    main()
