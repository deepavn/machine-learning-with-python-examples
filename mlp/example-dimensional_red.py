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

def main():

        
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

    #Use StandardScaler to help you standardize the dataset’s features -
    #  onto unit scale (mean = 0 and variance = 1) -
    #  which is a requirement for the optimal performance of many machine learning algorithms.

    from sklearn.preprocessing import StandardScaler
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()             

# Calling main function
if __name__=="__main__":
    main()