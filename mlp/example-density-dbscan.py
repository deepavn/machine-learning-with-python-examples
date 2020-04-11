import time as time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  

# Function importing Dataset
def importdata():
    py_class = ['Distance_Feature','Speeding_Feature']
    py_data = pd.read_csv('drivers_data.csv', header=None,  names=py_class)
    
    #Clean
    py_data = py_data.fillna(0)


    scaler = StandardScaler()  
    scaler.fit(py_data)

    X = scaler.transform(py_data)
    #X = py_data.values     
    # Printing the dataset shape
    print ("\nDataset Length: ", len(X))
    print ("\nDataset Shape: ", X.shape)

    py_data = X
 

    
 
    print(X)
    return X, py_class

def set_colors(labels, colors='rgbykcm'):
    colored_labels = []
    for label in labels:
        colored_labels.append(colors[label])
    return colored_labels

def set_legend():
    l_labels = []
     
    l_labels.append("Age")
    l_labels.append("Urband")
    l_labels.append("Speed")
    l_labels.append("Distance")
    return l_labels

def main():

  # Building Phase
    data, data_feature_names = importdata()
    X = data
    print("Data: X: ")
    print(X)
    print("Feature names: ")
    print(data_feature_names)
    # Define the structure A of the data. Here a 10 nearest neighbors
    from sklearn.neighbors import kneighbors_graph
    
    # Compute clustering
    print("Compute unstructured hierarchical clustering...")
    st = time.time()
    cls_dbscan =  DBSCAN(eps=0.3, min_samples=10).fit(X)
 

    # #############################################################################
    # Plot result
    # Clusters are given in the labels_ attribute
    labels = cls_dbscan.labels_
    print (Counter(labels))

    print("labels: ")
    print(labels)
    colors = set_colors(labels)
    cluster_labels = set_legend()
    
    plt.scatter(X[:,0], X[:,1], color=colors,  edgecolors='none')
 
    '''
   
        # Plot result
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(labels):
        ax.scatter(X[labels == l, 0], X[labefor l in np.unique(labels):ls == l, 1], 
                color=plt.cm.jet(np.float(l) / np.max(labels + 1)),label=cluster_labels[l],
                s=20, edgecolor='k')
   
    '''
    plt.title('Without connectivity constraints')
    plt.gca().legend(scatterpoints=1, loc='top left')
    plt.grid(True)    
    plt.show()
    
    '''
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)

    plt.show()
    '''
   

    # Calling main function
if __name__=="__main__":
    main()
