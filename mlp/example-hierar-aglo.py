import time as time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  

# Function importing Dataset
def importdata():
    py_class = ['Distance_Feature','Speeding_Feature']
    py_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\drivers_data.csv', header=None,  names=py_class)
    
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
    cls_ag =  AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average').fit(X)
 

    # #############################################################################
    # Plot result
    # Clusters are given in the labels_ attribute
    labels = cls_ag.labels_
    print (Counter(labels))

    print("labels: ")
    print(labels)
    colors = set_colors(labels)
    cluster_labels = set_legend()

    '''
    plt.scatter(X[:,0], X[:,1], color=colors,  edgecolors='none')
 
    
   
        # Plot result
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(labels):
        ax.scatter(X[labels == l, 0], X[labefor l in np.unique(labels):ls == l, 1], 
                color=plt.cm.jet(np.float(l) / np.max(labels + 1)),label=cluster_labels[l],
                s=20, edgecolor='k')
   
    
    plt.title('Without connectivity constraints')
    plt.gca().legend(scatterpoints=1, loc='top left')
    plt.grid(True)    
    plt.show()
    '''

    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************

    # visualize standardized vs. untouched dataset with PCA performed
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 7))

    legend_labels = ''
    #for l, c, m in zip(range(0, 1), colors, ('^', 's', 'o')):
    p= ax1.scatter(X[:,0], X[:,1],
                color=colors,
                label=legend_labels,
                alpha=0.5 
                )
    

    ax1.set_title('Training dataset after PCA')
     

    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid()

    plt.tight_layout()

    plt.show()

    # ****************************************************************************************
    # ****************************************************************************************
    # ****************************************************************************************

    
    '''
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)

    plt.show()
    '''
   

    # Calling main function
if __name__=="__main__":
    main()