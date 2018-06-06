# Load library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics

def importdata():

    train_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\mercari_train.tsv', sep='\t')
    test_data = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\mercari_test.tsv', sep='\t')

    train_data = train_data.sample(n=1000) # Get randomly 1000 records
    test_data = test_data.sample(n=1000) # Get randomly 1000 records

    #py_data = py_data.iloc[:,spec_cols] # Get only specific columns / elements
    print("\nTrain Data **********************************************************\n")
     
    print(train_data.head(5))
    print(train_data.shape)

    print("\nTest Data ***********************************************************\n")
 
    print(test_data.head(5))
    print(test_data.shape)

    return train_data, test_data

def clean_df(df):
    df = df.fillna({'category_Name':'', 'brand_name':''})
    return df

def get_evaluation(X, y, labels, algoname):

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    labels_true = y

    print('\n********* Evaluation for ALGORITHM {0} *********************************', algoname)
    print('\nEstimated number of clusters: %d' % n_clusters_)
    print("\nHomogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("\nCompleteness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("\nV-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("\nAdjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("\nAdjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("\nSilhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

def main():

    # Building Phase
    #GET DATA
    X_train, X_test = importdata()


    #CLEAN DATA
    X_train = clean_df(X_train)
    X_test = clean_df(X_test)

    #SPLIT DATA

    
 
# Calling main function
if __name__=="__main__":
    main()

