
import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import geopandas as gpd

from datetime import datetime
from shapely.geometry import Point, MultiPolygon, Polygon

from filtering import *

####################################################################################################
##################################### Automatic Extraction #########################################
####################################################################################################


########################################### K-means ################################################

def visualize_clusters(df_filtered_china):
    """
    Visualize the clustering of locations in China using K-means.

    Args:
    - df_filtered_china (DataFrame): Filtered DataFrame containing only locations within China.
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=10).fit(df_filtered_china[['Longitude', 'Latitude']])
    labels = kmeans.labels_

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.scatter(df_filtered_china.loc[labels == i, 'Longitude'], 
                    df_filtered_china.loc[labels == i, 'Latitude'], 
                    label=f'Cluster {i+1}')
    plt.title('Clustering of Cities in China')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.legend()
    plt.show()


dataset_path = 'test.csv'

df_filtered_china = filter_china_locations(dataset_path)

visualize_clusters(df_filtered_china)

