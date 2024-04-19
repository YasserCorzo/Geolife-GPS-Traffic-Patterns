
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


# def filter_china_locations(dataset_path):
#     """
#     Filters trajectories that have locations in China

#     [input]
#         - dataset_path (str): path to file containing trajectory dataset with features driver ID, longitude, latitude, distance, and speed
    
#     [output]
#         - dataset (pd.Dataframe): pandas dataframe containing filtered trajectories
#     """

#     # read csv file and convert contents to dataframe
#     df = pd.read_csv(dataset_path)

#     # read shapefile containing boundary of china
#     gdf = gpd.read_file('data/stanford_china_shapefile/china_boundary.shp')
#     geometry = gdf['geometry'][0]
    
#     # extract coordinates 
#     df_coord = df[['Longitude', 'Latitude']]
#     coordinates = []
#     for i in range(len(df_coord)):
#         coordinate = (df_coord[i, 'Longitude'], df_coord[i, 'Latitude'])
#         coordinates.append(coordinate)
    
#     # check which coordinates are in china and extract them
#     valid_coords_index = []
#     for i, coord in enumerate(coordinates):
#         point = Point(coord[0], coord[1])
#         if point.within(geometry):
#             valid_coords_index.append(i)
    
#     # filter datapoints in dataframe that have location in china
#     df_filtered = df.iloc[valid_coords_index, :]

#     return df_filtered


def filter_china_locations(dataset_path):
    """
    Filter the dataset to include only locations within China.

    Args:
    - dataset_path (str): Path to the dataset CSV file.

    Returns:
    - df_filtered_china (DataFrame): Filtered DataFrame containing only locations within China.
    """
    # Load the dataset into a DataFrame
    df = pd.read_csv(dataset_path)

    # Define boundaries for China
    min_longitude, max_longitude = 73.554, 135.083
    min_latitude, max_latitude = 3.86, 53.55

    # Filter locations within China
    df_filtered_china = df[(df['Longitude'] >= min_longitude) & (df['Longitude'] <= max_longitude) &
                           (df['Latitude'] >= min_latitude) & (df['Latitude'] <= max_latitude)]

    return df_filtered_china


# Example usage:
dataset_path = 'test.csv'
# df_filtered_china = filter_china_locations(dataset_path)

# step 4: Filter out specific cities / locations (China)
df_filtered_china = filter_china_locations(dataset_path)

####################################################################################################
##################################### Automatic Extraction #########################################
####################################################################################################


########################################### K-means ################################################
# cluster filtered dataset into 30 clusters (representing the 30 cities in China the datapoints are taken from)
# kmeans = KMeans(n_clusters=30).fit(df_filtered_china[['Longitude', 'Latitude']])
# label = kmeans.labels_
# u_labels = np.unique(label)

# for i in u_labels:
#     plt.scatter(df_filtered_china[label == i , 1] , df_filtered_china[label == i , 2] , label = i)
# plt.legend()
# plt.show()

def visualize_clusters(df_filtered_china):
    """
    Visualize the clustering of locations in China using K-means.

    Args:
    - df_filtered_china (DataFrame): Filtered DataFrame containing only locations within China.
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=30).fit(df_filtered_china[['Longitude', 'Latitude']])
    labels = kmeans.labels_

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for i in range(30):
        plt.scatter(df_filtered_china.loc[labels == i, 'Longitude'], 
                    df_filtered_china.loc[labels == i, 'Latitude'], 
                    label=f'Cluster {i}')
    plt.title('Clustering of Locations in China')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

# Example usage:
visualize_clusters(df_filtered_china)


