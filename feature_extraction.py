import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os

from filtering import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Path to the original data directory
original_data_dir = os.path.join(os.getcwd(), 'Geolife Trajectories 1.3/Data')

# Path to the new data directory
new_data_dir = os.path.join(os.getcwd(), 'Geolife Trajectories 1.3/Data_new')

# csv file containing datapoints with features driver ID, longitude, latitude, altitude, distance, and speed
dataset_path = 'data/china_trajectory_dataset.csv'

# Dataset Preprocessing

# step 1: Filter out users that have a label file
users_paths = extract_label_users(original_data_dir)

# step 2: Filter out users that have a transportation mode of car
users_car = extract_user_transportation_mode(users_paths, 'car')

# step 3: Filter out so that labels file so that it ONLY has transportation mode of car

# Assuming users_car contains paths to user directories with car or taxi transportation mode
for user_dir in users_car:
    # Get the relative path within the original data directory
    relative_path = os.path.relpath(user_dir, original_data_dir)
    
    # Create the corresponding directory structure in the new data directory
    new_user_dir = os.path.join(new_data_dir, relative_path)
    os.makedirs(new_user_dir, exist_ok=True)
    
    # Copy everything from the original directory to the new directory
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        if os.path.isfile(item_path):
            shutil.copy(item_path, new_user_dir)
        elif os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(new_user_dir, item))
    
    # Filter and create the new labels file
    label_file_path = os.path.join(new_user_dir, 'labels.txt')
    filtered_lines = filter_labels_file(label_file_path)
    create_filtered_labels_file(new_user_dir, filtered_lines)
    

# step 4: Filter out specific cities / locations (China)
df_filtered_china = filter_china_locations(dataset_path)

####################################################################################################
##################################### Automatic Extraction #########################################
####################################################################################################


########################################### K-means ################################################
# cluster filtered dataset into 30 clusters (representing the 30 cities in China the datapoints are taken from)
kmeans = KMeans(n_clusters=30).fit(df_filtered_china[['Longitude', 'Latitude']])
label = kmeans.labels_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(df_filtered_china[label == i , 1] , df_filtered_china[label == i , 2] , label = i)
plt.legend()
plt.show()


############################################# PCA ##################################################
std_scaler = StandardScaler()
# normalize data points so that features have zero mean and unit variance
scaled_df = std_scaler.fit_transform(df_filtered_china)

# find the optimal number of PCs needed by measuring the explained variance ratio for each pc_num
# pc_nums that have an explained variance ratio of at least 95% are optimal
pc_nums = np.arange(14)

var_ratio = []
for num in pc_nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

# Analyzing the Change in Explained Variance Ratio
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

# retrieve optimal pc_num value
optimal_pc_num = None # to be filled after analyzing graph


