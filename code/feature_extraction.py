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

# filter out users with label file
users_paths = extract_label_users(original_data_dir)

# filter out users with transportation mode of car / taxi
users_car = extract_user_transportation_mode(users_paths, 'car')
# print(users_car)


# Path to the new data directory
new_data_dir = os.path.join(os.getcwd(), 'Geolife Trajectories 1.3/Data_new')

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


# Process those time ranges and match them up with trajectories

# Example usage:
directory_path = "Geolife Trajectories 1.3/Data_new"  # Replace with the path to your directory
update_labels_files_in_directory(directory_path)

# Example usage:
# user_folder = 'Geolife Trajectories 1.3/Data_new/010'
directory_path = "Geolife Trajectories 1.3/Data_new"
# for user_folder in directory_path:
    # filter_plt_files(user_folder)
    # print(user_folder)

for item in os.listdir(directory_path):
    user_folder = os.path.join(directory_path, item)
    # print(item_path)
    filter_plt_files(user_folder)

# Example usage:
root_directory_path = "Geolife Trajectories 1.3/Data_new"
output_csv_path = "average_coordinates_altitude_time.csv"
process_filtered_plt_files(root_directory_path, output_csv_path)

