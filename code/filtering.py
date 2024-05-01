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

from geopy.distance import geodesic


def extract_label_users(dataset_dir):
    '''
    [input]:
        - dataset_dir (str): path of Geolife Trajectories 1.3/Data 
    [output]
        - res (list): list of strings containing the path of user directories that have labels.txt
    '''

    # extract user files storing trajectories
    user_dirs = os.listdir(dataset_dir)
    label_file_name = 'labels.txt'
    res = []

    # check which user directories contain labels file
    for user_dir in user_dirs:
        user_dir_path = os.path.join(dataset_dir, user_dir)
        label_file_path = os.path.join(user_dir_path, label_file_name)
        if os.path.exists(label_file_path):
            res.append(user_dir_path)
    
    return res

def extract_user_transportation_mode(users_paths, transportation_mode):
    '''
    [input]:
        - users_paths (list): list containing strings of paths to user files that contain 'labels.txt'
        - transportation_mode (str): string that can take form of 'car', 'taxi', etc
    [output]
        - res (list): list of strings containing the path of user directories that have transportation mode 'transportation_mode'
    '''
    res = []
    for user_path in users_paths:
        label_file_path = os.path.join(user_path, 'labels.txt')
        file = open(label_file_path, 'r')
        content = file.readlines()[1:]
        for line in content:
            # if transportation_mode == 'car' or transportation_mode == 'taxi':
            #     if 'car' in line or 'taxi' in line:
            #         res.append(user_path)
            #         break
            # else:
            if transportation_mode in line:
                res.append(user_path)
                break
    return res

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

def filter_labels_file(label_file_path):
    """
    Filter the lines in the labels.txt file to only include those with 'car' or 'taxi' as the transportation mode.
    
    Args:
    - label_file_path (str): Path to the labels.txt file
    
    Returns:
    - filtered_lines (list): List of filtered lines
    """
    filtered_lines = []
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        # Append the header line
        filtered_lines.append(lines[0])
        for line in lines[1:]:
            # if 'car' in line or 'taxi' in line:
            if 'car' in line:
                filtered_lines.append(line)
    return filtered_lines

def create_filtered_labels_file(user_dir, filtered_lines):
    """
    Create a new labels_new.txt file in the user directory and write the filtered lines into it.
    
    Args:
    - user_dir (str): Path to the user directory
    - filtered_lines (list): List of filtered lines from the labels.txt file
    """
    new_file_path = os.path.join(user_dir, 'labels_new.txt')
    with open(new_file_path, 'w') as file:
        for line in filtered_lines:
            file.write(line)

def convert_to_days_since_1899(date_time_str):
    """
    Convert the date and time string to the number of days since 12/30/1899.
    
    Args:
    - date_time_str (str): Date and time string in the format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
    - days_since_1899 (float): Number of days since 12/30/1899
    """
    reference_date = datetime(1899, 12, 30)
    date_time_obj = datetime.strptime(date_time_str, '%Y/%m/%d %H:%M:%S')
    delta = date_time_obj - reference_date
    return delta.days + delta.seconds / (24 * 3600)  # Convert seconds to days

def update_labels_file(labels_file_path):
    """
    Update the labels_new.txt file with start and end times represented as days since 12/30/1899,
    delete the first row, and separate the two values by a comma.
    
    Args:
    - labels_file_path (str): Path to the labels_new.txt file
    """
    with open(labels_file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    # Skip the first line (header row)
    for line in lines[1:]:
        # Split the line by tab and extract start and end times
        start_time_str, end_time_str, _ = line.strip().split('\t')
        
        # Convert start and end times to days since 12/30/1899
        start_time_days = convert_to_days_since_1899(start_time_str)
        end_time_days = convert_to_days_since_1899(end_time_str)
        
        # Update the line with start and end times represented as days since 12/30/1899
        updated_line = f"{start_time_days:.6f},{end_time_days:.6f}\n"
        updated_lines.append(updated_line)

    # Write the updated lines back to the file
    with open(labels_file_path, 'w') as file:
        file.writelines(updated_lines)

def update_labels_files_in_directory(directory_path):
    """
    Update all labels_new.txt files in the specified directory and its subdirectories.
    
    Args:
    - directory_path (str): Path to the directory containing labels_new.txt files
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file == 'labels_new.txt':
                labels_file_path = os.path.join(root, file)
                update_labels_file(labels_file_path)

def filter_plt_files(user_folder):
    """
    Filter .plt files in the user's trajectory folder based on data_new.txt files.

    Args:
    - user_folder (str): Path to the user's folder containing trajectory data.
    """
    new_path = os.path.join(user_folder, 'Trajectory_new')
    os.makedirs(new_path)

    # Iterate through data_new.txt files
    for data_file_name in os.listdir(user_folder):
        if data_file_name.startswith("labels_new") and data_file_name.endswith(".txt"):
            data_file_path = os.path.join(user_folder, data_file_name)
            start_end_times = []

            # Read start and end times from data_new.txt
            with open(data_file_path, 'r') as data_file:
                # next(data_file)  # Skip header
                for line in data_file:
                    start, end = line.strip().split(',')
                    start_end_times.append((float(start), float(end)))

            # Iterate through .plt files
            trajectory_folder = os.path.join(user_folder, "Trajectory")
            for plt_file_name in os.listdir(trajectory_folder):
                if plt_file_name.endswith(".plt"):
                    plt_file_path = os.path.join(trajectory_folder, plt_file_name)
                    filtered_rows = []

                    # Filter rows based on start and end times
                    with open(plt_file_path, 'r') as plt_file:
                        for i in range(6):
                            next(plt_file)
                        for line in plt_file:
                            parts = line.strip().split(',')
                            timestamp = float(parts[-3])
                            for start, end in start_end_times:
                                if start <= timestamp <= end:
                                    filtered_rows.append(line)
                                    break
                    
                    # Write filtered rows to a new .plt file
                    if filtered_rows:
                        filtered_plt_file_path = os.path.join(new_path, f"{plt_file_name[:-4]}_filtered.plt")
                        with open(filtered_plt_file_path, 'w') as filtered_plt_file:
                            filtered_plt_file.writelines(filtered_rows)


##### 4/19/2024 #####

def calculate_average_coordinates_altitude(plt_file_path):
    """
    Calculate the average latitude, average longitude, and average altitude from a .plt file.

    Args:
    - plt_file_path (str): Path to the .plt file.

    Returns:
    - average_latitude (float): Average latitude.
    - average_longitude (float): Average longitude.
    - average_altitude (float): Average altitude.
    """
    total_latitude = 0
    total_longitude = 0
    total_altitude = 0
    total_rows = 0

    with open(plt_file_path, 'r') as plt_file:
        for line in plt_file:
            parts = line.strip().split(',')
            latitude = float(parts[0])
            longitude = float(parts[1])
            altitude = float(parts[3])

            total_latitude += latitude
            total_longitude += longitude
            total_altitude += altitude
            total_rows += 1

    average_latitude = total_latitude / total_rows
    average_longitude = total_longitude / total_rows
    average_altitude = total_altitude / total_rows

    return average_latitude, average_longitude, average_altitude

def calculate_total_time(plt_file_path):
    """
    Calculate the total time duration for a filtered .plt file.

    Args:
    - plt_file_path (str): Path to the filtered .plt file.

    Returns:
    - total_time (float): Total time duration.
    """
    timestamps = []

    with open(plt_file_path, 'r') as plt_file:
        for line in plt_file:
            parts = line.strip().split(',')
            timestamp = float(parts[-3])
            timestamps.append(timestamp)

    total_time = timestamps[-1] - timestamps[0]
    return total_time


def calculate_distance_speed(plt_file_path):
    """
    Calculate the total distance and average speed from a .plt file.

    Args:
    - plt_file_path (str): Path to the .plt file.

    Returns:
    - total_distance (float): Total distance in kilometers.
    - average_speed (float): Average speed in kilometers per hour.
    """
    distances = []
    timestamps = []
    total_distance = 0
    total_time_seconds = 0

    epsilon = 1e-6

    with open(plt_file_path, 'r') as plt_file:
        # lines = plt_file.readlines()[6:]  # Skip header lines
        # for line in lines:
        for line in plt_file:
            parts = line.strip().split(',')
            latitude = float(parts[0])
            longitude = float(parts[1])
            altitude = float(parts[3])
            timestamp = float(parts[-3])
            timestamps.append(timestamp)

            if len(distances) > 0:
                prev_lat, prev_lon, prev_alt, prev_timestamp = distances[-1]
                distance = geodesic((prev_lat, prev_lon), (latitude, longitude)).kilometers
                time_diff_seconds = timestamp - prev_timestamp
                total_distance += distance
                total_time_seconds += time_diff_seconds
                # speeds = distance / time_diff_seconds * 3600  # Convert from meters per second to kilometers per hour
                distances.append((latitude, longitude, altitude, timestamp))
            else:
                distances.append((latitude, longitude, altitude, timestamp))

    average_speed = total_distance / (total_time_seconds + epsilon) * 3600  # Convert from meters per second to kilometers per hour

    return total_distance, average_speed


def process_filtered_plt_files(directory_path, output_csv_path):
    """
    Process filtered .plt files in Trajectory_new directories for every user and store average
    latitude, longitude, altitude, and total time in a new CSV file.

    Args:
    - directory_path (str): Path to the root directory containing Trajectory_new directories for each user.
    - output_csv_path (str): Path to the output CSV file.
    """
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # csv_writer.writerow(["User", "Average Latitude", "Average Longitude", "Average Altitude", "Total Time"])
        csv_writer.writerow(["User", "Latitude", "Longitude", "Average Altitude", "Total Distance", "Average Speed", "Total Time"])

        for user_folder in os.listdir(directory_path):
            user_folder_path = os.path.join(directory_path, user_folder)
            if os.path.isdir(user_folder_path):
                trajectory_new_path = os.path.join(user_folder_path, "Trajectory_new")
                if os.path.isdir(trajectory_new_path):
                    print(f"Processing Trajectory_new directory for user: {user_folder}")
                    for plt_file in os.listdir(trajectory_new_path):
                        if plt_file.endswith(".plt"):
                            plt_file_path = os.path.join(trajectory_new_path, plt_file)
                            print(f"Processing filtered .plt file: {plt_file_path}")
                            average_latitude, average_longitude, average_altitude = calculate_average_coordinates_altitude(plt_file_path)
                            total_time = calculate_total_time(plt_file_path)

                            # add speed and distance calculation
                            total_distance, average_speed = calculate_distance_speed(plt_file_path)
                            csv_writer.writerow([user_folder, average_latitude, average_longitude, average_altitude, total_distance, average_speed, total_time])

                            # csv_writer.writerow([user_folder, average_latitude, average_longitude, average_altitude, total_time])
                            print("Finished processing filtered .plt file.")
                else:
                    print(f"No Trajectory_new directory found for user: {user_folder}")


