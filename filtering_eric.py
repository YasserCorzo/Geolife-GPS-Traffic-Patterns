import os
import shutil
import numpy as np

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
            if transportation_mode == 'car' or transportation_mode == 'taxi':
                if 'car' in line or 'taxi' in line:
                    res.append(user_path)
                    break
            else:
                if transportation_mode in line:
                    res.append(user_path)
                    break
    return res



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
            if 'car' in line or 'taxi' in line:
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

from datetime import datetime

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

# Example usage:
directory_path = "Geolife Trajectories 1.3/Data_new"  # Replace with the path to your directory
update_labels_files_in_directory(directory_path)



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

# Example usage:
user_folder = 'Geolife Trajectories 1.3/Data_new/010'
filter_plt_files(user_folder)


