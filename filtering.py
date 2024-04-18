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


