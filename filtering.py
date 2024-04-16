import os
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
            else:
                if transportation_mode in line:
                    res.append(user_path)
    return res

# get current directory
dataset_dir = os.path.join(os.getcwd(), 'Geolife Trajectories 1.3/Data')

# filter out users with label file
users_paths = extract_label_users(dataset_dir)

# filter out users with transportation mode of car
users_car = extract_user_transportation_mode(users_paths, 'car')
print(users_car[0])

# Process those time ranges and match them up with trajectories
