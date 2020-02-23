import os

def create_folder_if_not_exits(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)