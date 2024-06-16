import json
import os
import shutil

# Define the paths
json_file_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_1_matches.json'
new_folder_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/entry_images/matched_inroi'

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract the first folder in the list
first_folder_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/entry_images/inroi'  # Assuming the JSON contains a list of folder paths

# Create the new folder if it doesn't exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
# print("data",data[0]['Folder1_Vehicle'])
# Copy subfolders from the first folder to the new folder
# for item in data:
for j in range(len(data)):
    s = os.path.join(first_folder_path, data[j]['Folder1_Vehicle'])
    d = os.path.join(new_folder_path, data[j]['Folder1_Vehicle'])
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Subfolders from {first_folder_path} have been copied to {new_folder_path}")
