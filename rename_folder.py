import os

# Define the directory where the subfolders are located
directory_path = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/cam2_inroi'
# List of new folder names corresponding to each original name by index
# folder_names_w2_cam1 = [
#     "1014", "1081", "1100", "1143", "1162", "1191", "1365", "1647", "1676", "1775",
#     "1962", "2112", "2143", "2330", "2317", "232", "2513", "259", "382", "412",
#     "427", "485", "504", "603", "837", "874"
# ]
# # List of original folder names (to be renamed)
# folder_names_w2_cam2 = [
#     "459", "913", "913", "2034", "2195", "140", "913", "2102", "459", "2117",
#     "913", "459", "459", "220", "1148", "459", "2169", "913", "1417", "2102",
#     "1395", "913", "1723", "459", "2169", "913"
# ]
folder_names_w1_cam1 = [
    "1067", "1282", "132", "1446", "1450", "1469", "1549", "1601", "174", "206", 
    "28", "440", "478", "530", "566", "755", "900", "903", "907"
]

folder_names_w1_cam2 = [
    "1417", "968", "496", "362", "1417", "1351", "660", "1448", "884", "363", 
    "1417", "1351", "761", "1146", "363", "154", "660", "164", "1417"
]


# Rename the subfolders one by one based on the index
for old_name, new_name in zip(folder_names_w1_cam2, folder_names_w1_cam1):
    old_path = os.path.join(directory_path, old_name)
    new_path = os.path.join(directory_path, new_name)
    
    # Check if the old folder exists and the new folder name does not exist to avoid overwriting
    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        if not os.path.exists(old_path):
            print(f"Folder {old_name} does not exist.")
        if os.path.exists(new_path):
            print(f"Folder {new_name} already exists, skipping renaming.")

