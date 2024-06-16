import os
import shutil

# List of folder names to be copied
folder_names_w2_cam1 = [
    "1014", "1081", "1100", "1143", "1162", "1191", "1365", "1647", "1676", "1775", 
    "1962", "2112", "2143", "2330", "2317", "232", "2513", "259", "382", "412",
    "427", "485", "504", "603", "837", "874"
]

folder_names_w2_cam2 = [
    "459", "913", "913", "2034", "2195", "140", "913", "2102", "459", "2117", 
    "913", "459", "459", "220", "1148", "459", "2169", "913", "1417", "2102",
    "1395", "913", "1723", "459", "2169", "913"
]

folder_names_w1_cam1 = [
    "1067", "1282", "132", "1446", "1450", "1469", "1549", "1601", "174", "206", 
    "28", "440", "478", "530", "566", "755", "900", "903", "907"
]

folder_names_w1_cam2 = [
    "1417", "968", "496", "362", "1417", "1351", "660", "1448", "884", "363", 
    "1417", "1351", "761", "1146", "363", "154", "660", "164", "1417"
]

# Define the source and destination directories
source_dir = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/exit_images/outroi'
destination_dir = '/home/meiqiu@ads.iu.edu/Mei_all/feature_extraction/weaving_in_outroi/weaving1/cam2_outroi'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Copy each folder
for folder_name in folder_names_w1_cam2:
    src_path = os.path.join(source_dir, folder_name)
    dest_path = os.path.join(destination_dir, folder_name)
    
    # Check if the folder exists in the source directory
    if os.path.exists(src_path):
        if not os.path.exists(dest_path):
        # Copy the folder
            shutil.copytree(src_path, dest_path)
            print(f"Folder {folder_name} copied successfully.")
    else:
        print(f"Folder {folder_name} does not exist in the source directory.")
