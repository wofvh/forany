import os
import shutil

source_folder = "C:/data/output/train_50/images/"  
destination_folder = "C:/data/output/passing/"  
new_name_prefix = "h"  

def rename_and_copy_images(source_folder, destination_folder, new_prefix):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):  
            old_path = os.path.join(source_folder, filename)
            new_filename = f"{new_prefix}_{filename}"
            new_path = os.path.join(destination_folder, new_filename)
            shutil.copy(old_path, new_path)
            print(f"Copied and renamed {filename} to {new_filename} in {destination_folder}")

rename_and_copy_images(source_folder, destination_folder, new_name_prefix)


import os
import shutil

source_folder = "C:/data/output/train_50/images/"  
destination_folder = "C:/data/output/passing_txt/"  
new_name_prefix = "hh"  

def rename_and_copy_txt_files(source_folder, destination_folder, new_prefix):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.txt'):  
            old_path = os.path.join(source_folder, filename)
            new_filename = f"{new_prefix}_{filename}"
            new_path = os.path.join(destination_folder, new_filename)
            shutil.copy(old_path, new_path)
            print(f"Copied and renamed {filename} to {new_filename} in {destination_folder}")

rename_and_copy_txt_files(source_folder, destination_folder, new_name_prefix)
