import os
import glob
import shutil

# names: [ "scaffolds","worker","boots","hardhat","safety_vest","hook","robot_dog","opened_hatch","closed_hatch"]
# label_mapping = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'7', '7':'8',}   
# names: ['0', '1', '2', '3']
# names: ['scaffolds', 'worker', 'hardhat', 'hook']

# names: [ "scaffolds","worker","boots","hardhat","safety_vest","hook","robot_dog","opened_hatch","closed_hatch"]
'''
0 , 1 , 2 ,3 ,4 ,5 ,6 ,7 ,8 

지우고 2 , 4 , 6 

0 , 1 , 3 , 5 , 7 , 8

3 : 2 , 5 : 3 , 7 : 4 , 8 : 5 
 
0 , 1 , 2 , 3 , 4  ,5
names: [ "scaffolds","worker","hardhat","hook","opened_hatch","closed_hatch"]
'''
# names: ['1_scaffold', '2_worker', '3_hard hat', '4_hook', '5_opend_hatch', '6_closed_hatch']
# 2 : 3
# 3 : 5
# 4 : 7
# 5 : 8

folder_path = "F:\yolov7-main\coco128/train\labels/"
save_folder_path = "F:\yolov7-main\coco128/train\labels/"

label_mapping = {"3":"2" , "5":"3" , "7":"4" , "8":"5" }   
# label_mapping = {'2': '3', '3':'5', '4':'7','5':'8',}    
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split(' ')

        if parts[0] in label_mapping:
            parts[0] = label_mapping[parts[0]]

        new_line = ' '.join(parts)
        new_lines.append(new_line)

    save_file_path = os.path.join(save_folder_path, os.path.basename(file_path))
    with open(save_file_path, 'w') as f:
        for line in new_lines:
            f.write(line)
