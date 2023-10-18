import os
import glob
import shutil
'''
0Fire_Extinguisher  ,Fire_Extinguisher

1Flammable_Material ,Flammable_material

2Welding Equipment

3Worker_with_PPE , Worker_with_PPE

4welding_machine
  
5Circular_Saw

6Fire_prevention_Net
# names: ['boots', 'hard hat', 'person', 'robot_dog', 'vest']
names: [ 'Safety Hook', 'Person', 'Scaffold', 'Hard Hat' ]
names: ['0', '1', '3', '5', '9']

 '''
# names: ['Fire_Extinguisher', 'Flammable_Material', 'Welding Equipment', 'Worker', 'welding_machine','Circular_Saw','Fire_prevention_Net']

# names: [ "scaffolds","worker","boots","hardhat","safety_vest","hook","robot_dog","opened_hatch","closed_hatch"]

folder_path = "G:\Synthetic Scaffolding/labels/"  
save_folder_path = "G:\Synthetic Scaffolding/passing/"   

label_mapping = {'0':'5','1':'1','2':'0','3':'3'}  
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
