import os
import shutil

source_folder = 'C:/323/'  # 수정 필요
destination_folder = 'F:/coco128_last/val2017/labels/'  # 수정 필요

def find_files_with_label(label, folder_path):
    matching_files = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0 and parts[0] == label:
                        matching_files.append(full_path)
                        break
    
    return matching_files

label_to_find = "2"
matching_files = find_files_with_label(label_to_find, source_folder)

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

if matching_files:
    print(f"label이 {label_to_find}인 파일들을 {destination_folder}에 저장 중...")
    for source_file in matching_files:
        filename = os.path.basename(source_file)
        destination_file = os.path.join(destination_folder, filename)
        shutil.copy(source_file, destination_file)
    print("파일 복사 완료")
else:
    print(f"label이 {label_to_find}인 파일은 없습니다.")
