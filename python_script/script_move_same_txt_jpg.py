import os
import shutil


source_folder = 'F:/total_dataset/datasets_need_annotation/total_scaffold/'    # 수정 필요
destination_folder = 'F:/total_dataset/datasets_need_annotation/train/' # 수정 필요

def merge_matching_files(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    copied_files = set()  # 이미 복사한 파일들을 추적하기 위한 집합
    
    for filename in os.listdir(source):
        base_name, extension = os.path.splitext(filename)
        if extension == '.txt':
            jpg_filename = base_name + '.jpg'
            txt_path = os.path.join(source, filename)
            jpg_path = os.path.join(source, jpg_filename)
            if os.path.exists(jpg_path) and filename not in copied_files:
                destination_txt = os.path.join(destination, filename)
                destination_jpg = os.path.join(destination, jpg_filename)
                shutil.copy(txt_path, destination_txt)
                shutil.copy(jpg_path, destination_jpg)
                copied_files.add(filename)

merge_matching_files(source_folder, destination_folder)
print("파일 합치기 완료")
