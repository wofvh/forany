import os
import glob
import shutil

# 텍스트 파일이 있는 폴더 경로를 지정합니다.
folder_path = 'C://alldata//store//valid//labels'

# 빈 파일을 저장할 폴더 경로를 지정합니다.
save_folder_path = 'C://alldata//save03val'

# 폴더 내의 모든 텍스트 파일 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 빈 파일일 경우 다른 경로에 저장합니다.
    if len(lines) == 0:
        save_file_path = os.path.join(save_folder_path, os.path.basename(file_path))
        shutil.copyfile(file_path, save_file_path)
