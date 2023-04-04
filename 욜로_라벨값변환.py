# import os
# import glob


# folder_path = '폴더 경로'


# # 폴더 내의 모든 텍스트 파일 경로를 가져옵니다.
# file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

# for file_path in file_paths:
#     with open(file_path, 'r') as f:
#         lines = f.readlines()

#     with open(file_path, 'w') as f:
#         for line in lines:
#             parts = line.split(' ')

#             if parts[0] == '0':
#                 parts[0] = '26'

#             new_line = ' '.join(parts)
#             f.write(new_line)



import os
import glob

# 텍스트 파일이 있는 폴더 경로를 지정합니다.
folder_path = 'C://alldata//test01'

# 수정된 파일을 저장할 폴더 경로를 지정합니다.
save_folder_path = 'C://alldata//save01'

# 폴더 내의 모든 텍스트 파일 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 수정된 파일을 다른 경로에 저장합니다.
    save_file_path = os.path.join(save_folder_path, os.path.basename(file_path))
    with open(save_file_path, 'w') as f:
        for line in lines:
            parts = line.split(' ')

            if parts[0] == '1':
                parts[0] = '26'

            new_line = ' '.join(parts)
            f.write(new_line)

