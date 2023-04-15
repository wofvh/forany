import os
import glob
import shutil

# 텍스트 파일이 있는 폴더 경로를 지정
folder_path = 'C://alldata//test02'

# 수정된 파일을 저장할 폴더 경로를 지정
save_folder_path = 'C://alldata//save00'

# 폴더 내의 모든 텍스트 파일 경로를 가져옵니다.
file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

for file_path in file_paths:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    save_file = False
    new_lines = []
    for line in lines:
        parts = line.split(' ')

        if parts[0] == '3':
            parts[0] = '29'
            save_file = True

        new_line = ' '.join(parts)
        new_lines.append(new_line)

    # 수정된 파일을 다른 경로에 저장합니다.
    if save_file:
        save_file_path = os.path.join(save_folder_path, os.path.basename(file_path))
        with open(save_file_path, 'w') as f:
            for line in new_lines:
                f.write(line)





'''

with open('C://alldata//test00//frame2553.txt', 'r') as f:
    lines = f.readlines()
    # 파일 내 모든 라인을 읽어들입니다.

with open('C://alldata//test00//frame255333.txt', 'w') as f:
    for line in lines:
        parts = line.split(' ')
        # 라벨 숫자와 좌표값을 분리합니다.

        if parts[0] == '3':
            parts[0] = '26'
            # 라벨 숫자를 26으로 변경합니다.

        new_line = ' '.join(parts)
        f.write(new_line)
        # 변경된 라벨 숫자와 기존 좌표값을 다시 합쳐서 파일에 씁니다.
'''
