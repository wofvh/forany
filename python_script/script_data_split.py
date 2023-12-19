import os
import random
from sklearn.model_selection import train_test_split
import shutil

def split_data(input_folder, output_folder, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05):
    # 폴더 내의 이미지 파일 및 좌표값 파일 가져오기
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # 이미지 및 텍스트 파일을 랜덤하게 섞음
    random.shuffle(image_files)
    random.shuffle(txt_files)

   # 데이터 분할
    train_images, test_val_images = train_test_split(image_files, test_size=1 - train_ratio, random_state=42)
    val_images, test_images = train_test_split(test_val_images, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

    for img_file in train_images:
        shutil.copy(os.path.join(input_folder, img_file), os.path.join(output_folder, 'train', img_file))
        shutil.copy(os.path.join(input_folder, img_file.replace('.jpg', '.txt')), os.path.join(output_folder, 'train', img_file.replace('.jpg', '.txt')))
    
    for img_file in val_images:
        shutil.copy(os.path.join(input_folder, img_file), os.path.join(output_folder, 'val', img_file))
        shutil.copy(os.path.join(input_folder, img_file.replace('.jpg', '.txt')), os.path.join(output_folder, 'val', img_file.replace('.jpg', '.txt')))

    for img_file in test_images:
        shutil.copy(os.path.join(input_folder, img_file), os.path.join(output_folder, 'test', img_file))
        shutil.copy(os.path.join(input_folder, img_file.replace('.jpg', '.txt')), os.path.join(output_folder, 'test', img_file.replace('.jpg', '.txt')))

if __name__ == "__main__":
    input_folder = "D:/last/train/images/"
    output_folder = "D:\last/train/data/"
     
    #makr folder
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)

    # 함수 호출
    split_data(input_folder, output_folder)

    print("done.")
