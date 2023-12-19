
import os

def find_empty_txt_files(folder_path="."):
    empty_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if not content.strip():
                            empty_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return empty_files

if __name__ == "__main__":
    folder_path = "F:/★NIPA Detector/1.Datasets/1.Scaffolds/6. 2023-11-25 Dataset/valid\labels/"  

    empty_files = find_empty_txt_files(folder_path)

    if empty_files:
        print("Empty txt files found:")
        for file in empty_files:
            print(file)
    else:
        print("No empty txt files found.")


# import os

# def find_corrupted_files(folder_path="."):
#     corrupted_files = []

#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".txt"):
#                 file_path = os.path.join(root, file)
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                         if "corrupted" in content.lower():
#                             corrupted_files.append(file_path)
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")

#     return corrupted_files

# if __name__ == "__main__":
#     folder_path = "D:/yolov7/coco128/test/labels/"  # 폴더 경로를 여기에 지정하세요.

#     corrupted_files = find_corrupted_files(folder_path)

#     if corrupted_files:
#         print("Corrupted files found:")
#         for file in corrupted_files:
#             print(file)
#     else:
#         print("No corrupted files found.")