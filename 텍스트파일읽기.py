import os

def get_txt_files_in_folder(folder_path):
    txt_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(folder_path, file))
    return txt_files

def extract_numbers_from_txt_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        numbers = [int(num) for num in content.split() if num.isdigit()]
    return numbers

def main():
    folder_path = "C:/Users/wofvh/Downloads/store.v4i.yolov7pytorch/test/labels"  # 폴더 경로를 적절히 수정해주세요.

    txt_files = get_txt_files_in_folder(folder_path)
    for file_path in txt_files:
        numbers = extract_numbers_from_txt_file(file_path)
        print(f"{file_path}number: {numbers}")

if __name__ == "__main__":
    main()
