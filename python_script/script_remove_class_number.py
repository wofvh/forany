import os

def remove_classes_and_coordinates(folder_path, target_classes):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        new_lines = []

        for line in lines:
            parts = line.split(' ')
            current_class = parts[0]

            if current_class not in target_classes:
                new_lines.append(line)

        with open(file_path, 'w') as file:
            file.writelines(new_lines)

if __name__ == "__main__":
    folder_path = "F:\yolov7-main\coco128/valid\labels/"
    target_classes = ["2","4","6"]
    remove_classes_and_coordinates(folder_path, target_classes)

    print("done")
