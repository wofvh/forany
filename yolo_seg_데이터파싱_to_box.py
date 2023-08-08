import os
import glob

def polygon_to_bbox(polygon_points, image_width, image_height):

    # Find minimum and maximum x, y coordinates of the polygon
    min_x, min_y = min(polygon_points, key=lambda p: p[0])[0], min(polygon_points, key=lambda p: p[1])[1]
    max_x, max_y = max(polygon_points, key=lambda p: p[0])[0], max(polygon_points, key=lambda p: p[1])[1]

    # Calculate center_x, center_y, width, and height of the bounding box
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    # Normalize the coordinates to range [0, 1]
    # center_x /= image_width
    # center_y /= image_height
    # width /= image_width
    # height /= image_height

    return center_x, center_y, width, height


def convert_polygon_to_bbox(input_file, output_file, image_width, image_height):
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as output:
        for line in lines:
            if line[0] =='0':
            # Split the line into class and polygon points
                class_label, *polygon_points = line.strip().split()

                # Convert polygon points to a list of tuples
                polygon_points = [(float(polygon_points[i]), float(polygon_points[i + 1])) for i in range(0, len(polygon_points), 2)]

                # Convert polygon to bounding box
                center_x, center_y, width, height = polygon_to_bbox(polygon_points, image_width, image_height)

                # Write the bounding box in YOLO format to the output file
                output.write(f"{class_label} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            else:
                output.write(line)



def read_files_using_glob(folder_path, file_extension='*'):
    file_contents = {}

    # Use glob to find all files in the folder with the specified file_extension
    files = glob.glob(os.path.join(folder_path, f"*.{file_extension}"))

    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as file:
            content = file.read()
            file_contents[filename] = content

    return file_contents

# Example usage:
folder_path_test = r'C:/label/'
output_folder_path_test = r'C:/out01/'
file_extension = "txt"  

try:
    all_files_contents = read_files_using_glob(folder_path_test, file_extension)
    for filename, content in all_files_contents.items():
        print(f"File: {filename}")
        print("Content:")
        print(content)
        print("-" * 30)
        input_file = os.path.join(folder_path_test, filename)
        output_file = os.path.join(output_folder_path_test, filename)    # Replace this with the desired output path for the bounding box file
        image_width = 640                 # Replace this with the width of your images
        image_height = 640                # Replace this with the height of your images
        convert_polygon_to_bbox(input_file, output_file, image_width, image_height)
except OSError as e:
    print(f"Error reading files from the folder: {e}")

