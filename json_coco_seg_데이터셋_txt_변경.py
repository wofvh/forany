import json

json_data_path = './ContilValidation.json'

# Read JSON data from the input JSON file
with open(json_data_path, "r") as json_file:
    json_data = json.load(json_file)

cat_dictList = []
img_dictList = []
annot_dictList = []

# Separate the data into categories, images, and annotations lists
for key, val in json_data.items():
    if key == 'categories':
        cat_dictList = val
    elif key == 'images':
        img_dictList = val
    elif key == 'annotations':
        annot_dictList = val

def polygon_to_bbox(polygon_points, image_width, image_height):
    x_coords = polygon_points[0][::2]
    y_coords = polygon_points[0][1::2]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Calculate center_x, center_y, width, and height of the bounding box
    center_x = ((max_x + min_x) / (2 * image_width))
    center_y = ((max_y + min_y) / (2 * image_height))
    width = (max_x - min_x) /image_width
    height = (max_y - min_y) /image_height

    return center_x, center_y, width, height


saved_folder_path = 'C:/json_to_txt/save_val/'

for img_list in img_dictList:
    img_id = img_list['id']
    image_width = 1920
    image_height = 1080
    
    # Collect all normalized bounding boxes for the current image
    all_annotation_file = []
    
    for annot_list in annot_dictList:
        if annot_list['image_id'] == img_id:
            for cat_list in cat_dictList:
                if annot_list['category_id'] == cat_list['id']:
                    polygon_points = [sum(annot_list['segmentation'], [])]
                    center_x, center_y, width, height = polygon_to_bbox(polygon_points, image_width, image_height)
                    normalized_bbox = [cat_list['id'], center_x, center_y, width, height]
                    all_annotation_file.append(normalized_bbox)

    if img_list['file_name']:
        file_name = img_list['file_name'].split(".")[0]
        file_type = file_name + ".txt"
        full_file_path = f"{saved_folder_path}\\{file_type}"
        
        content = ""
        for bbox in all_annotation_file:
            class_id, center_x, center_y, width, height = bbox
            yolo_bbox = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
            content += yolo_bbox
        
        with open(full_file_path, 'w') as file:
            file.write(content)
    else:
        print('Image file name not found.')




