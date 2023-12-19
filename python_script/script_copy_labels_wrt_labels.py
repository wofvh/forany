import os
import shutil

# Specify the folder containing the text files and images
img_folder = 'J:\\Sibtain\\NIPA_LADDER\\LadderNew\\mix\\Images'
#txt_folder = 'J:\\Sibtain\\NIPA_LADDER\\LabelNew\\mix\\Labels'
txt_folder = 'J:\\Sibtain\\NIPA_LADDER\\LadderNew\\mix\\Labels'
#txt_folder = 'H:\\Dataset\\MobileScaffolding\\combined\\MobileScaffolding2023.v4i.yolov8\\train\\images'
#img_folder = 'H:\\Dataset\\MobileScaffolding\\combined\\MobileScaffolding2023.v4i.yolov8\\train\\images'

# Get a list of text file names
txt_files = os.listdir(txt_folder)
print(len(txt_files))
# Iterate through the text files
for txt_file in txt_files:
    txt_file_name, txt_file_ext = os.path.splitext(txt_file)

    # Construct the corresponding image file name
    img_file = txt_file_name + '.txt'  # Change the extension as needed (e.g., '.jpg', '.png', etc.)
    print(img_file)
    # Check if the image file exists in the image folder
    img_path = os.path.join(img_folder, img_file)
    if os.path.isfile(img_path):
        # Copy the image file to a destination folder (e.g., 'output_folder')
        destination_folder = 'J:\\Sibtain\\NIPA_LADDER\\LadderNew\\mix\\NewLabels'
        shutil.copy(img_path, os.path.join(destination_folder, img_file))

# You can also handle different extensions for text and image files by modifying the code accordingly.
