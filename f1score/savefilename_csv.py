import os
import csv
import re

# Specify the folder path you want to read files from
#folder_path = "J:/sibtain/NIPA_CW/NIPA_Classification/Cutting and Welding/Prediction/"
#folder_path = "J:\\Sibtain\\NIPA_MS_22_11_2023\\Classification\\Prediction"
folder_path = "C:\\f1score\\test\\Prediction"
#folder_path = "J:/sibtain/NIPA_CW/NIPA_Classification/Cutting and Welding/Prediction/"

# Create a list to store the file names
file_names = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        file_names.append(filename)

# Specify the CSV file path where you want to save the file names
csv_file = "C:\\f1score\\test\\Prediction\\PT.csv"

# Write the file names to a CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(["File Name"])  # Write a header row
    for name in file_names:
        # if re.search("t0",name) or re.search("mg0",name) or re.search("h0",name) or re.search("3p",name):
        #     writer.writerow([name, 0])
        # else:
        #     writer.writerow([name, 1])
        if re.search("_ABCUNSAFE",name):
            writer.writerow([name, 0])
        else:
            writer.writerow([name, 1])


print("File names saved to", csv_file)
