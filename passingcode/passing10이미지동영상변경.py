
import cv2
import numpy as np
import os
import random


image_folder = "/home/nvidia/dcornic-media/src/video/save/"
video_path = '/home/nvidia/dcornic-media/src/video/output/video.mp4'


def images_to_video(image_folder,video_path,image_count_,fps = 16):
    image_files = [str(x)+'.jpg' for x in range(1, image_count_ + 1)]
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
        # print(image_file)
    video_writer.release()

images_to_video()