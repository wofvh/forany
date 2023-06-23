#!/usr/bin/env python
# -*- coding:utf-8 -*-
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
# import the necessary packages
import rospy
from time import sleep

from std_msgs.msg import Int32
from std_msgs.msg import Empty
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from PIL import Image, ImageDraw, ImageFilter,ImageFont
from PIL import Image as pImage
import cv2
import numpy as np
import os
import random
def images_to_video(image_folder,video_path,image_count_,fps = 13):
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

class imagesplit():
    def __init__ (self):
        self.takeaphoto_ = False
        
    def takeaphoto(self,data):
        self.takeaphoto_ = True
        
def main():
    rospy.init_node('life_shot_image', anonymous=False) 
    rateFloat = 15.0/1.0
    rate = rospy.Rate(rateFloat)
    
    ic = imagesplit() 
    
    # Pub
    # lifeshotimage = rospy.Publisher("/life_shot_image", Empty, queue_size = 1)
    requestImageHostingPublisher = rospy.Publisher('/request_image_hosting', Empty, queue_size = 1)
    lifeshotimage = rospy.Publisher('/lifeshot_image', Empty, queue_size = 1)
    # Sub
    rospy.Subscriber("/makevideo", Empty, ic.takeaphoto)

    image_folder = "/home/nvidia/dcornic-media/src/video/save/"
    video_path = '/home/nvidia/dcornic-media/src/video/output/video.mp4'
    image_count_ = 0
    for_count = 45
    width = 640
    height = 480
    #라인 굵기
    top_border_width = 10
    mid_width = 5
    mid2_width = 5
    bottom_border_width = 115
    left_border_width = 10
    right_border_width = 10
    # isSaying = Fals
    #랜덤 폰트 
    text_list = ("have a nice day!",
                "you are so cute",
                "you are beautiful",
                "you are so lovely",
                "you look so cool",
                "you look awesome",)
    

    while not rospy.is_shutdown():
        if ic.takeaphoto_ :
            random_text = random.choice(text_list)
            color_coverted01 = [cv2.cvtColor(cv2.imread("/home/nvidia/dcornic-media/src/video/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(1, 45)]
            color_coverted02 = [cv2.cvtColor(cv2.imread("/home/nvidia/dcornic-media/src/video/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(46, 90)]
            color_coverted03 = [cv2.cvtColor(cv2.imread("/home/nvidia/dcornic-media/src/video/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(91, 135)]
            color_coverted04 = [cv2.cvtColor(cv2.imread("/home/nvidia/dcornic-media/src/video/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(136, 180)]
            for i in range(1, 46):
                image_count_ = image_count_+ 1
                top_left_resized = cv2.resize(color_coverted01[i % len(color_coverted01)], (640, 450))
                top_right_resized = cv2.resize(color_coverted02[i % len(color_coverted02)], (640, 450))
                bottom_left_resized = cv2.resize(color_coverted03[i % len(color_coverted03)], (640, 510))
                bottom_right_resized = cv2.resize(color_coverted04[i % len(color_coverted04)], (640, 510))
                combined_image = np.zeros((960, 1280, 3), dtype=np.uint8)
                combined_image[:450, :640] = top_left_resized
                combined_image[:450, 640:1280] = top_right_resized
                combined_image[450:960, :640] = bottom_left_resized
                combined_image[450:960, 640:1280] = bottom_right_resized
                dst = cv2.resize(combined_image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                d1 = Image.fromarray(dst)
                ft = ImageFont.truetype("/home/nvidia/dcornic-media/src/KGPrimaryWhimsy.ttf", 40)
                d = ImageDraw.Draw(d1)
                d.line([(width-1, 0), (width-1, height-1)], fill="#371010", width=right_border_width)
                d.line([(0, 225), (640, 225)], fill="#371010", width=mid_width)
                d.line([(320, 0), (320, 480)], fill="#371010", width=mid2_width)
                d.line([(0, 0), (width-1, 0)], fill="#371010", width=top_border_width)
                d.line([(width-1, height-1), (0, height-1)], fill="#371010", width=bottom_border_width)
                d.line([(0, height-1), (0, 0)], fill="#371010", width=left_border_width)
                d.text((215, 430), random_text, font=ft, fill="white")
                numpy_image = np.array(d1)
                out = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("/home/nvidia/dcornic-media/src/video/save/{}.jpg".format(i), out)
                
            if for_count <= image_count_:
                print(image_count_)
                images_to_video(image_folder,video_path,image_count_,fps = 13)
                requestImageHostingPublisher.publish(Empty())
                ic.takeaphoto_ = False
                image_count_ == 0
                            
            # elif image_count == image_count_ + 5:
            #     ic.takeaphoto_ = False
            #     image_count == 0
        
        # if ic.can_publish
    rate.sleep()
        

if __name__ == '__main__':
    main()
    
