#!/usr/bin/env python
# -*- coding:utf-8 -*-
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
# import the necessary packages
import rospy
from tracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression

from std_msgs.msg import Int32
from std_msgs.msg import Empty
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from object_detector.msg import BoundingBoxes
from media_controller.msg import CongestionLevel
from geometry_msgs.msg import PoseArray

from std_msgs.msg import String

from PIL import Image, ImageDraw, ImageFilter,ImageFont
from PIL import Image as pImage
import cv2
import numpy as np
import os


#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Empty
from time import sleep


class imagesplit():
    def __init__ (self):
        self.imagelifeshot = False
        
    def image_split_callback(self,data):
        self.imagelifeshot = True
    
        
def main():
    rospy.init_node('images_split', anonymous=False) 
    rateFloat = 5.0/1.0
    rate = rospy.Rate(rateFloat)
    ic = imagesplit() 
    
    # Pub
    lifeshotimage = rospy.Publisher("/life_shot_image", String, queue_size = 1)
    requestImageHostingPublisher = rospy.Publisher('/request_image_hosting', Empty, queue_size = 1)
    # Sub
    rospy.Subscriber("/life_shot_image", Int32, ic.image_split_callback)
    rospy.Subscriber("/notice_to_jeon", String,  ic.noticetojeon)
    
    # rospy.Subscriber('/finger_right_result', String, ic.fingerRightResultCallback)
    
    # isSaying = False

    while not rospy.is_shutdown():
        
        if ic.imagelifeshot :
            for i in range(1,46):
                color_coverted01 = [cv2.cvtColor(cv2.imread("C:/allmodel/opencv/data/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(1, 45)]
                color_coverted02 = [cv2.cvtColor(cv2.imread("C:/allmodel/opencv/data/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(46, 90)]
                color_coverted03 = [cv2.cvtColor(cv2.imread("C:/allmodel/opencv/data/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(91, 135)]
                color_coverted04 = [cv2.cvtColor(cv2.imread("C:/allmodel/opencv/data/{}.jpg".format(i)), cv2.COLOR_BGR2RGB) for i in range(136, 181)]
                combined_image = np.zeros((960, 1280, 3), dtype=np.uint8)
                top_left_resized = cv2.resize(color_coverted01[(i-1) % 29], (640, 450))
                top_right_resized = cv2.resize(color_coverted02[(i-1) % 29], (640, 450))
                bottom_left_resized = cv2.resize(color_coverted03[(i-1) % 29], (640, 510))
                bottom_right_resized = cv2.resize(color_coverted04[(i-1) % 29], (640, 510))
                combined_image[:450, :640] = top_left_resized
                combined_image[:450, 640:1280] = top_right_resized
                combined_image[450:960, :640] = bottom_left_resized
                combined_image[450:960, 640:1280] = bottom_right_resized
                dst = cv2.resize(combined_image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                d1 = Image.fromarray(dst)
                ft = ImageFont.truetype("C:/allmodel/opencv/html/KGPrimaryWhimsy.ttf", 40)
                d = ImageDraw.Draw(d1)
                width = 640
                height = 480
                top_border_width = 10
                mid_width = 5
                mid2_width = 5
                bottom_border_width = 115
                left_border_width = 10
                right_border_width = 10
                d.line([(width-1, 0), (width-1, height-1)], fill="#241E1B", width=right_border_width)
                d.line([(0, 225), (640, 225)], fill="#241E1B", width=mid_width)
                d.line([(320, 0), (320, 480)], fill="#241E1B", width=mid2_width)
                d.line([(0, 0), (width-1, 0)], fill="#241E1B", width=top_border_width)
                d.line([(width-1, height-1), (0, height-1)], fill="#241E1B", width=bottom_border_width)
                d.line([(0, height-1), (0, 0)], fill="#241E1B", width=left_border_width)
                d.text((200, 430), "have a nice day!", font=ft, fill="white")
                numpy_image = np.array(d1)
                out = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("C:/allmodel/opencv/vdieo/save/{}.jpg".format(i), out)
                lifeshotimage.publish(out)

            
        # if ic.can_publish
        
            rate.sleep()
        

if __name__ == '__main__':
    main()
    
