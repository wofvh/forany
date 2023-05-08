#! /usr/bin/env python3.7
import tensorflow as tf
import rospy
import cv2
import numpy as np
import time
import os

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import Empty

from fdlite import FaceDetection, FaceDetectionModel
from PIL import Image as pImage
from tracker.centroidtracker import CentroidTracker
# from tracker.sort import Sort

class fdtflite():
    def __init__(self):
        self.detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
        
        # for age, gender estimation
        self.margin = 0.4 
        
        self.face_msg = Image()
        
        self.ct = CentroidTracker()
        self.state = True
        
        self.sub = rospy.Subscriber('/camera1/usb_cam1/image_raw', Image, self.callback, queue_size = 1)
        self.pub_screen = rospy.Publisher('/object_detector/detection_face', Image, queue_size = 1)
        self.pub_command = rospy.Publisher('/face_follow_cmd', String, queue_size = 1)
        
        
        self.command_pre = ''
        self.command = 'stop'
        self.th = 120
        self.th_ = -120
        self.angle = 
        self.stop_cnt = 0
        
    def callback(self, data):
        t1 = time.time()
        
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        image = cv2.flip(image, 1)
        
        img = pImage.fromarray(image)
        faces = self.detect_faces(img)
        # print(len(faces))
        max_area = 0
        box = []
        # 가장 큰 객체의 bbox 정보 찾기
        # rects_sort = []
        for face in faces:
            score = face.score
            if score < 0.6:
                continue
            bbox = face.bbox
            
            w = bbox.x.max - bbox.xmin
            h = bbox.y.max - bbox.ymin
            
            area = w*h
            # print(area)
            # 일정 크기 이상도 추가하자, 0.015?
            if area > max_area and area > 0.01:
                max_area = area
                box = bbox
                width = w
                height = h
        
        rects = []
        if box:
            # 크게(얼굴+배경)
            xmin = min(max(int((box.xmin - self.margin*width)*data.width), 0), data.width)
            ymin = min(max(int((box.ymin - self.margin*height)*data.height), 0), data.height)
            xmax = min(max(int((box.xmax + self.margin*width)*data.width), 0), data.width)
            ymax = min(max(int((box.ymax + self.margin*height)*data.height), 0), data.height)     
                  
            rects.append((xmin, ymin, xmax, ymax))
            
            self.stop_cnt = 0
            
        else:

            self.stop_cnt += 1
            if self.stop_cnt > 5:
                msg = String()
                msg.data = 'stop'
                self.pub_command.publish(msg)
                print('publish stop(no detected)')
                self.stop_cnt = 0
            # msg = String()
            # msg.data = 'stop'
            # self.pub_command.publish(msg)
            # print('publish stop(no detected)')

        objects = self.ct.update(rects)
        
        if self.command == 'stop':
            self.th_ = -100
            self.th = 100
        elif self.command == 'left':
            self.th_ = -150
            self.th = 50
        else:
            self.th_ = -50
            self.th = 150
        
        for (objectID, centroid) in objects.items():
            if self.ct.disappeared[objectID] == 0:
                
                bbox = self.ct.bbox[objectID]
                
                # 30프레임 동안 추적 시 따라가기
                if self.ct.appeared[objectID] > 1:
                    x_center = centroid[0]
                    x_diff = 320 - x_center
                    
                    
                    # 카메라 기준으로는 반대로 움직여야 함
                    if x_diff > self.th_ and x_diff < self.th:
                        self.command = 'stop'
                        msg = String()
                        msg.data = 'stop'
                        self.pub_command.publish(msg)

                    elif x_diff <= self.th_:
                        self.command = 'left'
                        msg = String()
                        msg.data = 'left'
                        self.pub_command.publish(msg)
                        self.position=(max*(degree-27))/360;
                    else:
                        self.command = 'right'
                        msg = String()
                        msg.data = 'right'
                        self.position=(max*(degree-27))/360;
                        self.pub_command.publish(msg)
                        

                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
                cv2.putText(image, str(objectID)+','+str(self.ct.appeared[objectID]), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)                
                # id 제외
                # cv2.putText(image, str(self.ct.appeared[objectID]), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)                
                

        
        t2 = time.time()
        fps = int(1/(t2-t1))
        cv2.putText(image, f'FPS : {fps}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        msg_screen = data
        msg_screen.data = image.tobytes()
        self.pub_screen.publish(msg_screen)
        
    
def main():
    rospy.init_node('fdtflite', anonymous=False)
    ic = fdtflite()
    
    rospy.spin()
    
    
if __name__ == '__main__':
    main()
        
            


            
                
        
        
