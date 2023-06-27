#!/usr/bin/env python3.7

import time
import math
import rospy
from std_msgs.msg import String
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
import sys
import cv2
import numpy as np
import mediapipe as mp

def imgmsg_to_cv2(img_msg):
    #if img_msg.encoding != "bgr8":
        #rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

class FingerProcessor:
    def __init__ (self):
        #self.bridge = CvBridge()
        self.frontImage_ = np.zeros((480,640,3),np.uint8)
        self.isCharging_ = False
    def frontImageCallback(self, data):
        #frontImageTemp = self.bridge.imgmsg_to_cv2(data, "rgb8")
        #frontImageTemp = imgmsg_to_cv2(data)
        #self.frontImage_ = cv2.flip(frontImageTemp, 1)
        self.frontImage_ = imgmsg_to_cv2(data)
    def isChargingCallback(self,data):
        self.isCharging_ = data.data


def main():
    rospy.init_node('fingerProcessor', anonymous=True)
    rate = rospy.Rate(30.0/1.0) # 1/2hz
    ic = FingerProcessor()
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')
    # model.classes = [0]
    # model.conf = 0.5
    # model.iou = 0.45
    # model.max_det = 32

    # maximum hand number
    max_num_hands = 2

    # gesture map
    gesture = {
        0 : 'Hi!', 1: 'Grab', 2: 'Yeah!', 3: 'Good', 4: 'Love', 5: 'Bad', 7: 'Grab', 9 : 'Love', 56 : 'No!!!', 41 : 'one', 43 : 'three', 44 : 'four', 52: 'palm'
        # 6: 'Grab_B', 51 : 'none', 52: 'none', 53: 'none', 54: 'none', 55: 'none', 57: 'none', 58 : 'none',
        # 7 : 'Fighting' -> 'Grab'
    }

    # mediapipe hands solutions
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands = max_num_hands,
        model_complexity = 0, 
        min_detection_confidence = 0.75,
        min_tracking_confidence = 0.75
    )

    file = np.genfromtxt('/home/nvidia/dcornic-media/src/finger_processor/src/dataset_v3.txt', delimiter=',')
    angleFile = file[:,:-1]
    labelFile = file[:, -1]
    angle = angleFile.astype(np.float32)
    label = labelFile.astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    startTime = time.time()
    sentence = ['none']
    cnt = 0

    motion = {0:['none'],
            1:['none']}
    cnt_p = 0
    t = 0.001
    status = True

    windows = []

    rospy.Subscriber("/resized_front_camera", Image, ic.frontImageCallback)
    rospy.Subscriber("/is_charging", Bool, ic.isChargingCallback)
    
    fingerLeftResultPublisher = rospy.Publisher("/finger_left_result", String, queue_size = 10)
    fingerRightResultPublisher = rospy.Publisher("/finger_right_result", String, queue_size = 10)

    fingerPoseArrayPublisher = rospy.Publisher("/finger_pose_array", PoseArray, queue_size = 10)
    takePhotoPublisher = rospy.Publisher("/grabbed_to_take_photo", Empty, queue_size = 10)

    while not rospy.is_shutdown():
        # time.sleep(t) 
        start = time.time()
        # ret, img = cap.read()
        # if not ret:
        #     continue
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=ic.frontImage_
        # imgRGB = cv2.resize(img, (320,240))
        imgRGB = ic.frontImage_
        result = hands.process(imgRGB)
        # result_yolo = model(imgRGB, size=(160,128))
        
        # for idx in range(len(result_yolo.xyxy[0])):
        #     xy = result_yolo.xyxy[0][idx].cpu().numpy()
        #     xy_min = (int(xy[0]), int(xy[1]))
        #     xy_max = (int(xy[2]), int(xy[3]))
        
        #     cv2.rectangle(img, xy_min, xy_max, color = (0,0,255), thickness = 3)
        # count_person = len(result_yolo.xyxy[0])  
        # h, w, c = img.shape 
        check_num = 16
        check_num_ = check_num * (-1)
        num_h = int(check_num/4)
        num_g = int(check_num/8) * -1
        
        check_h = ['Hi!']*int(check_num/4)
        check_g = ['Grab']*int(check_num/8)

        pose_array = PoseArray()

        if (motion[0][:num_h] == check_h and motion[0][num_g:] == check_g) or \
            (motion[1][:num_h] == check_h and motion[1][num_g:] == check_g):
            cnt_p += 1
            #print(f'take a {cnt_p} selfie!')
            takePhotoPublisher.publish(Empty())

            motion = {0:['none'],
                        1:['none']}
            
            #cv2.putText(img, 'cheese~!', (round(w*0.35), round(h*0.2)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)        
            t = 1
            status = False
            
        else:
            t = 0.001
            status = True
        
        if result.multi_hand_landmarks is not None and status:
            
            for idx, res in enumerate(result.multi_hand_landmarks):
                joint = np.zeros((21,3))
                
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z] # 
                    pose = Pose()
                    if (ic.isCharging_) :
                        pose.position.x = lm.x * 640 + 800
                        if (pose.position.x > 1280) :
                            pose.position.x = pose.position.x - 1280

                    else :
                        pose.position.x = lm.x * 640 + 160    
                    #pose.position.x = lm.x * 640 + 160
                    pose.position.y = lm.y * 480
                    pose_array.poses.append(pose)

                sign = []
                
                for i in range(21):
                    if i == 2:
                        continue
                    if (joint[i][1] - joint[2][1]) > 0:
                        sign.append(1)
                    else:
                        sign.append(90)
                                
                v1 = joint[[0,1,2,3,0,5,6,7,0,9, 10,11,0, 13,14,15,0, 17,18,19], :] 
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
                
                
                v = v2 - v1 #
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 
                
                compareV1 = v[[0,1,2,4,5,6,7,8, 9, 10,12,13,14,16,17], :]
                compareV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
                # 
                
                angle = np.degrees(angle)
                
                #
                sign = np.array(sign)
                angle_sign = np.concatenate((angle, sign))

                data = np.array([angle_sign], dtype=np.float32)
                # 
                ret, results, neighbours, dist = knn.findNearest(data, 10)
                index = int(results[0][0])
                
                # 
                try:
                    motion[idx].append(gesture[index])
                except:
                    pass
                    # motion[idx].append('none')
                # motion = motion[-10:]
                # print(gesture[index])
                if index in gesture.keys():

                    sentence.append(gesture[index])                        
                #     cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] -10),
                #                                             int(res.landmark[0].y * img.shape[0] + 40)), 
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1), #
                #     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),) #
        else:
            sentence.append('none')
        sentence = sentence[-5:]
        # 
        resultStringLeft = ""
        resultStringRight = ""
        if len(set(sentence)) == 1 and sentence[-1] != 'none':
            # cv2.putText(img, sentence[-1], (20, int(h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            resultStringLeft =  sentence[-1]
        elif len(set(sentence[::2])) == 1 and len(set(sentence[1::2])) == 1 and sentence[-1] != 'none' and sentence[-2] != 'none':
            # cv2.putText(img, sentence[-1]+ ',' + sentence[-2], (20, int(h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            resultStringLeft =  sentence[-1]
            resultStringRight =  sentence[-2]
        elif len(set(sentence[::2])) == 1 and len(set(sentence[1::2])) == 1 and sentence[-1] != 'none':
            # cv2.putText(img, sentence[-1], (20, int(h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            resultStringLeft =  sentence[-1]
            resultStringRight =  'none'
        elif len(set(sentence[::2])) == 1 and len(set(sentence[1::2])) == 1 and sentence[-2] != 'none':
            # cv2.putText(img, sentence[-2], (20, int(h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            resultStringLeft =  'none'
            resultStringRight =  sentence[-2]

        
            
        fingerLeftResultPublisher.publish(String(resultStringLeft))
        fingerRightResultPublisher.publish(String(resultStringRight))
        fingerPoseArrayPublisher.publish(pose_array)
        # print(len(pose_array.poses))
        motion[0] = motion[0][check_num_:]
        motion[1] = motion[1][check_num_:]
        #cv2.putText(img, f'Count :{count_person}', (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        end = time.time()
        fps = 1 / (end-start)
        # cv2.putText(img, f'FPS : {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # windows.append('HandTracking')
        # cv2.namedWindow('HandTracking', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('HandTracking', 640, 480)
        # cv2.imshow('HandTracking', img)
        # cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
