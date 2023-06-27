#!/usr/bin/env python3.6
import tensorflow as tf
import rospy
import cv2
import numpy as np
import time
import os

from sensor_msgs.msg import Image
# from obj_detect.msg import Images
# from obj_detect.msg import BoundingBox
# from obj_detect.msg import BoundingBoxes
from object_detector.msg import BoundingBoxes
from object_detector.msg import BoundingBox


class objdetect():
    def __init__(self):
        # model
        self.interpreter = tf.lite.Interpreter(model_path = "/home/nvidia/dcornic-media/src/object_detector/scripts/models/tf2_ssd_mobilenet_v2_coco17_ptq.tflite")
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index0 = self.interpreter.get_output_details()[0]['index']
        self.output_index1 = self.interpreter.get_output_details()[1]['index']
        self.output_index3 = self.interpreter.get_output_details()[3]['index']
        
        self.input_shape_x = 300
        self.input_shape_y = 300

        
        # config
        self.height = 480
        self.width = 640
        self.ratio = self.height / self.width
        self.input_shape_y_ = int(self.input_shape_x * self.ratio)
        self.fontScale = 0.5
        self.show_label = True
        
        
        # pub, sub
        self.result_pub = rospy.Publisher('/obj_detect/detection_img', Image, queue_size = 1)
        self.result_pub_bbox = rospy.Publisher('obj_detect/bounding_boxes', BoundingBoxes, queue_size = 1)        
        # self.subscriber = rospy.Subscriber('/usb_cam1/image_raw', Image, self.callback, queue_size = 1)
        self.subscriber = rospy.Subscriber('/concatenated_image', Image, self.callback, queue_size = 1)
        
        # detect_obj
        self.image = np.zeros((480, 640, 3))
        # self.bboxs = [] 
        
    def callback(self, data):
        t1 = time.time()
        msg = data
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.image = image      
        # resize
        img = cv2.resize(image, (self.input_shape_x, self.input_shape_y), interpolation=cv2.INTER_AREA)
        # data RGB로 오니까 변경 X
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis = 0)

        self.interpreter.set_tensor(self.input_index, img)
        self.interpreter.invoke()

        pred_score = self.interpreter.get_tensor(self.output_index0)[0]
        pred_bbox = self.interpreter.get_tensor(self.output_index1)[0]
        pred_class = self.interpreter.get_tensor(self.output_index3)[0]
        
        # draw bbox
        # bboxs = []
        msg_bboxes = BoundingBoxes()
        
        height = data.height
        width = data.width        
        
        for i in range(len(pred_score)):
            if pred_score[i] < 0.7 or pred_class[i] != 0:
                continue
            
            msg_bbox = BoundingBox()
            xmin = max(int(pred_bbox[i][1] * width), 0)
            ymin = max(int(pred_bbox[i][0] * height), 0)
            xmax = min(int(pred_bbox[i][3] * width), width)
            ymax = min(int(pred_bbox[i][2] * height), height)

            msg_bbox.xmin = xmin
            msg_bbox.ymin = ymin
            msg_bbox.xmax = xmax
            msg_bbox.ymax = ymax
            msg_bbox.Class = 'person'
            msg_bbox.probability = pred_score[i]
            
            msg_bboxes.bounding_boxes.append(msg_bbox)          
            
            # bboxs.append([xmin, ymin, xmax, ymax])
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            
            if self.show_label:
                bbox_msg = 'GUEST'
                t_size = cv2.getTextSize(bbox_msg, 0, self.fontScale, thickness=1)[0]
                c3 = (xmin+t_size[0], ymin-t_size[1]-3)
                cv2.rectangle(image, (xmin, ymin), (int(np.float32(c3[0])), int(np.float32(c3[1]))), (255,0,0), -1)
                cv2.putText(image, bbox_msg, (xmin, int(np.float32(ymin-2))), cv2.FONT_HERSHEY_SIMPLEX, self.fontScale, (255,255,255), 1, lineType=cv2.LINE_AA)                
        
        # self.bboxs = bboxs
        t2 = time.time()
        fps = int(1 / (t2-t1))
        print(fps)
        cv2.putText(image, f'FPS: {fps}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
        msg.data = image.tobytes()
        self.result_pub.publish(msg)
        self.result_pub_bbox.publish(msg_bboxes)
        
        

def main():
    rospy.init_node('obj_detect', anonymous=False)
    checker = objdetect()
    
    rospy.spin()

    # rate = rospy.Rate(1)
    # obj_pub = rospy.Publisher("/obj_detect/detection_obj", Images, queue_size=1)

    # while not rospy.is_shutdown():
    #     msgs = Images()
    #     for bbox in checker.bboxs:
    #         msg = Image()
    #         img = checker.image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #         msg.height = img.shape[0]
    #         msg.width = img.shape[1]
    #         msg.data = img.tobytes()
    #         msgs.images.append(msg)
    #     obj_pub.publish(msgs)
    #     rate.sleep()
    
        
if __name__ == '__main__':
    main()
