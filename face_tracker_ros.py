#!/usr/bin/env python
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from tracker.centroidtracker import CentroidTracker
# from imutils.video import VideoStream
# from imutils.object_detection import non_max_suppression

from std_msgs.msg import Int32
from std_msgs.msg import Empty
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
# from object_detector.msg import BoundingBoxes
from obj_detect.msg import BoundingBoxes

# from media_controller.msg import CongestionLevel
from geometry_msgs.msg import PoseArray
from mask_cls.msg import CovidImage

import rospy 
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import argparse
# import imutils
import cv2
import base64
import os
import sys
import random
from datetime import date


def logoOverlay(image,logo,alpha=1.0,x=0, y=0, scale=1.0): 
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    overlay = cv2.resize(logo, None,fx=scale,fy=scale)
    (wH, wW) = overlay.shape[:2]
    output = image.copy()
    # blend the two images together using transparent overlays
    try:
        if x<0 : x = w+x
        if y<0 : y = h+y
        if x+wW > w: wW = w-x  
        if y+wH > h: wH = h-y
        print(x,y,wW,wH)
        overlay=cv2.addWeighted(output[y:y+wH, 400:400+wW],alpha,overlay[:wH,:wW],1.0,0)
        output[y:y+wH, 400:400+wW ] = overlay
    except Exception as e:
        print("Error: Logo position is overshooting image!")
        print(e)
    output= output[:,:,:3]
    return output

def edge_mask(img, line_size, blur_value):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.medianBlur(gray, blur_value)
	edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
	return edges

def color_quantization(img, k):
	# Transform the image
	data = np.float32(img).reshape((-1, 3))

	# Determine criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

	# Implementing K-Means
	ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	result = center[label.flatten()]
	result = result.reshape(img.shape)
	return result

class FaceTracker:
	def __init__ (self):

		# Publisher
		# self.congestionPublisher = rospy.Publisher('object_detector/congestionLevel', CongestionLevel, queue_size = 10)
		self.cameraWakeupPublisher = rospy.Publisher('camera_wake_up', Empty, queue_size = 10)

		self.noMaskFacePublisher_ = rospy.Publisher("/nomask_face_raw", CovidImage, queue_size=10)
		self.resizedFrontCameraImagePublisher = rospy.Publisher("/resized_front_camera", Image, queue_size=10)

		# initialize our centroid tracker and frame dimensions
		self.ct = CentroidTracker()

		# Global 
		self.numberOfPeople_ = 0
		self.screenNOP_ = 0
		self.currentSeconds_ = 0.0

		self.bridge = CvBridge()

		self.frontImage_ = np.zeros((480,640,3),np.uint8)
		self.rearImage_ = np.zeros((480,640,3),np.uint8)
		
		self.frontImage1_ = np.zeros((240,80,3),np.uint8)
		self.frontImage2_ = np.zeros((240,80,3),np.uint8)
		self.frontImage3_ = np.zeros((240,80,3),np.uint8)
		self.frontImage4_ = np.zeros((240,80,3),np.uint8)

		self.rearImage1_ = np.zeros((240,80,3),np.uint8)
		self.rearImage2_ = np.zeros((240,80,3),np.uint8)
		self.rearImage3_ = np.zeros((240,80,3),np.uint8)
		self.rearImage4_ = np.zeros((240,80,3),np.uint8)

		self.originalFrontImage1_ = np.zeros((480,160,3),np.uint8)
		self.originalFrontImage2_ = np.zeros((480,160,3),np.uint8)
		self.originalFrontImage3_ = np.zeros((480,160,3),np.uint8)
		self.originalFrontImage4_ = np.zeros((480,160,3),np.uint8)

		self.originalRearImage1_ = np.zeros((480,160,3),np.uint8)
		self.originalRearImage2_ = np.zeros((480,160,3),np.uint8)
		self.originalRearImage3_ = np.zeros((480,160,3),np.uint8)
		self.originalRearImage4_ = np.zeros((480,160,3),np.uint8)

		self.originalAddH_ = np.zeros((480,1280),np.uint8)

		self.detectionImageVertical1_ = np.zeros((240,640,3),np.uint8)
		self.detectionImageVertical2_ = np.zeros((240,1280,3),np.uint8)
		self.detectionImageVertical3_ = np.zeros((120,640,3),np.uint8)

		self.fingerLeftResult_ = ""		
		self.fingerRightResult_ = ""

		self.isHandgrabbed_ = False
		self.isCaptured_ = False

		self.fingerPoseArray_ = PoseArray()

		self.isDrawingFinger_ = False

		self.isAiModShowing_ = False
		self.aiModeShowingCount_ = 0

		self.grabToTakePhoto_ = False

		self.isSaying_ = False
		self.isCharging_ = False

		self.personBoundingBoxes_ = BoundingBoxes()
		self.allBoundingBoxes = BoundingBoxes()

		self.noMaskBoundingBoxes_ = BoundingBoxes()
		self.maskBoundingBoxes_ = BoundingBoxes()

		self.isDrawing = False

		self.qrCount_ = 0

	# def congestionPublishCallback(self, event=None):
	# 	msg = CongestionLevel()
	# 	msg.time = rospy.get_time()
	# 	msg.duration = 5
	# 	msg.congestionLevel = self.numberOfPeople_
	# 	self.congestionPublisher.publish(msg)

	# 	self.ct.__init__()
	# 	self.numberOfPeople_ = 0
	# 	self.screenNOP_ = 0
				
	def yoloCallback(self, data):
		size = len(data.bounding_boxes)
		self.allBoundingBoxes.bounding_boxes = []
		self.allBoundingBoxes = data

		personBoxSize = 0
		
		if self.isDrawing == False :
			self.personBoundingBoxes_.bounding_boxes = []
			self.noMaskBoundingBoxes_.bounding_boxes = []
			self.maskBoundingBoxes_.bounding_boxes = []
			
			for i in range(size):
				# if data.bounding_boxes[i].Class == 'person' and data.bounding_boxes[i].probability >= 0.4:
				personBoxSize = personBoxSize +1
				self.personBoundingBoxes_.bounding_boxes.append(data.bounding_boxes[i])
				
				# elif data.bounding_boxes[i].Class == 'no mask' and data.bounding_boxes[i].probability >= 0.7:
				# 	self.noMaskBoundingBoxes_.bounding_boxes.append(data.bounding_boxes[i])
				# 	noMaskCroppedImg = self.originalAddH_[data.bounding_boxes[i].ymin*2: data.bounding_boxes[i].ymax*2, data.bounding_boxes[i].xmin*2: data.bounding_boxes[i].xmax*2]
				# 	noMaskCovidImageMsg = CovidImage()
				# 	noMaskCovidImageMsg.image = self.bridge.cv2_to_imgmsg(noMaskCroppedImg,"rgb8")
				# 	noMaskCovidImageMsg.cls = 1
				# 	noMaskCovidImageMsg.temperature = random.uniform(36,37.5)
				# 	self.noMaskFacePublisher_.publish(noMaskCovidImageMsg)
				# 	print('nomask published.')

				# elif data.bounding_boxes[i].Class == 'mask' and data.bounding_boxes[i].probability >= 0.4:
				# 	self.maskBoundingBoxes_.bounding_boxes.append(data.bounding_boxes[i])

		if size == 0 :
			self.currentSeconds_ = rospy.get_time()
			#print('size : 0')
		
		rects = []

		isDetectedBigPerson = False

		for i in range(personBoxSize):
			startX = self.personBoundingBoxes_.bounding_boxes[i].xmin
			startY = self.personBoundingBoxes_.bounding_boxes[i].ymin
			endX = self.personBoundingBoxes_.bounding_boxes[i].xmax
			endY = self.personBoundingBoxes_.bounding_boxes[i].ymax

			rects.append( (startX, startY, endX, endY) )

			width = endX - startX
			height = endY - startY

			# print('width : %d, height : ', width, height)
			# if width >= 400 and height >= 300 :				
			if width >= 200 and height >= 150 :				
				isDetectedBigPerson = True

		if isDetectedBigPerson : 
			seconds = rospy.get_time() - self.currentSeconds_
			if seconds >= 3.0:
				print('seconds : %d', seconds)
				emptyMsg = Empty()
				self.cameraWakeupPublisher.publish(emptyMsg)
				self.currentSeconds_ = rospy.get_time()
		else :
			self.currentSeconds_ = rospy.get_time()

		objects = self.ct.update(rects)
		objectID_ = 0

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			objectID_ = objectID

		self.screenNOP_ = objectID_
		self.numberOfPeople_ = max(self.numberOfPeople_, objectID_)

		self.isYoloCallbackCalled = True
	

	def rearImageCallback(self, data):
		rearImageTemp = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.rearImage_ = cv2.flip(rearImageTemp, 1)
		resize_img = cv2.resize(self.rearImage_, (320, 240))

		self.rearImage1_ = resize_img[0: 240, 0: 80]
		self.rearImage2_ = resize_img[0: 240, 80: 160]
		self.rearImage3_ = resize_img[0: 240, 160: 240]
		self.rearImage4_ = resize_img[0: 240, 240: 320]

		self.originalRearImage1_ = self.rearImage_[0: 480, 0: 160]
		self.originalRearImage2_ = self.rearImage_[0: 480, 160: 320]
		self.originalRearImage3_ = self.rearImage_[0: 480, 320: 480]
		self.originalRearImage4_ = self.rearImage_[0: 480, 480: 640]

	def frontImageCallback(self, data):
		frontImageTemp = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.frontImage_ = cv2.flip(frontImageTemp, 1)
		resize_img = cv2.resize(self.frontImage_, (320, 240))

		self.resizedFrontCameraImagePublisher.publish(self.bridge.cv2_to_imgmsg(resize_img))

		self.frontImage1_ = resize_img[0: 240, 0: 80]
		self.frontImage2_ = resize_img[0: 240, 80: 160]
		self.frontImage3_ = resize_img[0: 240, 160: 240]
		self.frontImage4_ = resize_img[0: 240, 240: 320]
		
		self.originalFrontImage1_ = self.frontImage_[0: 480, 0: 160]
		self.originalFrontImage2_ = self.frontImage_[0: 480, 160: 320]
		self.originalFrontImage3_ = self.frontImage_[0: 480, 320: 480]
		self.originalFrontImage4_ = self.frontImage_[0: 480, 480: 640]


	def handGrabCallback(self, data):
		self.isHandgrabbed_ = True

	def detectionImageCallback(self, data):
		self.detectionImageVertical1_ = self.bridge.imgmsg_to_cv2(data, "rgb8")

	def fingerLeftResultCallback(self, data):
		self.fingerLeftResult_ = data.data

	def fingerRightResultCallback(self, data):
		self.fingerRightResult_ = data.data

	def fingerPoseArrayCallback(self, data):
		if self.isDrawingFinger_ == False :
			self.fingerPoseArray_ = data
	
	def grabToTakePhotoCallback(self, data):
		self.grabToTakePhoto_ = True

	def isSayingCallback(self,data):
		self.isSaying_ = data.data
	
	def isChargingCallback(self,data):
		self.isCharging_ = data.data
		

def main():
	rospy.init_node('face_tracker', anonymous=False)
	rateFloat = 15.0/1.0
	# rateFloat = 30.0/1.0
	rate = rospy.Rate(rateFloat) # 1/2hz

	ic = FaceTracker()

	# Publisher
	nopPublisher = rospy.Publisher('/face_tracker/nop', Int32, queue_size = 10)
	concatenatedImagePublisher = rospy.Publisher('/concatenated_image', Image, queue_size = 10)

	# detectionImagePublisher = rospy.Publisher('/darknet_ros/detection_image', Image, queue_size = 10)
	resultFullscreenImagePublisher = rospy.Publisher('/result_fullscreen_detection_image', Image, queue_size = 10)

	requestImageHostingPublisher = rospy.Publisher('/request_image_hosting', Empty, queue_size = 1)
	requestShutterSoundPublisher = rospy.Publisher('/request_shutter_sound', Empty, queue_size = 1)

	hdmiSwitchPublisher = rospy.Publisher('/hdmi_switch', Int32, queue_size = 1)
	moveRequestPublisher = rospy.Publisher('move_request', Int32, queue_size = 1) # 1:pause, 2:move

	requestDisplayPublisher = rospy.Publisher("/request_display", Int32, queue_size=1)

	speechInfoAtWaypointPublisher = rospy.Publisher("/speech_info_at_waypoint", String, queue_size=1)

	handEventPublisher = rospy.Publisher("/hand_event", String, queue_size=1)

	# Subscriber
	# rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, ic.yoloCallback)
	rospy.Subscriber("/obj_detect/bounding_boxes", BoundingBoxes, ic.yoloCallback)

	#rospy.Subscriber("/darknet_ros/detection_image", Image, ic.detectionImageCallback)
	rospy.Subscriber("/usb_cam1/image_raw", Image, ic.frontImageCallback)
	rospy.Subscriber("/usb_cam2/image_raw", Image, ic.rearImageCallback)

	rospy.Subscriber("/hand_grab", Empty, ic.handGrabCallback)

	# rospy.Subscriber("/darknet_ros/detection_image", Image, ic.detectionImageCallback)
	rospy.Subscriber("/obj_detect/detection_image", Image, ic.detectionImageCallback) 

	rospy.Subscriber("/finger_left_result", String, ic.fingerLeftResultCallback)
	rospy.Subscriber("/finger_right_result", String, ic.fingerRightResultCallback)

	rospy.Subscriber("/finger_pose_array", PoseArray, ic.fingerPoseArrayCallback)

	rospy.Subscriber("/grabbed_to_take_photo", Empty, ic.grabToTakePhotoCallback)

	rospy.Subscriber("/is_saying", Bool, ic.isSayingCallback)

	rospy.Subscriber("/is_charging", Bool, ic.isChargingCallback)

	handGrabbedCount = 0

	capturedCount =0

	captureCountDown = 3

	dcornicLogoImg = cv2.imread('/home/nvidia/rb5-media/src/obj_detect/scripts/dcornic_logo_image.png', cv2.IMREAD_UNCHANGED)
	dcornicLogoImg = cv2.resize(dcornicLogoImg, (240, 61), interpolation=cv2.INTER_AREA)
	#Timer
	# rospy.Timer(rospy.Duration(5), ic.congestionPublishCallback)
		
	while not rospy.is_shutdown():
		
		if ic.isCharging_ :
			addh = cv2.hconcat([ic.frontImage4_, ic.rearImage1_, ic.rearImage2_, ic.rearImage3_, ic.rearImage4_, ic.frontImage1_, ic.frontImage2_, ic.frontImage3_])
			ic.originalAddH_ = cv2.hconcat([ic.originalFrontImage4_, ic.originalRearImage1_, ic.originalRearImage2_, ic.originalRearImage3_, ic.originalRearImage4_, ic.originalFrontImage1_, ic.originalFrontImage2_, ic.originalFrontImage3_])
			
		else : 
			addh = cv2.hconcat([ic.rearImage4_, ic.frontImage1_, ic.frontImage2_, ic.frontImage3_, ic.frontImage4_, ic.rearImage1_, ic.rearImage2_, ic.rearImage3_])
			ic.originalAddH_ = cv2.hconcat([ic.originalRearImage4_, ic.originalFrontImage1_, ic.originalFrontImage2_, ic.originalFrontImage3_, ic.originalFrontImage4_, ic.originalRearImage1_, ic.originalRearImage2_, ic.originalRearImage3_])

		image_message = ic.bridge.cv2_to_imgmsg(addh,"rgb8")
		concatenatedImagePublisher.publish(image_message)

		if not ic.isSaying_ :
			nopPublisher.publish(ic.screenNOP_)
			#addh = cv2.hconcat([ic.frontImage_, ic.rearImage_])
						
			if ic.isCaptured_ :
				capturedCount = capturedCount + 1
				if capturedCount == 1 :
					requestShutterSoundPublisher.publish(Empty())
					#addv = cv2.vconcat([addh, ic.detectionImageVertical2_])
					addv = cv2.vconcat([ic.originalAddH_, ic.detectionImageVertical2_])

					imgTemp = ic.frontImage_.copy()
					# imgTemp = cv2.cvtColor(imgTemp,cv2.COLOR_BGR2BGRA);
					# imgTemp[0:61,400:640,:] = dcornicLogoImg[0:61,0:240,:]
					line_size = 5
					blur_value = 3
					edges = edge_mask(imgTemp, line_size, blur_value)
					
					total_color = 9
					img = color_quantization(imgTemp, total_color)
					blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
					cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

					imgTemp = logoOverlay(imgTemp, dcornicLogoImg)

					capturedAddV = cv2.vconcat([imgTemp, cartoon])

					cv2.imwrite('/home/nvidia/dconic_media/src/13.jpg',cv2.cvtColor(capturedAddV, cv2.COLOR_BGRA2RGB))
				elif capturedCount == 7 :
					ic.isCaptured_ = False
					capturedCount = 0
					requestImageHostingPublisher.publish(Empty())			
			else :
				
				ic.isDrawing = True
				boundingBoxesTemp = ic.personBoundingBoxes_
				noMaskBoundingBoxesTemp = ic.noMaskBoundingBoxes_
				# maskBoundingBoxesTemp = ic.maskBoundingBoxes_
				
				tempAddH = ic.originalAddH_.copy()
				bbox_color = (255,0,0)
				for i in range(len(boundingBoxesTemp.bounding_boxes)) :
					name = boundingBoxesTemp.bounding_boxes[i].Class
					if name == 'person':
						name = 'GUEST'
						bbox_color = (0,255,0)
					t_size = cv2.getTextSize(name, 0, 0.55, thickness=2)[0]
					xmin = boundingBoxesTemp.bounding_boxes[i].xmin*2
					ymin = boundingBoxesTemp.bounding_boxes[i].ymin*2+13
					c3 = (xmin+t_size[0], ymin-t_size[1]-3)
					cv2.rectangle(tempAddH,(xmin,ymin),(boundingBoxesTemp.bounding_boxes[i].xmax*2,boundingBoxesTemp.bounding_boxes[i].ymax*2),bbox_color,2 )
					cv2.rectangle(tempAddH,(xmin,ymin),(int(np.float32(c3[0])), int(np.float32(c3[1]))),bbox_color,-1 )
					cv2.putText(tempAddH, name,(xmin,ymin-2), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
					# cv2.putText(tempAddH, 'GUEST',(boundingBoxesTemp.bounding_boxes[i].xmin*2,boundingBoxesTemp.bounding_boxes[i].ymin*2+13), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

				# for i in range(len(maskBoundingBoxesTemp.bounding_boxes)) :
				# 	cv2.rectangle(originalAddH,(maskBoundingBoxesTemp.bounding_boxes[i].xmin*2,maskBoundingBoxesTemp.bounding_boxes[i].ymin*2),(maskBoundingBoxesTemp.bounding_boxes[i].xmax*2,maskBoundingBoxesTemp.bounding_boxes[i].ymax*2),(0,0,255),1 )

				addv = cv2.vconcat([tempAddH, ic.detectionImageVertical2_])

				ic.isDrawing = False
			
				if (len(ic.fingerPoseArray_.poses) >= 21) :
					ic.isDrawingFinger_ = True
					for pose in ic.fingerPoseArray_.poses :
						addv = cv2.circle(addv, (int(pose.position.x), int(pose.position.y)), 3, (255,127,0), -1 )

					if not ic.isCharging_ :
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[0].position.x), int(ic.fingerPoseArray_.poses[0].position.y) ) , ( (int(ic.fingerPoseArray_.poses[1].position.x), int(ic.fingerPoseArray_.poses[1].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[1].position.x), int(ic.fingerPoseArray_.poses[1].position.y) ) , ( (int(ic.fingerPoseArray_.poses[2].position.x), int(ic.fingerPoseArray_.poses[2].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[2].position.x), int(ic.fingerPoseArray_.poses[2].position.y) ) , ( (int(ic.fingerPoseArray_.poses[3].position.x), int(ic.fingerPoseArray_.poses[3].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[3].position.x), int(ic.fingerPoseArray_.poses[3].position.y) ) , ( (int(ic.fingerPoseArray_.poses[4].position.x), int(ic.fingerPoseArray_.poses[4].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[0].position.x), int(ic.fingerPoseArray_.poses[0].position.y) ) , ( (int(ic.fingerPoseArray_.poses[5].position.x), int(ic.fingerPoseArray_.poses[5].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[5].position.x), int(ic.fingerPoseArray_.poses[5].position.y) ) , ( (int(ic.fingerPoseArray_.poses[6].position.x), int(ic.fingerPoseArray_.poses[6].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[6].position.x), int(ic.fingerPoseArray_.poses[6].position.y) ) , ( (int(ic.fingerPoseArray_.poses[7].position.x), int(ic.fingerPoseArray_.poses[7].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[7].position.x), int(ic.fingerPoseArray_.poses[7].position.y) ) , ( (int(ic.fingerPoseArray_.poses[8].position.x), int(ic.fingerPoseArray_.poses[8].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[5].position.x), int(ic.fingerPoseArray_.poses[5].position.y) ) , ( (int(ic.fingerPoseArray_.poses[9].position.x), int(ic.fingerPoseArray_.poses[9].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[9].position.x), int(ic.fingerPoseArray_.poses[9].position.y) ) , ( (int(ic.fingerPoseArray_.poses[10].position.x), int(ic.fingerPoseArray_.poses[10].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[10].position.x), int(ic.fingerPoseArray_.poses[10].position.y) ) , ( (int(ic.fingerPoseArray_.poses[11].position.x), int(ic.fingerPoseArray_.poses[11].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[11].position.x), int(ic.fingerPoseArray_.poses[11].position.y) ) , ( (int(ic.fingerPoseArray_.poses[12].position.x), int(ic.fingerPoseArray_.poses[12].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[9].position.x), int(ic.fingerPoseArray_.poses[9].position.y) ) , ( (int(ic.fingerPoseArray_.poses[13].position.x), int(ic.fingerPoseArray_.poses[13].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[13].position.x), int(ic.fingerPoseArray_.poses[13].position.y) ) , ( (int(ic.fingerPoseArray_.poses[14].position.x), int(ic.fingerPoseArray_.poses[14].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[14].position.x), int(ic.fingerPoseArray_.poses[14].position.y) ) , ( (int(ic.fingerPoseArray_.poses[15].position.x), int(ic.fingerPoseArray_.poses[15].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[15].position.x), int(ic.fingerPoseArray_.poses[15].position.y) ) , ( (int(ic.fingerPoseArray_.poses[16].position.x), int(ic.fingerPoseArray_.poses[16].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[13].position.x), int(ic.fingerPoseArray_.poses[13].position.y) ) , ( (int(ic.fingerPoseArray_.poses[17].position.x), int(ic.fingerPoseArray_.poses[17].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[17].position.x), int(ic.fingerPoseArray_.poses[17].position.y) ) , ( (int(ic.fingerPoseArray_.poses[18].position.x), int(ic.fingerPoseArray_.poses[18].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[18].position.x), int(ic.fingerPoseArray_.poses[18].position.y) ) , ( (int(ic.fingerPoseArray_.poses[19].position.x), int(ic.fingerPoseArray_.poses[19].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[19].position.x), int(ic.fingerPoseArray_.poses[19].position.y) ) , ( (int(ic.fingerPoseArray_.poses[20].position.x), int(ic.fingerPoseArray_.poses[20].position.y) ) ), (36,255,36), 1 )
						addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[0].position.x), int(ic.fingerPoseArray_.poses[0].position.y) ) , ( (int(ic.fingerPoseArray_.poses[17].position.x), int(ic.fingerPoseArray_.poses[17].position.y) ) ), (36,255,36), 1 )

						if (len(ic.fingerPoseArray_.poses) == 42) :
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[21].position.x), int(ic.fingerPoseArray_.poses[21].position.y) ) , ( (int(ic.fingerPoseArray_.poses[22].position.x), int(ic.fingerPoseArray_.poses[22].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[22].position.x), int(ic.fingerPoseArray_.poses[22].position.y) ) , ( (int(ic.fingerPoseArray_.poses[23].position.x), int(ic.fingerPoseArray_.poses[23].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[23].position.x), int(ic.fingerPoseArray_.poses[23].position.y) ) , ( (int(ic.fingerPoseArray_.poses[24].position.x), int(ic.fingerPoseArray_.poses[24].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[24].position.x), int(ic.fingerPoseArray_.poses[24].position.y) ) , ( (int(ic.fingerPoseArray_.poses[25].position.x), int(ic.fingerPoseArray_.poses[25].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[21].position.x), int(ic.fingerPoseArray_.poses[21].position.y) ) , ( (int(ic.fingerPoseArray_.poses[26].position.x), int(ic.fingerPoseArray_.poses[26].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[26].position.x), int(ic.fingerPoseArray_.poses[26].position.y) ) , ( (int(ic.fingerPoseArray_.poses[27].position.x), int(ic.fingerPoseArray_.poses[27].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[27].position.x), int(ic.fingerPoseArray_.poses[27].position.y) ) , ( (int(ic.fingerPoseArray_.poses[28].position.x), int(ic.fingerPoseArray_.poses[28].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[28].position.x), int(ic.fingerPoseArray_.poses[28].position.y) ) , ( (int(ic.fingerPoseArray_.poses[29].position.x), int(ic.fingerPoseArray_.poses[29].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[26].position.x), int(ic.fingerPoseArray_.poses[26].position.y) ) , ( (int(ic.fingerPoseArray_.poses[30].position.x), int(ic.fingerPoseArray_.poses[30].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[30].position.x), int(ic.fingerPoseArray_.poses[30].position.y) ) , ( (int(ic.fingerPoseArray_.poses[31].position.x), int(ic.fingerPoseArray_.poses[31].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[31].position.x), int(ic.fingerPoseArray_.poses[31].position.y) ) , ( (int(ic.fingerPoseArray_.poses[32].position.x), int(ic.fingerPoseArray_.poses[32].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[32].position.x), int(ic.fingerPoseArray_.poses[32].position.y) ) , ( (int(ic.fingerPoseArray_.poses[33].position.x), int(ic.fingerPoseArray_.poses[33].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[30].position.x), int(ic.fingerPoseArray_.poses[30].position.y) ) , ( (int(ic.fingerPoseArray_.poses[34].position.x), int(ic.fingerPoseArray_.poses[34].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[34].position.x), int(ic.fingerPoseArray_.poses[34].position.y) ) , ( (int(ic.fingerPoseArray_.poses[35].position.x), int(ic.fingerPoseArray_.poses[35].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[35].position.x), int(ic.fingerPoseArray_.poses[35].position.y) ) , ( (int(ic.fingerPoseArray_.poses[36].position.x), int(ic.fingerPoseArray_.poses[36].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[36].position.x), int(ic.fingerPoseArray_.poses[36].position.y) ) , ( (int(ic.fingerPoseArray_.poses[37].position.x), int(ic.fingerPoseArray_.poses[37].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[34].position.x), int(ic.fingerPoseArray_.poses[34].position.y) ) , ( (int(ic.fingerPoseArray_.poses[38].position.x), int(ic.fingerPoseArray_.poses[38].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[38].position.x), int(ic.fingerPoseArray_.poses[38].position.y) ) , ( (int(ic.fingerPoseArray_.poses[39].position.x), int(ic.fingerPoseArray_.poses[39].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[39].position.x), int(ic.fingerPoseArray_.poses[39].position.y) ) , ( (int(ic.fingerPoseArray_.poses[40].position.x), int(ic.fingerPoseArray_.poses[40].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[40].position.x), int(ic.fingerPoseArray_.poses[40].position.y) ) , ( (int(ic.fingerPoseArray_.poses[41].position.x), int(ic.fingerPoseArray_.poses[41].position.y) ) ), (36,255,36), 1 )
							addv = cv2.line( addv, ( int(ic.fingerPoseArray_.poses[21].position.x), int(ic.fingerPoseArray_.poses[21].position.y) ) , ( (int(ic.fingerPoseArray_.poses[38].position.x), int(ic.fingerPoseArray_.poses[38].position.y) ) ), (36,255,36), 1 )

					ic.isDrawingFinger_ = False

				if ic.isHandgrabbed_ :
					handGrabbedCount = handGrabbedCount +1
					if handGrabbedCount >= 0 and handGrabbedCount < 45 :
						cv2.putText(addv, str(captureCountDown), (430, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
					
					if handGrabbedCount == rateFloat:
						captureCountDown = 2
					elif handGrabbedCount == rateFloat *2:
						captureCountDown = 1
					elif handGrabbedCount == rateFloat *3:
						ic.isCaptured_ = True
						captureCountDown = 3
					elif handGrabbedCount == rateFloat *6:
						ic.isHandgrabbed_ = False
						handGrabbedCount = 0
						ic.grabToTakePhoto_ = False
					
				else :
					if ((ic.fingerLeftResult_ !='') and (ic.fingerRightResult_ != '')) and ( ic.fingerLeftResult_ != ic.fingerRightResult_ ) :
						if (ic.fingerLeftResult_ == 'Hi!' or ic.fingerRightResult_ == 'Hi!') :
							cv2.putText(addv, 'Grab to Take Photo', (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
						else :
							cv2.putText(addv, ic.fingerLeftResult_+ ',' + ic.fingerRightResult_, (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
					else :
						if ic.fingerLeftResult_ == 'Hi!' :
							cv2.putText(addv, 'Grab to Take Photo', (320, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
						else : 
							cv2.putText(addv, ic.fingerLeftResult_, (430, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
					# if (ic.fingerLeftResult_ == 'Grab') or (ic.fingerRightResult_ == 'Grab'):
					# 	ic.isHandgrabbed_ = True


					if ic.grabToTakePhoto_:
						ic.isHandgrabbed_ = True
						ic.grabToTakePhoto_ = False
						handEventPublisher.publish(String('capture'))

			
			if ((ic.fingerLeftResult_ !='') or (ic.fingerRightResult_ != '')) :
				if not ic.isAiModShowing_ :
					moveRequestPublisher.publish(Int32(1))
					hdmiSwitchPublisher.publish(Int32(2))
					requestDisplayPublisher.publish(Int32(11))
					#speechInfoAtWaypointPublisher.publish(String('hand'))
					ic.isAiModShowing_ = True
					handEventPublisher.publish(String('detect'))

				
				ic.aiModeShowingCount_ = 0

			if ic.isAiModShowing_ :
				if (ic.aiModeShowingCount_ >= rateFloat *30) :
					if not ic.isCharging_ :
						moveRequestPublisher.publish(Int32(2))
					hdmiSwitchPublisher.publish(Int32(1))
					requestDisplayPublisher.publish(Int32(10))
					ic.isAiModShowing_ = False
					ic.aiModeShowingCount_ = 0
				else : 
					ic.aiModeShowingCount_ = ic.aiModeShowingCount_ +1
					if not ic.isCharging_ :
						moveRequestPublisher.publish(Int32(1))


			#addv = cv2.vconcat([ic.detectionImageVertical1_, ic.detectionImageVertical2_])
			#size(addv, (1279,720), interpolation=cv2.INTER_AREA)

			fullscreen_image_message = ic.bridge.cv2_to_imgmsg(addv,"rgb8")
			resultFullscreenImagePublisher.publish(fullscreen_image_message)

		rate.sleep()

if __name__ == '__main__':
	main()