#!/usr/bin/env python3.8
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from std_msgs.msg import Int32
from std_msgs.msg import Empty
from std_msgs.msg import String
from dotenv import load_dotenv
load_dotenv()

import rospy
import base64
import os
import sys
from datetime import date

import cloudinary
import cloudinary.uploader
import cloudinary.api
import json
import qrcode
from PIL import ImageFont, ImageDraw, Image

config = cloudinary.config(secure=True)

print("****1. Set up and configure the SDK:****\nCredentials: ", config.cloud_name, config.api_key, "\n")

today = date.today()
yymmdd = today.strftime("%Y%m%d")

class ImageHostingManager:
	def __init__ (self):

		# Publisher
		self.isRequestedImageHosting_ = False

	def requestImageHostingCallback(self, data):
		self.isRequestedImageHosting_ = True


def main():
	rospy.init_node('image_hosting_manager')
	rate = rospy.Rate(1.0/1.0) # 1hz

	ihm = ImageHostingManager()

	# Publisher
	selfCamUrlPublisher = rospy.Publisher('/self_cam_url', String, queue_size = 1)

	# Subscriber

	rospy.Subscriber("/request_image_hosting", Empty, ihm.requestImageHostingCallback)


	requestedImageHostingCount = 0
	
		
	while not rospy.is_shutdown():
		if ihm.isRequestedImageHosting_ :
			requestedImageHostingCount = requestedImageHostingCount + 1
			print('requested Image Hosting')
			if requestedImageHostingCount == 2 :
				result = cloudinary.uploader.upload_large('/home/nvidia/dcornic-media/src/video/output/video.mp4',
                                              resource_type = "video",
                                              eager_async = True)
				print(result)
				# print(result.file_id)

				# print(type(result))
				# json_object = json.loads(result)
				# print(type(json_object))
				srcURL = result["url"]

				print("****2. Upload a image****\nDelivery URL: ", srcURL, "\n")
				
				qr = qrcode.QRCode(
					error_correction=qrcode.constants.ERROR_CORRECT_M,
					box_size=8,
					border=6
				)
				qr.add_data(srcURL)
				qr.make()

				imgQr = qr.make_image()
				
				#width, height = imgQr.size
				text_pos = (26, 11)
				color = (0,0,0)
				font_size = 25
				font = ImageFont.truetype("/home/nvidia/dcornic-media/src/NanumSquare_acB.ttf", font_size)
				
				draw = ImageDraw.Draw(imgQr)
				
				draw.text(text_pos, '사진은 저장되지 않고 삭제됩니다.', outline=color, font=font) # font 설정 
				imgQr.save('/home/nvidia/dcornic-media/src/qr.png')

				urlStringMsg = String(srcURL)
				selfCamUrlPublisher.publish(urlStringMsg)

				#print(result.url)
				ihm.isRequestedImageHosting_ = False
				requestedImageHostingCount = 0

		rate.sleep()

if __name__ == '__main__':
	main()
