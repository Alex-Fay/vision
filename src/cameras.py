#! /usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('vision')
import sys
import rospkg
rospack = rospkg.RosPack()
visionpath = rospack.get_path('vision') + "/src/localization"
sys.path.append(visionpath)
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gate import Gate
from buoy import Buoy
from localize import Localize
import copy
import numpy as np
from time import time

class Cameras:
	
	def __init__(self):
		rospy.init_node('Cameras', anonymous=True)

		self.cameraMap = {'down': None, 'front': None}
		self.cvImage0 = None
		self.cvImage1 = None
		self.bridge = CvBridge()
		self.image0Sub = rospy.Subscriber("usb_cam_0/image_raw", Image, self.callback0)
		self.image1Sub = rospy.Subscriber("usb_cam_1/image_raw", Image, self.callback1)

		self.gatePub = rospy.Publisher("out_gate", Image,queue_size=1)
		self.buoyPub = rospy.Publisher("out_buoy", Image, queue_size=1)

		self.cam0Ready = False
		self.cam1Ready = False
		
		self.loadCameras()
	def loadCameras(self):
		#load camera names from /dev/video[n]
		try:
			f0 = open("/sys/class/video4linux/video0/name")
		except Exception as e:
			print("Whoops, couldn't find camera 0!")
			return
		try:
			f1 = open("/sys/class/video4linux/video1/name")
		except Exception as e:
			print("Whoops, couldn't find camera 1! Assuming only camera is front-facing")
			self.cameraMap['front'] = 0
			self.cameraMap['down'] = None
			return
			
		if(f0.read(1) == 'U'):
			#0 corresponds to /dev/video0, 1 to /dev/video1
			self.cameraMap['down'] = 0
			self.cameraMap['front'] = 1
		else:
			self.cameraMap['front'] = 0
			self.cameraMap['down'] = 1	
		
	
	def getFrame(self, camera):
		try:
			dev = self.cameraMap[camera]
		except Exception as e:
			print("Error in get frame:", e)
		if dev is None:
			print("Tried to use None camera")
			return
		if(dev == 0):
			return self.cvImage0
		else:
			return self.cvImage1

	def getFrontFrame(self):
		return self.getFrame('front')
		
	def getDownFrame(self):
		return self.getFrame('down')
		
	def callback0(self, data):
		self.cam0Ready = True
		try:
			cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		self.cvImage0 = cv2.resize(cvImage,(640,480))

	def callback1(self, data):
		self.cam1Ready = True

		try:
			cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		self.cvImage1 = cv2.resize(cvImage,(640,480))

	def buoyLinePrediction(self, img, angle, green=True):
		x = angle
		# experimentally found camera mapping from angle to pixel.cubic fit
		p = 0.000763833735942*x**3 + 0.002223064300152*x**2+6.65366583093281*x+318.3566
		p = int(p)
		if green:
			cv2.line(img, (p,0), (p,480), (0,255,0),3)
		else:
			cv2.line(img, (p,0),(p,480),(255,0,0),3)

def main(args):
	rospy.init_node('Cameras', anonymous=True)
	cams = Cameras()
	gate = Gate()
	buoy = Buoy()
	loc = Localize()
	rate = rospy.Rate(30)
	
	lastTime = time()*1000
	try:
		while True:
			if rospy.is_shutdown():
				rospy.logerr("FUCK")
				break
			if(cams.cam0Ready and (cams.cam1Ready or cams.cameraMap['down'] is None)):
				img_gate = np.copy(cams.getFrontFrame())
				img_buoy = np.copy(img_gate)
				if ((img_gate == np.copy(None)).any() or img_gate is None):
					rospy.logwarn("none image recieved")
					continue
				bars = gate.findBars(img_gate)
				loc.updateGate([bars[1], bars[2], bars[3]])
						
				buoys = buoy.mainImg(img_buoy)
				img_buoy = buoy.getResultImg()
				cams.buoyLinePrediction(img_buoy, loc.firstBuoyYaw.getPredictedState()[0])
				cams.buoyLinePrediction(img_buoy, loc.secondBuoyYaw.getPredictedState()[0], green=False)
				loc.updateBuoy(buoys)

				try:
					cams.buoyPub.publish(cams.bridge.cv2_to_imgmsg(img_buoy, "bgr8"))
					cams.gatePub.publish(cams.bridge.cv2_to_imgmsg(img_gate,"bgr8"))
				except Exception as e:
					rospy.logerr("EXCEPTION IN CAMERAS: ")
					rospy.logerr(e)
				rospy.logwarn("LOOP TIME %d", time()*1000 - lastTime)
				lastTime = time()*1000

			else:
				pass
				#print("NOT READY")
				
	except KeyboardInterrupt:
		rospy.logerr("Shutting down")
	cv2.destroyAllWindows()
	rospy.logerr("KILL")
if __name__==("__main__"):
	main(sys.argv)
