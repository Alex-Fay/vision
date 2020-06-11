#SURFimplementation @Alex Fay 6/3/2020 - 6/10/2020
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#initiate orb and read test image
img_sample = cv.imread('rock_dude.png') #TODO: apply to 3 input images
img_sample = img_sample[0: 250, 0: 300]
plt.imshow(img_sample)
orb = cv.ORB_create(nfeatures = 1500)
keypoints_sample = orb.detect(img_sample,None)

#input image 
img_feed = cv.imread('input.jpg') #TODO: change to video
orb = cv.ORB_create(nfeatures = 1500)
keypoints_feed = orb.detect(img_sample, None)

# compute the descriptors with ORB
keypoints_sample, des_sample = orb.compute(img_sample, keypoints_sample)
keypoints_input, des_input = orb.compute(img_feed, keypoints_feed)

# match images
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des_sample, des_input)
matches = sorted(matches, key = lambda x:x.distance)
img_match = cv.drawMatches(img_feed, keypoints_feed, img_sample, keypoints_input, matches[:30], None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# draw keypoints location, excludes size and orientation
mapImg = cv.drawKeypoints(img_sample, keypoints_sample, None, color=(0,255,0), flags=0)
plt.imshow(mapImg), plt.show()
plt.imshow(img_match)
plt.show()
