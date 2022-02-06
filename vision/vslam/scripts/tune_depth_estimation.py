# SOURCE: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

import numpy as np 
import cv2

import vslam

stereo_cam = vslam.camera.StereoCamera(960, 540)
depth_estimator = vslam.depth.DepthEstimator(960, 540)

def nothing(x):
    pass

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',960,540)

cv2.createTrackbar('numDisparities','disp',8,17,nothing)
cv2.createTrackbar('blockSize','disp',8,50,nothing)
cv2.createTrackbar('P1','disp',8*3*3*3,2000,nothing)
cv2.createTrackbar('P2','disp',32*3*3*3,2000,nothing)
# cv2.createTrackbar('preFilterType','disp',1,1,nothing)
# cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',0,62,nothing)
# cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',5,100,nothing)
cv2.createTrackbar('speckleRange','disp',32,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',50,50,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',6,25,nothing)
cv2.createTrackbar('minDisparity','disp',16,25,nothing)
cv2.createTrackbar('mode','disp',2,3,nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM_create()

while True:

	# Capturing and storing left and right camera images
	l, r = stereo_cam.read()
	
	# Proceed only if the frames have been captured
	imgR_gray = cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
	imgL_gray = cv2.cvtColor(l,cv2.COLOR_BGR2GRAY)

	# Applying stereo image rectification on the left image
	Left_nice, Right_nice = depth_estimator.applyImageRemap(imgL_gray, imgR_gray)

	# Updating the parameters based on the trackbar positions
	numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
	blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
	P1 = cv2.getTrackbarPos('P1','disp')
	P2 = cv2.getTrackbarPos('P2','disp')
	# preFilterType = cv2.getTrackbarPos('preFilterType','disp')
	# preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
	preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
	# textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
	uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
	speckleRange = cv2.getTrackbarPos('speckleRange','disp')
	speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
	disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
	minDisparity = cv2.getTrackbarPos('minDisparity','disp')
	mode = cv2.getTrackbarPos('mode','disp')
	
	# Setting the updated parameters before computing disparity map
	stereo.setNumDisparities(numDisparities)
	stereo.setBlockSize(blockSize)
	stereo.setP1(P1)
	stereo.setP2(P2)
	# stereo.setPreFilterType(preFilterType)
	# stereo.setPreFilterSize(preFilterSize)
	stereo.setPreFilterCap(preFilterCap)
	# stereo.setTextureThreshold(textureThreshold)
	stereo.setUniquenessRatio(uniquenessRatio)
	stereo.setSpeckleRange(speckleRange)
	stereo.setSpeckleWindowSize(speckleWindowSize)
	stereo.setDisp12MaxDiff(disp12MaxDiff)
	stereo.setMinDisparity(minDisparity)
	stereo.setMode(mode)

	# Calculating disparity using the StereoBM algorithm
	disparity = stereo.compute(Left_nice,Right_nice)
	# NOTE: Code returns a 16bit signed single channel image,
	# CV_16S containing a disparity map scaled by 16. Hence it 
	# is essential to convert it to CV_32F and scale it down 16 times.

	# Converting to float32 
	disparity = disparity.astype(np.float32)

	# Scaling down the disparity values and normalizing them 
	disparity = (disparity/16.0 - minDisparity)/numDisparities

	# Displaying the disparity map
	cv2.imshow("disp",disparity)

	# Close window using esc key
	if cv2.waitKey(1) == 27:
		break
