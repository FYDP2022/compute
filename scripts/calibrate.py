import argparse
import queue
import threading
import time
import numpy as np
import cv2 as cv
import glob

class VideoCapture:
  def __init__(self, name):
    self.cap = cv.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


parser = argparse.ArgumentParser(description='Generate calibration parameters.')
parser.add_argument('camera', type=int, choices=range(0, 2), help='the index of the camera to sample from')

args = parser.parse_args()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

capture = VideoCapture('''
  nvarguscamerasrc sensor-id={}
  ! video/x-raw(memory:NVMM),
    width=1920,
    height=1080,
    format=(string)NV12,
    framerate=(fraction)10/1
  ! nvvidconv flip-method=0
  ! video/x-raw, width=640,
    height=480,
    format=(string)BGRx
  ! videoconvert
  ! video/x-raw,
    format=(string)BGR
  ! appsink
'''.format(args.camera))

if not capture.cap.isOpened():
  raise RuntimeError('failed to capture data from camera {}'.format(args.camera))

while True:
  frame = capture.read()
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  # Find the chess board corners
  ret, corners = cv.findChessboardCorners(gray, (7,6), flags=cv.CALIB_CB_FAST_CHECK)
  # If found, add object points, image points (after refining them)
  if ret == True:
    print('found')
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    # cv.drawChessboardCorners(frame, (7,7), corners2, ret)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h,  w = frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # write resulting image
    cv.imwrite('calibrate.jpg', dst)
    cv.imshow('img', dst)
    cv.waitKey(500)
    break
  else:
    print('not found')
    cv.imwrite('test.jpg', frame)
cv.destroyAllWindows()