import os
import cv2 as cv
from datetime import datetime

from vslam.parameters import CalibrationParameters
from vslam.config import CONFIG, DebugWindows
from vslam.depth import DepthEstimator
from vslam.camera import StereoCamera
from vslam.state import State

class App:
  SIFT_WINDOW_NAME = 'SIFT'

  def __init__(self) -> 'App':
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.camera = StereoCamera(CONFIG.width, CONFIG.height, self.params)
    self.depth = DepthEstimator(CONFIG.width, CONFIG.height)
    self.state = State()
    self.sift = cv.xfeatures2d.SIFT_create()
    self.last_time = datetime.now()
    self.current_time = datetime.now()
    if DebugWindows.SIFT in CONFIG.windows:
      cv.namedWindow(App.SIFT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.SIFT_WINDOW_NAME, CONFIG.width, CONFIG.height)
  
  def run(self):
    while True:
      left, right = self.camera.read()
      grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
      grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
      disparity = self.depth.process(grayL, grayR)
      kp = self.sift.detect(grayL)
      if DebugWindows.SIFT in CONFIG.windows:
        display = cv.drawKeypoints(grayL, kp, left, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow(App.SIFT_WINDOW_NAME, display)
      points3d = cv.reprojectImageTo3D(disparity, self.params.Q)
      cv.waitKey(30)
