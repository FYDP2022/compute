import os
import cv2 as cv
from datetime import datetime

from vslam.segmentation import SemanticSegmentationModel
from vslam.parameters import CalibrationParameters
from vslam.database import Feature, feature_database
from vslam.config import CONFIG, DebugWindows
from vslam.depth import DepthEstimator
from vslam.camera import StereoCamera
from vslam.state import State

class App:
  KEYPOINT_WINDOW_NAME = 'KEYPOINT'

  def __init__(self) -> 'App':
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.camera = StereoCamera(CONFIG.width, CONFIG.height)
    self.depth = DepthEstimator(CONFIG.width, CONFIG.height, self.params)
    self.semantic = SemanticSegmentationModel()
    self.state = State()
    self.keypoint = cv.SIFT_create()
    if DebugWindows.KEYPOINT in CONFIG.windows:
      cv.namedWindow(App.KEYPOINT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.KEYPOINT_WINDOW_NAME, CONFIG.width, CONFIG.height)
  
  def run(self):
    accum = 0.0
    framerate = 0.0
    n = 0
    last_time = datetime.now()
    current_time = datetime.now()
    while True:
      metrics = accum >= CONFIG.interval
      if metrics:
        print("Average framerate: {}".format(framerate / n))
        accum = 0.0
        framerate = 0.0
        n = 0
      left, right = self.camera.read()
      grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
      grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
      disparity = self.depth.process(grayL, grayR)
      kp = self.keypoint.detect(grayL)
      if DebugWindows.KEYPOINT in CONFIG.windows:
        display = cv.drawKeypoints(grayL, kp, left, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow(App.KEYPOINT_WINDOW_NAME, display)
      points3d = cv.reprojectImageTo3D(disparity, self.params.Q)
      features = [Feature.create(k, left, points3d, disparity) for k in kp]
      adjust, processed = feature_database.process_features(self.state, features)
      feature_database.apply_features(adjust.negate(), processed)
      # Timing & metrics
      delta = (current_time - last_time).total_seconds()
      accum += delta
      framerate += 1.0 / delta
      n += 1
      last_time = current_time
      current_time = datetime.now()
      cv.waitKey(30)
