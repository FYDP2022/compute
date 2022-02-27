import os
import traceback
import cv2 as cv
from datetime import datetime
from vslam.slam import SLAM

from vslam.dynamics import DynamicsModel
from vslam.segmentation import SemanticSegmentationModel
from vslam.parameters import CalibrationParameters
from vslam.database import Feature, Observe, feature_database
from vslam.config import CONFIG, DebugWindows
from vslam.depth import DepthEstimator
from vslam.camera import StereoCamera
from vslam.state import ControlState, State

class App:
  KEYPOINT_WINDOW_NAME = 'KEYPOINT'

  def __init__(self) -> 'App':
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.camera = StereoCamera(CONFIG.width, CONFIG.height)
    self.depth = DepthEstimator(CONFIG.width, CONFIG.height, self.params)
    self.semantic = SemanticSegmentationModel()
    self.dynamics = DynamicsModel()
    self.slam = SLAM()
    self.state = State()
    self.keypoint = cv.SIFT_create()
    if DebugWindows.KEYPOINT in CONFIG.windows:
      cv.namedWindow(App.KEYPOINT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.KEYPOINT_WINDOW_NAME, CONFIG.width, CONFIG.height)

  def clear(self):
    feature_database.clear()
  
  def run(self):
    try:
      # test = feature_database.batch_select([6, 7])
      # print(test[0].probability(test[1], self.state))
      # raise RuntimeError()
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
        dynamics_delta = self.dynamics.step(self.state, ControlState())
        estimate = self.state.apply_delta(dynamics_delta)
        kp = self.keypoint.detect(grayL)
        if DebugWindows.KEYPOINT in CONFIG.windows:
          display = cv.drawKeypoints(grayL, kp, left, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
          cv.imshow(App.KEYPOINT_WINDOW_NAME, display)
        points3d = cv.reprojectImageTo3D(disparity, self.params.Q)
        features = [Feature.create(k, left, points3d, disparity) for k in kp]
        features = [f for f in features if f]
        vision_delta, probability = self.slam.step(estimate, ControlState(), features)
        estimate = estimate.apply_delta(vision_delta)
        # TODO: add variance term computed from dynamics & sensor calculation -> use SGD error to compute sigma
        processed, probability = feature_database.observe(estimate, features, Observe.PROCESSED)
        feature_database.apply_features(processed)
        self.state = estimate
        print(self.state)
        # Timing & metrics
        delta = (current_time - last_time).total_seconds()
        accum += delta
        framerate += 1.0 / delta
        n += 1
        last_time = current_time
        current_time = datetime.now()
        cv.waitKey(30)
    except Exception as e:
      print('ERROR: {}'.format(e))
      print(traceback.format_exc())
      self.camera.close()