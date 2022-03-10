import os
import traceback
from typing import Any
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.image as mpimg
use_cuda = torch.cuda.is_available()
from vslam.sensors import IMUSensor

from vslam.slam import GradientAscentSLAM
from vslam.dynamics import DynamicsModel
from vslam.semantic import SemanticSegmenation, Net
from vslam.parameters import CalibrationParameters
from vslam.database import Feature, Observe, feature_database, occupancy_database
from vslam.config import CONFIG, DebugWindows
from vslam.depth import DepthEstimator
from vslam.camera import StereoCamera
from vslam.state import ControlState, State
from vslam.utils import normalize, spherical_rotation_matrix

class App:
  KEYPOINT_WINDOW_NAME = 'KEYPOINT'
  SEMANTIC_WINDOW_NAME = 'SEMANTIC'

  def __init__(self) -> 'App':
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.camera = StereoCamera(CONFIG.width, CONFIG.height)
    self.semanticSegmentation = SemanticSegmenation()
    model_path = os.path.abspath(os.path.join(os.path.dirname('main.py'), 'semantic', 'checkpoint_test.tar'))
    self.semantic = Net()
    device =  torch.device("cuda" if use_cuda else "cpu")
    model = torch.load(model_path)
    self.trained_model = model['model']
    self.trained_model = self.trained_model.to(device).eval()
    self.sensor = IMUSensor()
    self.depth = DepthEstimator(CONFIG.width, CONFIG.height, self.params)
    self.dynamics = DynamicsModel()
    self.slam = GradientAscentSLAM()
    self.state = self.sensor.calibrate()
    self.keypoint = cv.SIFT_create()
    if DebugWindows.KEYPOINT in CONFIG.windows:
      cv.namedWindow(App.KEYPOINT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.KEYPOINT_WINDOW_NAME, CONFIG.width, CONFIG.height)
    if DebugWindows.SEMANTIC in CONFIG.windows:
      cv.namedWindow(App.SEMANTIC_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.SEMANTIC_WINDOW_NAME, CONFIG.width, CONFIG.height)


  def clear(self):
    feature_database.clear()
    occupancy_database.clear()
  
  def visualize(self) -> Any:
    return occupancy_database.visualize()
  
  def run(self):
    feature_database.initialize()
    try:
      test = feature_database.batch_select([6, 7])
      print(test[0].probability(test[1], self.state))
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
        image = left.copy()
        grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        disparity = self.depth.process(grayL, grayR)
        dynamics_delta, dynamics_deviation = self.dynamics.step(self.state, ControlState())
        estimate = self.state.apply_delta(dynamics_delta)
        sensor_delta, sensor_deviation = self.sensor.step(self.state)
        estimate = self.state.apply_delta(sensor_delta)
        kp = self.keypoint.detect(grayL)
        if DebugWindows.KEYPOINT in CONFIG.windows:
          display = cv.drawKeypoints(grayL, kp, image.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
          cv.imshow(App.KEYPOINT_WINDOW_NAME, display)
        points3d = cv.reprojectImageTo3D(disparity, self.params.Q)
        features = [Feature.create(k, image, points3d, disparity) for k in kp]
        features = [f for f in features if f]
        vision_delta, probability, deviation = self.slam.step(estimate, sensor_deviation, features)
        estimate = estimate.apply_delta(vision_delta)
        estimate = estimate.apply_deviation(deviation)
        processed, probability = feature_database.observe(estimate, sensor_deviation, features, Observe.PROCESSED)
        feature_database.apply_features(processed)
        occupancy_database.apply_voxels(image, points3d, disparity, estimate)
        #TODO add semenatic Image to pipeline
        semantic_Image = self.getSemanticImage(image)
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
    self.close()
  
  def getSemanticImage(self, image):
    img = cv.resize(image, (500, 500), interpolation=cv.INTER_AREA)
    imgProcessed = self.semanticSegmentation.process_inp_image(img)
    semanticImage = self.semanticSegmentation.get_forwarded_image(self.trained_model, imgProcessed.cuda())
    return semanticImage

  def close(self):
    self.sensor.close()
    self.camera.close()
