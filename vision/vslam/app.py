import os
from time import sleep
import traceback
from typing import Any
import cv2 as cv
from datetime import datetime
import matplotlib.image as mpimg

from vslam.client import MQTTClient
from vslam.sensors import IMUSensor
from vslam.arduino import BladeMotorCommand, OnOffCommand, SerialInterface
from vslam.slam import GradientAscentSLAM
from vslam.dynamics import DynamicsModel
from vslam.segmentation import SemanticSegmentationModel
from vslam.parameters import CalibrationParameters
from vslam.database import Feature, Observe, feature_database, occupancy_database
from vslam.config import CONFIG, AppMode, DebugWindows
from vslam.depth import DepthEstimator
from vslam.camera import StereoCamera
from vslam.state import AppState, ControlState, State
from vslam.utils import normalize, spherical_rotation_matrix

class App:
  KEYPOINT_WINDOW_NAME = 'KEYPOINT'

  def __init__(self) -> 'App':
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.camera = StereoCamera(CONFIG.width, CONFIG.height)
    self.sensor = IMUSensor()
    self.depth = DepthEstimator(CONFIG.width, CONFIG.height, self.params)
    self.semantic = SemanticSegmentationModel()
    self.dynamics = DynamicsModel()
    self.slam = GradientAscentSLAM()
    self.state = self.sensor.calibrate()
    self.app_state = AppState()
    map, x1, x2, z1, z2 = occupancy_database.visualize()
    mpimg.imsave(os.path.join(CONFIG.databasePath, 'map.png'), map)
    self.mqtt = MQTTClient(self.app_state, x1, x2, z1, z2)
    self.serial = SerialInterface(self.mqtt)
    self.mqtt.initialize(self.serial)
    self.mqtt.publish_image()
    self.mqtt.update_map_state(self.state)
    self.keypoint = cv.SIFT_create()
    if DebugWindows.KEYPOINT in CONFIG.windows:
      cv.namedWindow(App.KEYPOINT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(App.KEYPOINT_WINDOW_NAME, CONFIG.width, CONFIG.height)

  def clear(self):
    feature_database.clear()
    occupancy_database.clear()
  
  def visualize(self) -> Any:
    return occupancy_database.visualize()
  
  def run(self):
    feature_database.initialize()
    try:
      self.serial.write_message(BladeMotorCommand('OFF'))
      if CONFIG.mode == AppMode.FOLLOW:
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
          self.state = estimate
          self.mqtt.update_map_state(self.state)
          print(self.state)
          # Timing & metrics
          delta = (current_time - last_time).total_seconds()
          accum += delta
          framerate += 1.0 / delta
          n += 1
          last_time = current_time
          current_time = datetime.now()
          cv.waitKey(30)
      elif CONFIG.mode == AppMode.BLADE:
        last = self.app_state.active
        while True:
          if last != self.app_state.active:
            if self.app_state.active:
              self.serial.write_message(BladeMotorCommand('ON'))
            else:
              self.serial.write_message(BladeMotorCommand('OFF'))
    except Exception as e:
      print('ERROR: {}'.format(e))
      print(traceback.format_exc())
    self.close()
  
  def close(self):
    self.serial.write_message(BladeMotorCommand('OFF'))
    self.sensor.close()
    self.camera.close()
