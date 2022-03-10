import os
from typing import Optional, Tuple
import numpy as np
import cv2 as cv
from imutils.object_detection import non_max_suppression
from imutils import paths
from scipy import ndimage
import matplotlib.image as mpimg

from vslam.config import CONFIG, DebugWindows
from vslam.parameters import CameraParameters
from vslam.state import ControlAction
from vslam.semantic import Material
from vslam.arduino import DriveMotorCommand, SerialInterface

class FollowControl:
  STOP_DISTANCE = 1.0
  CONTROL_WINDOW_NAME = 'CONTROL'

  def __init__(self) -> None:
    self.hog = cv.HOGDescriptor()
    self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    if DebugWindows.CONTROL in CONFIG.windows:
      cv.namedWindow(FollowControl.CONTROL_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(FollowControl.CONTROL_WINDOW_NAME, CONFIG.width, CONFIG.height)


  def track(self, image, points3d) -> Tuple[ControlAction, float]:
    THRESHOLD = 0.1
    rects, weights = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    max = None
    for idx, weight in enumerate(weights):
      if max is None or weight[0] > weights[idx][0]:
        max = idx
    print(weights)
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # for (xA, yA, xB, yB) in pick:
    #   cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    if DebugWindows.CONTROL in CONFIG.windows:
      for (x, y, w, h) in rects:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
      mpimg.imsave(os.path.join(CONFIG.databasePath, 'test.png'), image)
      cv.imshow(FollowControl.CONTROL_WINDOW_NAME, image)
    
    if max is not None and weights[max][0] > THRESHOLD:
      x, y, w, h = rects[max]
      box = points3d[x:x+w, y:y+h, 2] / 10000.0
      box[box < 0.0] = np.inf
      masked = np.ma.masked_where(box == np.inf, box)
      depth = np.ma.mean(masked)
      xmid = x + w / 2
      angle = (CameraParameters.FOVX / 2) * (xmid - CONFIG.width / 2) / (CONFIG.width / 2)
      if xmid > 1.10 * (CONFIG.width / 2):
        if depth < FollowControl.STOP_DISTANCE:
          return ControlAction.TURN_RIGHT, angle
        else:
          return ControlAction.MOVE_RIGHT, angle
      elif xmid < 0.9 * (CONFIG.width / 2):
        if depth < FollowControl.STOP_DISTANCE:
          return ControlAction.TURN_LEFT, angle
        else:
          return ControlAction.MOVE_LEFT, angle
      else:
        if depth < FollowControl.STOP_DISTANCE:
          return ControlAction.NONE, depth
        else:
          return ControlAction.MOVE_FORWARD, depth
    return ControlAction.NONE, 0.0
  
  def execute_action(self, serial: SerialInterface, action: ControlAction, depth_or_angle: float):
    distance = int((depth_or_angle - FollowControl.STOP_DISTANCE) * 100)
    if action == ControlAction.NONE:
      serial.write_message(DriveMotorCommand('STOP', -7, 0))
    elif action == ControlAction.MOVE_FORWARD:
      serial.write_message(DriveMotorCommand('FORWARD', -7, int(distance)))
    elif action == ControlAction.MOVE_LEFT:
      serial.write_message(DriveMotorCommand('FWD_LEFT', -7, int(depth_or_angle)))
    elif action == ControlAction.MOVE_RIGHT:
      serial.write_message(DriveMotorCommand('FWD_RIGHT', -7, int(depth_or_angle)))
    elif action == ControlAction.TURN_LEFT:
      serial.write_message(DriveMotorCommand('POINT_LEFT', -7, int(depth_or_angle)))
    elif action == ControlAction.TURN_RIGHT:
      serial.write_message(DriveMotorCommand('POINT_RIGHT', -7, int(depth_or_angle)))
