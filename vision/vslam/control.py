from typing import Optional, Tuple
import numpy as np
import cv2 as cv
from scipy import ndimage

from vslam.state import ControlAction
from vslam.segmentation import Material
from vslam.arduino import DriveMotorCommand, SerialInterface

class FollowControl:
  STOP_DISTANCE = 1.0

  def track(self, semantic, points3d) -> Tuple[ControlAction, float]:
    THRESHOLD = 500
    
    height, width = semantic.shape
    mask = semantic == Material.PERSON
    if mask.sum() >= THRESHOLD:
      scaled = cv.resize(mask, dsize=(width, height), interpolation=cv.INTER_NEAREST)
      depth = np.ma.mean(points3d[:, :, 2], mask=scaled) / 1000.0
      if depth < FollowControl.STOP_DISTANCE:
        return ControlAction.NONE, depth
      else:
        _, x = ndimage.measurements.center_of_mass(mask)
        if x > 1.10 * (width / 2):
          return ControlAction.TURN_RIGHT, depth
        elif x < 0.9 * (width / 2):
          return ControlAction.TURN_LEFT, depth
        else:
          return ControlAction.MOVE_FORWARD, depth
    return ControlAction.NONE, 0.0
  
  def handle_action(self, serial: SerialInterface, action: ControlAction, depth: float):
    distance = int((depth - FollowControl.STOP_DISTANCE) * 100)
    if action == ControlAction.NONE:
      serial.write_message(DriveMotorCommand('STOP', 0, 0))
    elif action == ControlAction.MOVE_FORWARD:
      serial.write_message(DriveMotorCommand('FORWARD', 0, distance))
    elif action == ControlAction.TURN_LEFT:
      serial.write_message(DriveMotorCommand('FWD_LEFT', 0, distance))
    elif action == ControlAction.TURN_RIGHT:
      serial.write_message(DriveMotorCommand('FWD_RIGHT', 0, distance))
