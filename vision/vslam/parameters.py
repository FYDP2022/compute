import os
from typing import Any, Tuple
import cv2 as cv
import numpy as np

class CalibrationParameters:
  def __init__(self, mapL1: Any, mapL2: Any, mapR1: Any, mapR2: Any) -> 'CalibrationParameters':
    self.mapL1 = mapL1
    self.mapL2 = mapL2
    self.mapR1 = mapR1
    self.mapR2 = mapR2

  def load(path: str) -> 'CalibrationParameters':
    return CalibrationParameters(
      np.load(os.path.join(path, 'mapL1.npy')),
      np.load(os.path.join(path, 'mapL2.npy')),
      np.load(os.path.join(path, 'mapR1.npy')),
      np.load(os.path.join(path, 'mapL1.npy'))
    )

  def save(self, path: str):
    np.save(os.path.join(path, 'mapL1.npy'), self.mapL1)
    np.save(os.path.join(path, 'mapL2.npy'), self.mapL2)
    np.save(os.path.join(path, 'mapR1.npy'), self.mapR2)
    np.save(os.path.join(path, 'mapL1.npy'), self.mapL1)

class UndistortRectifier:
  """Applies stereo calibration to image."""

  def __init__(self, path: str = './data/calibration') -> 'UndistortRectifier':
    self.params = CalibrationParameters.load(path)

  def undistortRectify(self, imageLeft: Any, imageRight: Any) -> Tuple[Any, Any]:
    undistorted_rectified_left = cv.remap(imageLeft, self.params.mapL1, self.params.mapL2, cv.INTER_LINEAR)
    undistorted_rectified_right = cv.remap(imageRight, self.params.mapR1, self.params.mapR2, cv.INTER_LINEAR)
    return undistorted_rectified_left, undistorted_rectified_right
