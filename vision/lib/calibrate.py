from dataclasses import dataclass
import os
from typing import Any, Tuple
import cv2 as cv
import numpy as np

@dataclass
class CalibrationParameters:
  mapL1: Any
  mapL2: Any
  mapR1: Any
  mapR2: Any

  def load(path: str) -> 'CalibrationParameters':
    CalibrationParameters(
      np.load(os.path.join(path, 'mapL1')),
      np.load(os.path.join(path, 'mapL2')),
      np.load(os.path.join(path, 'mapR1')),
      np.load(os.path.join(path, 'mapL1'))
    )

  def save(self, path: str):
    np.save(os.path.join(path, 'mapL1'), self.mapL1)
    np.save(os.path.join(path, 'mapL2'), self.mapL2)
    np.save(os.path.join(path, 'mapR1'), self.mapR2)
    np.save(os.path.join(path, 'mapL1'), self.mapL1)

class UndistortRectifier:
  """Applies stereo calibration to image."""

  def __init__(self, path: str = './data/calibration') -> 'UndistortRectifier':
    self.params = CalibrationParameters.load(path)

  def undistortRectify(self, imageLeft: Any, imageRight: Any) -> Tuple[Any, Any]:
    undistorted_rectified_left = cv.remap(imageLeft, self.params.mapL1, self.params.mapL2, cv.INTER_LINEAR)
    undistorted_rectified_right = cv.remap(imageRight, self.params.mapR1, self.params.mapR2, cv.INTER_LINEAR)
    return undistorted_rectified_left, undistorted_rectified_right
