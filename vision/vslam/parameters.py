import os
from typing import Any, Tuple
from dataclasses import dataclass
import cv2 as cv
import numpy as np

class CameraParameters:
  BASELINE = 0.06 # 60mm baseline length
  FOCAL_LENGTH = 0.0026 # 2.6mm focal length
  FOVX = 73 # Horizontal field of view (degrees)
  FOVY = 50 # Vertical field of view (degrees)

@dataclass
class CalibrationParameters:
  mapL1: Any
  mapL2: Any
  mapR1: Any
  mapR2: Any
  Q: Any

  def load(path: str) -> 'CalibrationParameters':
    cv_file = cv.FileStorage(os.path.join(path, 'calibration.xml'), cv.FILE_STORAGE_READ)
    Q = cv_file.getNode("Q").mat()
    return CalibrationParameters(
      np.load(os.path.join(path, 'mapL1.npy')),
      np.load(os.path.join(path, 'mapL2.npy')),
      np.load(os.path.join(path, 'mapR1.npy')),
      np.load(os.path.join(path, 'mapR2.npy')),
      Q
    )

class UndistortRectifier:
  """Applies stereo calibration to image."""

  def __init__(self, path: str = './data/calibration') -> 'UndistortRectifier':
    self.params = CalibrationParameters.load(path)

  def undistortRectify(self, imageLeft: Any, imageRight: Any) -> Tuple[Any, Any]:
    undistorted_rectified_left = cv.remap(imageLeft, self.params.mapL1, self.params.mapL2, cv.INTER_LINEAR)
    undistorted_rectified_right = cv.remap(imageRight, self.params.mapR1, self.params.mapR2, cv.INTER_LINEAR)
    return undistorted_rectified_left, undistorted_rectified_right
