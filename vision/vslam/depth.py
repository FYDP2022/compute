import os
from typing import Any
import cv2 as cv
import numpy as np

from vslam.config import CONFIG, DebugWindows
from vslam.parameters import CalibrationParameters

class DepthEstimator:
  """Stereo depth estimation."""

  WINDOW_NAME = 'SGBM'
  MAX_DISPARITY = 128
  BLOCK_SIZE = 21
  APPLY_COLORMAP = False

  def __init__(self, width: int, height: int) -> 'DepthEstimator':
    self.width = width
    self.height = height
    if DebugWindows.DEPTH in CONFIG.windows:
      cv.namedWindow(DepthEstimator.WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.WINDOW_NAME, self.width, self.height)
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    self.estimator = cv.StereoSGBM_create(0, DepthEstimator.MAX_DISPARITY, DepthEstimator.BLOCK_SIZE)

  def process(self, left_gray: Any, right_gray: Any) -> Any:
    expected = (self.height, self.width)
    if left_gray.shape != expected or right_gray.shape != expected:
      raise RuntimeError('invalid stereo image shape')

    undistorted_rectifiedL = cv.remap(left_gray, self.params.mapL1, self.params.mapL2, cv.INTER_LINEAR)
    undistorted_rectifiedR = cv.remap(right_gray, self.params.mapR1, self.params.mapR2, cv.INTER_LINEAR)

    disparity = self.estimator.compute(undistorted_rectifiedL, undistorted_rectifiedR)
    cv.filterSpeckles(disparity, 0, 40, DepthEstimator.MAX_DISPARITY)
    _, disparity = cv.threshold(disparity, 0, DepthEstimator.MAX_DISPARITY * 16, cv.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.0).astype(np.uint8) * (256.0 / DepthEstimator.MAX_DISPARITY)
    if DepthEstimator.APPLY_COLORMAP:
      output = cv.applyColorMap(
        disparity_scaled.astype(np.uint8),
        cv.COLORMAP_HOT
      )
    else:
      output = disparity_scaled

    if DebugWindows.DEPTH in CONFIG.windows:
      cv.imshow(DepthEstimator.WINDOW_NAME, output)

    return output
