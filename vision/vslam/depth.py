import os
from typing import Any
import cv2 as cv
import numpy as np

from vslam.config import CONFIG, DebugWindows
from vslam.parameters import CalibrationParameters

class DepthEstimator:
  """Stereo depth estimation."""

  DEPTH_WINDOW_NAME = 'SGBM'
  LEFT_REMAP_WINDOW_NAME = 'REMAP.LEFT'
  RIGHT_REMAP_WINDOW_NAME = 'REMAP.RIGHT'
  MIN_DISPARITY = 16
  MAX_DISPARITY = 128
  APPLY_COLORMAP = False

  def __init__(self, width: int, height: int) -> 'DepthEstimator':
    self.width = width
    self.height = height
    if DebugWindows.DEPTH in CONFIG.windows:
      cv.namedWindow(DepthEstimator.DEPTH_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.DEPTH_WINDOW_NAME, self.width, self.height)
    if DebugWindows.REMAP in CONFIG.windows:
      cv.namedWindow(DepthEstimator.LEFT_REMAP_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.LEFT_REMAP_WINDOW_NAME, self.width, self.height)
      cv.namedWindow(DepthEstimator.RIGHT_REMAP_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.RIGHT_REMAP_WINDOW_NAME, self.width, self.height)
    self.params = CalibrationParameters.load(os.path.join(CONFIG.dataPath, 'calibration'))
    window_size = 3
    self.estimator = cv.StereoSGBM_create(
      minDisparity=DepthEstimator.MIN_DISPARITY,
      numDisparities=DepthEstimator.MAX_DISPARITY,
      blockSize=16,
      P1=8 * 3 * window_size ** 2,
      P2=32 * 3 * window_size ** 2,
      disp12MaxDiff=6,
      uniquenessRatio=5,
      speckleRange=32,
      speckleWindowSize=100
    )
  
  def applyImageRemap(self, left: Any, right: Any) -> Any:
    undistorted_rectifiedL = cv.remap(left, self.params.mapL1, self.params.mapL2, cv.INTER_LINEAR)
    undistorted_rectifiedR = cv.remap(right, self.params.mapR1, self.params.mapR2, cv.INTER_LINEAR)
    return undistorted_rectifiedL, undistorted_rectifiedR

  def process(self, left_gray: Any, right_gray: Any) -> Any:
    expected = (self.height, self.width)
    if left_gray.shape != expected or right_gray.shape != expected:
      raise RuntimeError('invalid stereo image shape')

    undistorted_rectifiedL, undistorted_rectifiedR = self.applyImageRemap(left_gray, right_gray)

    if DebugWindows.REMAP in CONFIG.windows:
      cv.imshow(DepthEstimator.LEFT_REMAP_WINDOW_NAME, undistorted_rectifiedL)
      cv.imshow(DepthEstimator.RIGHT_REMAP_WINDOW_NAME, undistorted_rectifiedR)

    disparity = self.estimator.compute(undistorted_rectifiedL, undistorted_rectifiedR)
    cv.filterSpeckles(disparity, 0, 40, DepthEstimator.MAX_DISPARITY)
    disparity = (disparity / 16.0 - DepthEstimator.MIN_DISPARITY) / DepthEstimator.MAX_DISPARITY

    if DebugWindows.DEPTH in CONFIG.windows:
      display = disparity
      if DepthEstimator.APPLY_COLORMAP:
        display = cv.applyColorMap(
          (disparity * 255.0).astype(np.uint8),
          cv.COLORMAP_HOT
        )
      cv.imshow(DepthEstimator.DEPTH_WINDOW_NAME, display)

    return disparity
