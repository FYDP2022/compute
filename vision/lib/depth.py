from typing import Any
import cv2 as cv

from lib.config import CONFIG, DebugWindows

class DepthEstimator:
  """Stereo depth estimation."""

  LEFT_WINDOW_NAME = 'Left'
  RIGHT_WINDOW_NAME = 'Right'

  def __init__(self, width: int, height: int) -> 'DepthEstimator':
    self.width = width
    self.height = height
    if DebugWindows.DEPTH in CONFIG.windows:
      cv.namedWindow(DepthEstimator.LEFT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.LEFT_WINDOW_NAME, self.width, self.height)
      cv.namedWindow(DepthEstimator.RIGHT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(DepthEstimator.RIGHT_WINDOW_NAME, self.width, self.height)

    # self.
  
  def process(self, leftFrame: Any, rightFrame: Any) -> Any:
    expected = (self.height, self.width, 3)
    if leftFrame.shape != expected or rightFrame != expected:
      raise RuntimeError('invalid stereo image shape')
    
    if DebugWindows.DEPTH in CONFIG.windows:
      cv.imshow(DepthEstimator.LEFT_WINDOW_NAME, leftFrame)
      cv.imshow(DepthEstimator.RIGHT_WINDOW_NAME, rightFrame)
    
    cv.waitKey(30)

    
