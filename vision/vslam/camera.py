from enum import IntEnum
import math
import queue
import signal
import sys
import threading
from typing import Any, List, Tuple
import cv2 as cv
import numpy as np
from vslam.parameters import CameraParameters
from vslam.state import Vector

from vslam.config import CONFIG, DebugWindows

def angle_axis(axis, theta):
  """
  Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  """
  axis = np.asarray(axis)
  axis = axis / math.sqrt(np.dot(axis, axis))
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([
    [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
  ])

def normalize(vector: np.array) -> np.array:
  norm = np.linalg.norm(vector)
  if norm == 0.0:
    return vector
  return vector / norm 

def angle_between(v1, v2):
  v1_u = normalize(v1)
  v2_u = normalize(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def projection(x: np.array, y: np.array) -> np.array:
  """
  Projection of x onto y.
  """
  return np.dot(x, y) / np.linalg.norm(y)

def pixel_ray(direction: Vector, i: float, j: float) -> np.array:
  """
  Returns the ray vector given by the direction projected from the camera
  corresponding to the given image coords
  """
  xang = -((i / CONFIG.width) * CameraParameters.FOVX - CameraParameters.FOVX / 2.0)
  yang = -((j / CONFIG.height) * CameraParameters.FOVY - CameraParameters.FOVY / 2.0)
  rotation = np.dot(angle_axis(direction, math.radians(xang)), angle_axis(direction, math.radians(yang)))
  return normalize(np.dot(rotation, direction))

class CameraIndex(IntEnum):
  RIGHT = 0
  LEFT = 1

class StereoCamera:
  """Streams data from binocular camera sensor"""

  LEFT_WINDOW_NAME = 'CAMERA.LEFT'
  RIGHT_WINDOW_NAME = 'CAMERA.RIGHT'
  WIDTH = 1920
  HEIGHT = 1080

  def __init__(self, width: int, height: int) -> 'StereoCamera':
    self.width = width
    self.height = height
    self.reverse = False
    self.stopped = False
    self.queue = (queue.Queue(), queue.Queue())
    self.barrier = threading.Barrier(2)
    self.capture: List[cv.VideoCapture] = [None, None]
    self.capture[CameraIndex.LEFT] = cv.VideoCapture(self._cameraString(CameraIndex.LEFT))
    self.capture[CameraIndex.RIGHT] = cv.VideoCapture(self._cameraString(CameraIndex.RIGHT))
    signal.signal(signal.SIGINT, self._signalHandler)
    if not self.capture[CameraIndex.LEFT].isOpened():
      raise RuntimeError('failed to capture data from left camera')
    if not self.capture[CameraIndex.RIGHT].isOpened():
      raise RuntimeError('failed to capture data from right camera')
    self.t1 = threading.Thread(target=self._reader, args=[CameraIndex.LEFT])
    self.t1.start()
    self.t2 = threading.Thread(target=self._reader, args=[CameraIndex.RIGHT])
    self.t2.start()
    if DebugWindows.CAMERA in CONFIG.windows:
      cv.namedWindow(StereoCamera.LEFT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(StereoCamera.LEFT_WINDOW_NAME, self.width, self.height)
      cv.namedWindow(StereoCamera.RIGHT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(StereoCamera.RIGHT_WINDOW_NAME, self.width, self.height)

  def close(self):
    self.stopped = True
    self.t1.join()
    self.t2.join()
    self.capture[CameraIndex.LEFT].release()
    self.capture[CameraIndex.RIGHT].release()

  def _signalHandler(self, sig, frame):
    self.close()
    sys.exit(0)

  def _reader(self, camera: CameraIndex):
    while not self.stopped:
      ret, frame = self.capture[camera].read()
      self.barrier.wait()
      if not ret:
        raise RuntimeError('failed to read camera sensor: {}'.format(camera))
      if not self.queue[camera].empty():
        try:
          # discard previous (unprocessed) frame
          self.queue[camera].get_nowait()
        except queue.Empty:
          pass
      self.queue[camera].put(frame)
  
  def _cameraString(self, camera: CameraIndex) -> str:
    return """
      nvarguscamerasrc sensor-id={}
      ! video/x-raw(memory:NVMM),
        width={},
        height={},
        format=(string)NV12,
        framerate=(fraction)10/1
      ! nvvidconv flip-method=0
      ! video/x-raw,
        width={},
        height={},
        format=(string)BGRx
      ! videoconvert
      ! video/x-raw,
        format=(string)BGR
      ! appsink
    """.format(camera, StereoCamera.WIDTH, StereoCamera.HEIGHT, self.width, self.height)

  def swapCameras(self):
    self.reverse = not self.reverse

  def _getFrame(self):
    if self.reverse:
      return (self.queue[CameraIndex.RIGHT].get(), self.queue[CameraIndex.LEFT].get())
    else:
      return (self.queue[CameraIndex.LEFT].get(), self.queue[CameraIndex.RIGHT].get())
  
  def read(self) -> Tuple[Any, Any]:
    left, right = self._getFrame()
    if DebugWindows.CAMERA in CONFIG.windows:
      cv.imshow(StereoCamera.LEFT_WINDOW_NAME, left)
      cv.imshow(StereoCamera.RIGHT_WINDOW_NAME, right)
    return left, right
  