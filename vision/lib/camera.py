from enum import Enum
import queue
import threading
from typing import Any, Tuple
import cv2 as cv

class CameraIndex(Enum):
  LEFT = 0
  RIGHT = 1

class StereoCamera:
  """Streams data from binocular camera sensor"""

  WIDTH = 3280
  HEIGHT = 2464

  def __init__(self, width: int, height: int) -> 'StereoCamera':
    self.width = width
    self.height = height
    self.reverse = False
    self.queue = (queue.Queue(), queue.Queue())
    self.capture = (
      cv.VideoCapture(self._cameraString(CameraIndex.LEFT)),
      cv.VideoCapture(self._cameraString(CameraIndex.RIGHT))
    )
    if not self.capture[CameraIndex.LEFT].isOpened():
      raise RuntimeError('failed to capture data from left camera')
    if not self.capture[CameraIndex.RIGHT].isOpened():
      raise RuntimeError('failed to capture data from right camera')
    t1 = threading.Thread(target=self._reader, args=CameraIndex.LEFT)
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=self._reader, args=CameraIndex.RIGHT)
    t2.daemon = True
    t2.start()

  def _reader(self, camera: CameraIndex):
    while True:
      ret, frame = self.capture[camera].read()
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
    '''
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
    '''.format(camera, StereoCamera.WIDTH, StereoCamera.HEIGHT, self.width, self.height)

  def swapCameras(self):
    self.reverse = not self.reverse
  
  def read(self) -> Tuple[Any, Any]:
    if self.reverse:
      return (self.queue[CameraIndex.RIGHT].get(), self.queue[CameraIndex.LEFT].get())
    else:
      return (self.queue[CameraIndex.LEFT].get(), self.queue[CameraIndex.RIGHT].get())
  