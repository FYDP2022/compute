from enum import IntEnum
import queue
import signal
import sys
import threading
from typing import Any, List, Tuple
import cv2 as cv

class CameraIndex(IntEnum):
  RIGHT = 0
  LEFT = 1

class StereoCamera:
  """Streams data from binocular camera sensor"""

  WIDTH = 1920
  HEIGHT = 1080

  def __init__(self, width: int, height: int) -> 'StereoCamera':
    self.width = width
    self.height = height
    self.reverse = False
    self.stopped = False
    self.queue = (queue.Queue(), queue.Queue())
    self.capture: List[cv.VideoCapture] = [None, None]
    self.capture[CameraIndex.LEFT] = cv.VideoCapture(self._cameraString(CameraIndex.LEFT))
    self.capture[CameraIndex.RIGHT] = cv.VideoCapture(self._cameraString(CameraIndex.RIGHT))
    signal.signal(signal.SIGINT, self._signalHandler)
    if not self.capture[CameraIndex.LEFT].isOpened():
      raise RuntimeError('failed to capture data from left camera')
    if not self.capture[CameraIndex.RIGHT].isOpened():
      raise RuntimeError('failed to capture data from right camera')
    self.t1 = threading.Thread(target=self._reader, args=[CameraIndex.LEFT])
    self.t1.daemon = True
    self.t1.start()
    self.t2 = threading.Thread(target=self._reader, args=[CameraIndex.RIGHT])
    self.t2.daemon = True
    self.t2.start()

  def _signalHandler(self, sig, frame):
    self.stopped = True
    self.t1.join()
    self.t2.join()
    self.capture[CameraIndex.LEFT].release()
    self.capture[CameraIndex.RIGHT].release()
    sys.exit(0)

  def _reader(self, camera: CameraIndex):
    while not self.stopped:
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
  
  def read(self) -> Tuple[Any, Any]:
    if self.reverse:
      return (self.queue[CameraIndex.RIGHT].get(), self.queue[CameraIndex.LEFT].get())
    else:
      return (self.queue[CameraIndex.LEFT].get(), self.queue[CameraIndex.RIGHT].get())
  