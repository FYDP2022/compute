from enum import IntEnum
import queue
import signal
import threading
from typing import Any, List, Tuple
import cv2 as cv
import time

from vslam.config import CONFIG, DebugWindows

class CameraIndex(IntEnum):
  LEFT = 0
  RIGHT = 1

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
    self.mutex = threading.Lock()
    self.ready = [0, 0]
    self.capture: List[cv.VideoCapture] = [None, None]
    self.capture[CameraIndex.LEFT] = cv.VideoCapture(self._cameraString(CameraIndex.LEFT))
    if not self.capture[CameraIndex.LEFT].isOpened():
      raise RuntimeError('failed to capture data from camera {}'.format(CameraIndex.LEFT))
    self.capture[CameraIndex.RIGHT] = cv.VideoCapture(self._cameraString(CameraIndex.RIGHT))
    if not self.capture[CameraIndex.RIGHT].isOpened():
      raise RuntimeError('failed to capture data from camera {}'.format(CameraIndex.RIGHT))
    self.t1 = threading.Thread(target=self._reader, args=[CameraIndex.LEFT])
    self.t1.daemon = True
    self.t1.start()
    self.t2 = threading.Thread(target=self._reader, args=[CameraIndex.RIGHT])
    self.t2.daemon = True
    self.t2.start()
    self.handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, self._signalHandler)
    time.sleep(2)
    if DebugWindows.CAMERA in CONFIG.windows:
      cv.namedWindow(StereoCamera.LEFT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(StereoCamera.LEFT_WINDOW_NAME, self.width, self.height)
      cv.namedWindow(StereoCamera.RIGHT_WINDOW_NAME, cv.WINDOW_NORMAL)
      cv.resizeWindow(StereoCamera.RIGHT_WINDOW_NAME, self.width, self.height)

  def close(self):
    if not self.stopped:
      print('STOPPED')
      self.stopped = True
      self.barrier.reset()
      self.capture[CameraIndex.LEFT].release()
      self.capture[CameraIndex.RIGHT].release()

  def _signalHandler(self, sig, frame):
    self.close()
    signal.signal(signal.SIGINT, self.handler)

  def _reader(self, camera: CameraIndex):
    try:
      while not self.stopped:
        self.barrier.wait()
        ret, frame = self.capture[camera].read()
        if self.stopped:
          break
        if not ret:
          raise RuntimeError('failed to read camera sensor: {}'.format(camera))
        self.mutex.acquire()
        if not self.queue[camera].empty():
          try:
            # discard previous (unprocessed) frame
            self.queue[camera].get_nowait()
          except queue.Empty:
            pass
        self.queue[camera].put(cv.flip(frame, -1))
        self.ready[camera] += 1
        self.mutex.release()
    except threading.BrokenBarrierError:
      pass

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
    """.format(camera.value, StereoCamera.WIDTH, StereoCamera.HEIGHT, self.width, self.height)

  def swapCameras(self):
    self.reverse = not self.reverse

  def _getFrame(self):
    while True:
      self.mutex.acquire()
      if self.stopped:
        raise RuntimeError('Tried to read closed camera')
      if self.ready[CameraIndex.LEFT] == self.ready[CameraIndex.RIGHT] and not self.queue[CameraIndex.LEFT].empty():
        break
      self.mutex.release()
      time.sleep(30 / 1000.0)
    result = (self.queue[CameraIndex.LEFT].get_nowait(), self.queue[CameraIndex.RIGHT].get_nowait())
    if self.reverse:
      result = (result[1], result[0])
    self.mutex.release()
    return result
  
  def read(self) -> Tuple[Any, Any]:
    left, right = self._getFrame()
    if DebugWindows.CAMERA in CONFIG.windows:
      cv.imshow(StereoCamera.LEFT_WINDOW_NAME, left)
      cv.imshow(StereoCamera.RIGHT_WINDOW_NAME, right)
    return left, right
  