from datetime import datetime
import math
from queue import Queue
import threading
import time
from typing import Any, Tuple
from icm20948 import ICM20948
from scipy.spatial.transform import Rotation
import numpy as np

from vslam.state import Delta, State
from vslam.utils import X_AXIS, Y_AXIS, Z_AXIS, angle_axis, angle_between, normalize, normalize_basis, rotate_to

class IMUSensor:
  """
  IMU sensor reader.
  """

  SAMPLES = 30

  def __init__(self) -> 'IMUSensor':
    self.imu = ICM20948()
    self.stopped = False
    self.first = False
    self.gyro_bias = np.array([0.0, 0.0, 0.0])
    self.accel_bias = np.array([0.0, 0.0, 0.0])
    self.accel = []
    self.gyro = []
    self.delta = []
    self.state = State()
    self.mutex = threading.Lock()
    self.thread = threading.Thread(target=self._runner)
    self.thread.start()
  
  def close(self):
    if not self.stopped:
      self.stopped = True

  def _runner(self):
    last_time = datetime.now()
    current_time = datetime.now()
    while not self.stopped:
      ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()
      current_time = datetime.now()
      self.mutex.acquire()
      self.accel.append(-np.asarray([ax, ay, az]) - self.accel_bias)
      # Negate for CCW rotations
      self.gyro.append(np.radians(np.asarray([gx, gy, gz])) - self.gyro_bias)
      self.delta.append((current_time - last_time).total_seconds())
      self.mutex.release()
      last_time = current_time
      self.first = True
      time.sleep(1.0 / 15.0)
  
  def read(self) -> Tuple[Any, Any, Any]:
    self.mutex.acquire()
    accel = self.accel
    self.accel = []
    gyro = self.gyro
    self.gyro = []
    delta = self.delta
    self.delta = []
    self.mutex.release()
    return accel, gyro, delta
  
  def calibrate(self) -> State:
    SAMPLES = 45
    a = []
    g = []
    while len(a) < SAMPLES:
      new_a, new_g, _ = self.read()
      a.extend(new_a.copy())
      g.extend(new_g.copy())
      time.sleep(0.5)
    mean_a = np.mean(a, axis=0)
    self.mutex.acquire()
    self.gyro_bias = np.mean(g, axis=0)
    self.gyro.clear()
    self.accel_bias = mean_a - normalize(mean_a)
    self.accel.clear()
    self.delta.clear()
    self.mutex.release()
    a = np.add(a, -self.accel_bias)
    world_y = normalize(-np.mean(a, axis=0))
    # Rotate world_y to y axis
    rotation = rotate_to(world_y, Y_AXIS)
    forward = np.dot(rotation, Z_AXIS)
    up = np.dot(rotation, Y_AXIS)
    forward, up, right = normalize_basis(forward, up)
    print("Forward: {}, Up: {}".format(forward, up))
    return State(forward=forward, up=up, right=right)
  
  def step(self, state: State) -> State:
    a, g, dt = self.read()
    translation = np.array([0.0, 0.0, 0.0])
    accum_rotation = rotate_to(Y_AXIS, state.up)
    forward = state.forward.copy()
    up = state.up.copy()
    right = np.cross(forward, up)
    for i in range(len(g)):
      dt2 = dt[i] * dt[i]
      # Compute rotated acceleration and add y axis unit vector to adjust for gravity
      ra = np.dot(accum_rotation, a[i])
      fa = ra + Y_AXIS
      translation += 0.5 * fa[0] * dt2 * right + 0.5 * fa[1] * dt2 * up + 0.5 * fa[2] * dt2 * forward
      rotation = np.linalg.multi_dot([
        angle_axis(right, g[i][0] * dt[i]),
        angle_axis(up, g[i][1] * dt[i]),
        angle_axis(forward, g[i][2] * dt[i])
      ])
      forward, up, right = normalize_basis(
        np.dot(rotation, forward),
        np.dot(rotation, up)
      )
      accum_rotation = np.dot(rotation, accum_rotation)
    return Delta(translation, forward - state.forward, up - state.up)
