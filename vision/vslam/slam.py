import math
from typing import List, Tuple
import numpy as np
from particlefilter import ParticleFilterLoc
import copy

from vslam.camera import angle_axis
from vslam.state import ControlState, Delta, Measurement, State
from vslam.database import Feature, feature_database

def find_orthogonal_axes(v: np.array) -> Tuple[np.array, np.array]:
  x = np.random.randn(3)
  while x == v:
    x = np.random.randn(3)
  x -= x.dot(v) * v
  x /= np.linalg.norm(x)
  y = np.cross(v, x)
  return x, y

class SLAM:
  """
  Simultaneous mapping and localization module.
  """

  BASIS = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ])

  def __init__(self) -> 'SLAM':
    pass

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float]:
    RESOLUTION_POSITION = 0.01 # 1cm
    RESOLUTION_ANGLE = math.radians(1) # 1 degree
    MAX_ITERATIONS = 5
    DECAY = float(MAX_ITERATIONS - 1) / float(MAX_ITERATIONS)
    PROBABILITY_THRESHOLD = 0.95
    LR = 0.01

    # Apply Gradient Ascent on visual measurement probability by computing derivatives
    # using the fundamental theorem of calculus
    delta = Delta()
    last_probability = feature_database.observe(estimate, frame)
    lr = LR
    for _ in range(MAX_ITERATIONS):
      if last_probability > PROBABILITY_THRESHOLD:
        break
      position_gradient = []
      for axis in SLAM.BASIS:
        positive = Delta(delta.delta_position + axis * RESOLUTION_POSITION, delta.delta_orientation)
        _, p_upper = feature_database.observe(estimate.apply_delta(positive), frame)
        p_lower = last_probability
        position_gradient.append((p_upper - p_lower) / RESOLUTION_POSITION)
      orientation_gradient = []
      rotation_axes = find_orthogonal_axes(estimate.apply_delta(delta).forward)
      for axis in rotation_axes:
        positive = estimate.apply_delta(delta)
        positive.rotate(RESOLUTION_ANGLE, axis)
        _, p_upper = feature_database.observe(positive, frame)
        p_lower = last_probability
        orientation_gradient.append((p_upper - p_lower) / RESOLUTION_ANGLE)
      delta.delta_position += LR * np.asarray(position_gradient)
      orientation = estimate.forward + delta.delta_orientation
      orientation = np.dot(angle_axis(rotation_axes[0], LR * orientation_gradient[0]), orientation)
      orientation = np.dot(angle_axis(rotation_axes[1], LR * orientation_gradient[1]), orientation)
      delta.delta_orientation = orientation - estimate.forward
      _, probability = feature_database.observe(estimate.apply_delta(delta), frame)
      if probability <= last_probability:
        break
      lr *= DECAY
    return delta, last_probability