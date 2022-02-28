import math
from typing import List, Tuple
import numpy as np
import scipy

from vslam.state import Delta
from vslam.utils import angle_axis, find_orthogonal_axes, normalize, spherical_angles, spherical_rotation_matrix
from vslam.state import ControlState, Delta, State
from vslam.database import Feature, Observe, ProcessedFeatures, feature_database

class SLAM:
  """
  Simultaneous mapping and localization module.
  """

  def __init__(self) -> 'SLAM':
    pass

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float]:
    raise NotImplementedError()

class GradientAscentSLAM(SLAM):
  """
  Simultaneous mapping and localization using stochastic gradient ascent.
  """

  BASIS = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ])

  def __init__(self) -> 'SLAM':
    super().__init__()

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float]:
    RESOLUTION_POSITION = 0.01 # 1cm
    RESOLUTION_ANGLE = math.radians(1) # 1 degree
    MAX_ITERATIONS = 5
    SAMPLES = 30
    DECAY = float(MAX_ITERATIONS - 1) / float(MAX_ITERATIONS)
    PROBABILITY_THRESHOLD = 0.95
    LR = 0.01

    # Apply Gradient Ascent on visual measurement probability by computing derivatives
    # using the fundamental theorem of calculus
    delta = Delta()
    _, last_probability = feature_database.observe(estimate, np.random.choice(frame, SAMPLES))
    print(last_probability)
    if last_probability == 0.0:
      return delta, 0.0
    lr = LR
    for _ in range(MAX_ITERATIONS):
      if last_probability > PROBABILITY_THRESHOLD:
        break
      position_gradient = []
      for axis in GradientAscentSLAM.BASIS:
        positive = estimate.apply_delta(Delta(delta.delta_position + axis * RESOLUTION_POSITION, delta.delta_theta, delta.delta_phi))
        positive.position_deviation += RESOLUTION_POSITION
        _, p_upper = feature_database.observe(positive, np.random.choice(frame, SAMPLES))
        p_lower = last_probability
        position_gradient.append((p_upper - p_lower) / RESOLUTION_POSITION)
      orientation_gradient = []
      rotation_axes = find_orthogonal_axes(estimate.apply_delta(delta).forward)
      for axis in rotation_axes:
        positive = estimate.apply_delta(delta)
        positive.rotate(RESOLUTION_ANGLE, axis)
        _, p_upper = feature_database.observe(positive, np.random.choice(frame, SAMPLES))
        p_lower = last_probability
        orientation_gradient.append((p_upper - p_lower) / RESOLUTION_ANGLE)
      next = Delta()
      next.delta_position += LR * np.asarray(position_gradient)
      orientation = normalize(np.dot(spherical_rotation_matrix(delta.delta_theta, delta.delta_phi), estimate.forward))
      orientation = np.dot(angle_axis(rotation_axes[0], LR * orientation_gradient[0]), orientation)
      orientation = np.dot(angle_axis(rotation_axes[1], LR * orientation_gradient[1]), orientation)
      t1, p1 = spherical_angles(estimate.forward)
      t2, p2 = spherical_angles(orientation)
      next.delta_theta = t2 - t1
      next.delta_phi = p2 - p1
      _, probability = feature_database.observe(estimate.apply_delta(next), np.random.choice(frame, SAMPLES))
      # if probability <= last_probability:
      #   break
      lr *= DECAY
      last_probability = probability
      delta = next
      print(probability)
    return delta, last_probability

class SoftmaxSLAM(SLAM):
  """
  Simultaneous mapping and localization using one pass correlation weighted
  by the softmax distribution of.
  """

  def __init__(self) -> 'SoftmaxSLAM':
    super().__init__()

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float]:
    measurements, pr = feature_database.observe(estimate, frame, Observe.VISUAL_MEASUREMENT)
    if len(measurements) == 0:
      return Delta(), 0.0
    # measurements.sort(key=lambda x: x.importance_weight)
    softmax = scipy.special.softmax(list(map(lambda x: x.importance_weight, measurements)))
    delta = Delta()
    for i, measurement in enumerate(measurements):
      delta.delta_position += softmax[i] * measurement.delta.delta_position
      delta.delta_theta += softmax[i] * measurement.delta.delta_theta
      delta.delta_phi += softmax[i] * measurement.delta.delta_phi
    return delta, pr