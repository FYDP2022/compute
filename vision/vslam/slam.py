import math
from statistics import variance
from typing import List, Tuple
import numpy as np
import scipy

from vslam.state import Delta, Deviation
from vslam.utils import angle_axis, angle_between, find_orthogonal_axes, normalize, random_basis, spherical_angles, spherical_rotation_matrix
from vslam.state import ControlState, Delta, State
from vslam.database import Feature, Observe, ProcessedFeatures, feature_database

class SLAM:
  """
  Simultaneous mapping and localization module.
  """

  def __init__(self) -> 'SLAM':
    pass

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float, Deviation]:
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

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float, Deviation]:
    RESOLUTION_POSITION = 0.05 # 5cm
    RESOLUTION_ANGLE = math.radians(5) # 5degree
    MAX_ITERATIONS = 5
    SAMPLES = 1000
    DECAY = float(MAX_ITERATIONS - 1) / float(MAX_ITERATIONS)
    PROBABILITY_THRESHOLD = 0.95
    LR = 0.01

    samples = min(SAMPLES, len(frame))

    # Apply Gradient Ascent on visual measurement probability by computing derivatives
    # using the fundamental theorem of calculus
    delta = Delta()
    _, last_probability = feature_database.observe(estimate, np.random.choice(frame, samples, False))
    print(last_probability)
    if last_probability == 0.0:
      return delta, 0.0, Deviation()
    lr = LR
    for _ in range(MAX_ITERATIONS):
      if last_probability > PROBABILITY_THRESHOLD:
        break
      position_gradient = []
      translation_axis = random_basis()
      for axis in translation_axis:
        positive = estimate.apply_delta(Delta(delta.delta_position + axis * RESOLUTION_POSITION, delta.delta_orientation))
        positive.position_deviation += RESOLUTION_POSITION
        negative = estimate.apply_delta(Delta(delta.delta_position - axis * RESOLUTION_POSITION, delta.delta_orientation))
        negative.position_deviation += RESOLUTION_POSITION
        _, p_upper = feature_database.observe(positive, np.random.choice(frame, samples, False))
        _, p_lower = feature_database.observe(negative, np.random.choice(frame, samples, False))
        position_gradient.append((p_upper - p_lower) / (2 * RESOLUTION_POSITION))
      orientation_gradient = []
      rotation_axes = find_orthogonal_axes(estimate.apply_delta(delta).forward)
      for axis in rotation_axes:
        positive = estimate.apply_delta(delta)
        positive.rotate(RESOLUTION_ANGLE, axis)
        negative = estimate.apply_delta(delta)
        negative.rotate(-RESOLUTION_ANGLE, axis)
        _, p_upper = feature_database.observe(positive, np.random.choice(frame, samples, False))
        _, p_lower = feature_database.observe(negative, np.random.choice(frame, samples, False))
        orientation_gradient.append((p_upper - p_lower) / (2 * RESOLUTION_ANGLE))
      next = Delta()
      next.delta_position += lr * (position_gradient[0] * translation_axis[0])
      next.delta_position += lr * (position_gradient[1] * translation_axis[1])
      next.delta_position += lr * (position_gradient[2] * translation_axis[2])
      orientation = estimate.forward + delta.delta_orientation
      orientation = np.dot(angle_axis(rotation_axes[0], LR * orientation_gradient[0]), orientation)
      orientation = np.dot(angle_axis(rotation_axes[1], LR * orientation_gradient[1]), orientation)
      next.delta_orientation = orientation - estimate.forward
      _, probability = feature_database.observe(estimate.apply_delta(next), np.random.choice(frame, samples, False))
      last_probability = probability
      delta = next
      lr *= DECAY
      print(last_probability)
    return delta, last_probability, Deviation()

class SoftmaxSLAM(SLAM):
  """
  Simultaneous mapping and localization using one pass correlation weighted
  by the softmax distribution of.
  """

  def __init__(self) -> 'SoftmaxSLAM':
    super().__init__()

  def step(self, estimate: State, action: ControlState, frame: List[Feature]) -> Tuple[Delta, float, Deviation]:
    measurements, pr = feature_database.observe(estimate, frame, Observe.VISUAL_MEASUREMENT)
    if len(measurements) == 0:
      return Delta(), 0.0, Deviation()
    # measurements.sort(key=lambda x: x.importance_weight)
    softmax = scipy.special.softmax(list(map(lambda x: x.importance_weight, measurements)))
    delta = Delta()
    delta_angle = 0.0
    squared_position = np.asarray([0.0, 0.0, 0.0])
    squared_angle = 0.0
    for i, measurement in enumerate(measurements):
      delta.delta_position += softmax[i] * measurement.delta.delta_position
      delta.delta_orientation += softmax[i] * measurement.delta.delta_orientation
      angle = angle_between(estimate.forward, estimate.forward + measurement.delta.delta_orientation)
      delta_angle += softmax[i] * angle
      squared_position += softmax[i] * np.power(measurement.delta.delta_position, 2)
      squared_angle += softmax[i] * (angle ** 2)
    deviation = Deviation(
      np.max(np.sqrt(squared_position + np.power(delta.delta_position, 2))),
      np.sqrt(squared_angle + np.power(delta_angle, 2))
    )
    delta.delta_orientation = normalize(delta.delta_orientation)
    return delta, pr, deviation