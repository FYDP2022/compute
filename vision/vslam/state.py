from enum import Enum
import math
from typing import Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from vslam.utils import Z_AXIS, Position, Vector, angle_axis, angle_between, normalize, normalize_basis, spherical_rotation_matrix

@dataclass
class Delta:
  delta_position: Vector = field(default_factory=lambda: np.asarray([0.0, 0.0, 0.0]))
  delta_forward: Vector = field(default_factory=lambda: np.asarray([0.0, 0.0, 0.0]))
  delta_up: Vector = field(default_factory=lambda: np.asarray([0.0, 0.0, 0.0]))
  
  def negate(self) -> 'Delta':
    return Delta(
      -self.delta_position,
      -self.delta_forward,
      -self.delta_up
    )

@dataclass
class Deviation:
  position_deviation: float = 0.0
  forward_deviation: float = 0.0
  up_deviation: float = 0.0
  
@dataclass
class State:
  position: Position = field(default_factory=lambda: np.asarray((0.0, 0.0, 0.0)))
  forward: Vector = field(default_factory=lambda: np.asarray((0.0, 0.0, 1.0)))
  up: Vector = field(default_factory=lambda: np.asarray((0.0, 1.0, 0.0)))
  right: Vector = field(default_factory=lambda: np.asarray((0.0, 0.0, -1.0)))
  position_deviation: float = 0.0
  orientation_deviation: float = 0.0
  
  def apply_delta(self, delta: Delta) -> 'State':
    f, u, r = normalize_basis(self.forward + delta.delta_forward, self.up + delta.delta_up)
    return State(
      position=self.position + delta.delta_position,
      forward=f,
      up=u,
      right=r,
      position_deviation=self.position_deviation,
      orientation_deviation=self.orientation_deviation
    )
  
  def apply_deviation(self, deviation: Deviation) -> 'State':
    self.position_deviation += deviation.position_deviation
    self.orientation_deviation += deviation.orientation_deviation
    return self
  
  def rotate(self, angle: float, axis: np.array):
    mat = angle_axis(axis, angle)
    self.forward = np.dot(mat, self.forward)
    self.up = np.dot(mat, self.up)

class ControlAction(Enum):
  NONE = 0

@dataclass
class ControlState:
  action: ControlAction = ControlAction.NONE
  value: Any = None

# @dataclass
# class Measurement:
#   visual_measurements: List[VisualMeasurement] = []
#   sensor_measurement: SensorMeasurement = Delta()
