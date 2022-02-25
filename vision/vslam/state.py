from enum import Enum
import math
from typing import Any, List, Tuple
from dataclasses import dataclass
import numpy as np

from vslam.database import VisualMeasurement
from vslam.sensors import SensorMeasurement

Color = Tuple[int, int, int]
Position = Tuple[float, float, float]
Vector = Tuple[float, float, float]

@dataclass
class Delta:
  delta_position: Vector = np.asarray((0.0, 0.0, 0.0))
  delta_orientation: Vector = np.asarray((0.0, 0.0, 0.0))
  
  def negate(self) -> 'Delta':
    return Delta(
      -self.delta_position,
      -self.delta_orientation
    )
  
@dataclass
class State:
  position: Position = np.asarray((0.0, 0.0, 0.0))
  forward: Vector = np.asarray((0.0, 0.0, 1.0))
  up: Vector = np.asarray((0.0, 1.0, 0.0))
  position_deviation: float = 0.0
  orientation_deviation: float = 0.0
  
  def apply_delta(self, delta: Delta) -> 'State':
    State(
      position=self.position + delta.delta_position,
      forward=self.forward + delta.delta_orientation,
      up=self.up + delta.delta_orientation
    )

class ControlAction(Enum):
  NONE = 0

@dataclass
class ControlState:
  action: ControlAction = ControlAction.NONE
  value: Any = None

@dataclass
class Measurement:
  visual_measurements: List[VisualMeasurement] = []
  sensor_measurement: SensorMeasurement = Delta()
