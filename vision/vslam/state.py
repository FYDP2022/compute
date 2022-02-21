import math
from typing import Tuple

Color = Tuple[int, int, int]
Position = Tuple[float, float, float]
Vector = Tuple[float, float, float]
Orientation = Tuple[float, float, float]

class Delta:
  def __init__(self, delta_position: Vector = (0.0, 0.0, 0.0), delta_orientation: Vector = (0.0, 0.0, 0.0)) -> 'Delta':
    self.delta_position = delta_position
    self.delta_orientation = delta_orientation

class State:
  THRESHOLD = 1.0

  def __init__(self) -> 'State':
    self.position: Position = (0.0, 0.0, 0.0)
    self.orientation: Orientation = (0.0, 0.0, 0.0)
    self.variance = math.inf
