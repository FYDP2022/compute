import math
from typing import Tuple
from dataclasses import dataclass

Color = Tuple[int, int, int]
Position = Tuple[float, float, float]
Vector = Tuple[float, float, float]

class Delta:
  def __init__(self, delta_position: Vector = (0.0, 0.0, 0.0), delta_orientation: Vector = (0.0, 0.0, 0.0)) -> 'Delta':
    self.delta_position = delta_position
    self.delta_orientation = delta_orientation
  
  def negate(self) -> 'Delta':
    return Delta(
      -self.delta_position,
      -self.delta_orientation
    )

@dataclass
class State:
  position: Position = (0.0, 0.0, 0.0)
  forward: Vector = (0.0, 0.0, 1.0)
  up: Vector = (0.0, 1.0, 0.0)
  variance = math.inf    
  
  def apply_delta(self, delta: Delta) -> 'State':
    State(
      position=self.position + delta.delta_position,
      forward=self.forward + delta.delta_orientation,
      up=self.up + delta.delta_orientation
    )
