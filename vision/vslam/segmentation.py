from enum import IntEnum, unique
from typing import Any
import numpy as np

from vslam.config import CONFIG

@unique
class Material(IntEnum):
  ASPHALT = 0
  GRAVEL = 1
  SOIL = 2
  SAND = 3
  BUSH = 4
  FOREST = 5
  LOW_GRASS = 6
  HIGH_GRASS = 7
  MISC_VEGETATION = 8
  TREE_CROWN = 9
  TREE_TRUNK = 10
  BUILDING = 11
  FENCE = 12
  WALL = 13
  CAR = 14
  BUS = 15
  SKY = 16
  MISC_OBJECT = 17
  POLE = 18
  TRAFFIC_SIGN = 19
  PERSON = 20
  ANIMAL = 21
  EGO_VEHICLE = 22
  UNDEFINED = 255

class SemanticSegmentationModel:
  def __init__(self) -> 'SemanticSegmentationModel':
    pass

  def segment(self, image) -> Any:
    return np.full((CONFIG.width, CONFIG.height), Material.UNDEFINED.value)
