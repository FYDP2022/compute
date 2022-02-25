import math
from typing import Tuple
import numpy as np

from vslam.config import CONFIG
from vslam.parameters import CameraParameters

Color = Tuple[int, int, int]
Position = Tuple[float, float, float]
Vector = Tuple[float, float, float]

def angle_axis(axis, theta):
  """
  Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  """
  axis = np.asarray(axis)
  axis = axis / math.sqrt(np.dot(axis, axis))
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([
    [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
  ])

def normalize(vector: np.array) -> np.array:
  norm = np.linalg.norm(vector)
  if norm == 0.0:
    return vector
  return vector / norm 

def angle_between(v1, v2):
  v1_u = normalize(v1)
  v2_u = normalize(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def projection(x: np.array, y: np.array) -> np.array:
  """
  Projection of x onto y.
  """
  return np.dot(x, y) / np.linalg.norm(y)

def pixel_ray(direction: Vector, i: float, j: float) -> np.array:
  """
  Returns the ray vector given by the direction projected from the camera
  corresponding to the given image coords
  """
  xang = -((i / CONFIG.width) * CameraParameters.FOVX - CameraParameters.FOVX / 2.0)
  yang = -((j / CONFIG.height) * CameraParameters.FOVY - CameraParameters.FOVY / 2.0)
  rotation = np.dot(angle_axis(direction, math.radians(xang)), angle_axis(direction, math.radians(yang)))
  return normalize(np.dot(rotation, direction))