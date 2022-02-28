import math
from typing import Tuple
import numpy as np

from vslam.config import CONFIG
from vslam.parameters import CameraParameters

Color = Tuple[int, int, int]
Position = Tuple[float, float, float]
Vector = Tuple[float, float, float]

X_AXIS = np.asarray([1.0, 0.0, 0.0])
Y_AXIS = np.asarray([0.0, 1.0, 0.0])
Z_AXIS = np.asarray([0.0, 0.0, 1.0])

def angle_axis(axis, theta):
  """
  Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  """
  axis = np.asarray(axis)
  dot = np.dot(axis, axis)
  if theta == 0.0 or dot == 0.0:
    return np.identity(3)
  axis = axis / math.sqrt(dot)
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
  norm = np.linalg.norm(y)
  if norm == 0.0:
    return 0.0
  else:
    return np.dot(x, y) / norm

def pixel_ray(direction: Vector, i: float, j: float) -> np.array:
  """
  Returns the ray vector given by the direction projected from the camera
  corresponding to the given image coords
  """
  xang = -((i / float(CONFIG.width)) * CameraParameters.FOVX - CameraParameters.FOVX / 2.0)
  yang = -((j / float(CONFIG.height)) * CameraParameters.FOVY - CameraParameters.FOVY / 2.0)
  rotation = np.dot(angle_axis(X_AXIS, math.radians(xang)), angle_axis(Y_AXIS, math.radians(yang)))
  return normalize(np.dot(rotation, direction))

def find_orthogonal_axes(v: np.array) -> Tuple[np.array, np.array]:
  x = np.random.randn(3)
  while np.array_equal(x, v):
    x = np.random.randn(3)
  x -= x.dot(v) * v
  x /= np.linalg.norm(x)
  y = np.cross(v, x)
  return x, y

def random_basis() -> Tuple[np.array, np.array, np.array]:
  a1 = normalize(np.random.randn(3))
  a2, a3 = find_orthogonal_axes(a1)
  return a1, a2, a3

def spherical_angles(v: np.array) -> Tuple[float, float]:
  """Returns (theta, phi)."""
  # Z-angle
  phi = angle_between(v, Z_AXIS)
  # XY-angle
  theta = np.arctan2(v[1], v[0])
  return theta % (2 * math.pi), phi % (2 * math.pi)

def spherical_coordinates(theta: float, phi: float) -> np.array:
  return np.asarray([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

def spherical_rotation_matrix(theta: float, phi: float) -> np.ndarray:
  v = spherical_coordinates(theta, phi)
  z_rot_axis = np.cross(v, Z_AXIS)
  return np.dot(angle_axis(z_rot_axis, phi), angle_axis(Z_AXIS, theta))
