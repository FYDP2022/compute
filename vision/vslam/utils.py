import math
from typing import Any, Tuple
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
  # TODO: is this CCW??
  axis = axis / math.sqrt(dot)
  a = math.cos(theta / 2.0)
  b, c, d = axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([
    [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
  ])

def rotate_to(start, end):
  """
  Returns rotation matrix which rotates `start` onto `end`
  """
  v = np.cross(start, end)
  sine = np.linalg.norm(v)
  cosine = np.dot(start, end)
  skew_symmetric = np.array([
    [0.0, -v[2], v[1]],
    [v[2], 0.0, -v[0]],
    [-v[1], v[0], 0.0]
  ])
  return np.identity(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * (1 - cosine) / (sine ** 2)

def normalize(vector: np.array) -> np.array:
  norm = np.linalg.norm(vector)
  if norm == 0.0:
    return vector
  return vector / norm 

def angle_between(v1, v2):
  v1_u = normalize(v1)
  v2_u = normalize(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_about(v1, v2, n):
  """
  Angle between accounting for sign assuming rotation axis `n`
  """
  v1_u = normalize(v1)
  v2_u = normalize(v2)
  angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  cross = np.cross(v1, v2)
  if np.dot(cross, n) > 0.0:
    # CW rotation needed
    return -angle
  return angle

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

# def look_at_matrix(forward, up):
#   right = np.cross(forward, up)
#   return np.array([
#     right,
#     up,
#     forward
#   ])

def rotate_to_worldcoords(forward, up):
  right = np.cross(forward, up)
  return np.array([
    [up[0], right[0], forward[0]],
    [up[1], right[1], forward[1]],
    [up[2], right[2], forward[2]]
  ])

def normalize_basis(forward, up) -> Tuple[Any, Any, Any]:
  f = normalize(forward)
  r = normalize(np.cross(f, up))
  u = normalize(np.cross(r, f))
  return f, u, r