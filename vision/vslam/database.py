from cmath import isinf
from copy import deepcopy
from enum import Enum, unique
import json
import math
import random
import os
from rtree import index
import sqlite3
import statistics
import scipy.stats as st
from typing import Any, Generator, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from vslam.parameters import CameraParameters
from vslam.utils import X_AXIS, Y_AXIS, Z_AXIS, Color, Position, Vector, angle_axis, angle_between, normalize, pixel_ray, projection, spherical_angles, spherical_coordinates, spherical_rotation_matrix
from vslam.config import CONFIG
from vslam.segmentation import Material
from vslam.state import Delta, State

# NOTE: bounding boxes are in the form (x1, x2, y1, y2, z1, z2)
BoundingBox = Tuple[float, float, float, float, float, float]

@dataclass
class VisualMeasurement:
  feature: 'Feature'
  delta: Delta
  # Pr {yt - yt-1 = xt - xt-1 | xt-1, yt, yt-1}
  importance_weight: float

class MetadataDatabase:
  def __init__(self) -> 'MetadataDatabase':
    pass

@dataclass
class Feature:
  id: int = 0
  n: int = 1
  age: int = 0
  color: Color = field(default_factory=lambda: np.asarray((0, 0, 0)))
  position_mean: Position = field(default_factory=lambda: np.asarray((0.0, 0.0, 0.0)))
  position_deviation: Vector = field(default_factory=lambda: np.asarray((math.inf, math.inf, math.inf)))
  orientation_mean: Vector = field(default_factory=lambda: np.asarray((0.0, 0.0, 1.0)))
  orientation_deviation: float = math.inf
  radius_mean: float = 0.0
  radius_deviation: float = 0.0
  material: Material = Material.UNDEFINED

  def create(kp, image, points3d, disparity) -> Optional['Feature']:
    THRESHOLD = 5
    SIZE_THRESHOLD = 4.0 # 4pixel diameter
    RADIUS_THRESHOLD = 0.05 # 5cm
    x, y = round(kp.pt[0]), round(kp.pt[1])
    window_size = math.ceil(kp.size / 2.0)
    if kp.size < SIZE_THRESHOLD:
      return None
    color = np.asarray((0, 0, 0))
    n = 0.0
    depth = []
    for i in range(x - window_size, x + window_size):
      for j in range(y - window_size, y + window_size):
        if i < CONFIG.width and j < CONFIG.height and math.sqrt((i - x) ** 2 + (j - y) ** 2) <= window_size:
          n += 1.0
          color = color * (n - 1) / n + np.asarray(image[j, i]) / n
          if disparity[j, i] > THRESHOLD and not isinf(disparity[j, i]):
            depth.append(float(points3d[j, i][2] / 1000.0)) # Q matrix was computed using mm

    if len(depth) < 2:
      return None

    median_depth = statistics.median(depth)
    radius = math.tan(math.radians(CameraParameters.FOVX) * (kp.size / 2.0) / CONFIG.width) * median_depth
    if radius < RADIUS_THRESHOLD:
      return None
    v = pixel_ray(Z_AXIS, kp.pt[0], kp.pt[1])
    deviation = statistics.stdev(depth) + radius
    
    return Feature(
      color=np.asarray((math.floor(color[0]), math.floor(color[1]), math.floor(color[2]))),
      position_mean=v * median_depth,
      position_deviation=np.abs(deviation * v) + (radius / 4) * np.asarray([1.0, 1.0, 1.0]), # Maybe basis should be exact here
      orientation_mean=math.cos(kp.angle) * X_AXIS + math.sin(kp.angle) * Y_AXIS,
      orientation_deviation=math.pi / 2,
      radius_mean=radius,
      radius_deviation=kp.response
    )
  
  def incremental_mean(self, mean, sample, n: int) -> Any:
    """
    Merge two means incrementally (self has n - 1 samples other has 1 sample).
    """
    return mean * float(n - 1) / n + sample / float(n)
  
  def expectation(self, last, sample, pr: float, n: int) -> Any:
    return ((n - pr) * last + pr * sample) / n

  def incremental_deviation(self, deviation, mean, sample, n: int) -> Any:
    """
    Merge two deviations incrementally (self has n - 1 samples other has 1 sample).
    """
    return (float(n - 2) / (n - 1)) * deviation + np.power(sample - mean, 2) / float(n)

  def deviation(self, other: 'Feature') -> float:
    """
    Compute position, radius and orientation deviations between two features.
    """
    delta_p = other.position_mean - self.position_mean
    # Unit vector from feature 1 -> 2
    r = normalize(delta_p)

    # Projection of deviation vector onto r for both features
    position_deviation = abs(projection(other.position_deviation, r)) + abs(projection(self.position_deviation, r))
    return position_deviation
  
  def similarity(self, other: 'Feature', estimate: State) -> float:
    delta_p = other.position_mean - self.position_mean
    dist = np.linalg.norm(delta_p)
    r = normalize(delta_p)
    position_range = abs(projection(self.position_deviation, r)) + abs(projection(other.position_deviation, r)) + estimate.position_deviation
    radius_range = self.radius_mean + self.radius_deviation + other.radius_mean + other.radius_deviation
    orientation_range = self.orientation_deviation + other.orientation_deviation + estimate.orientation_deviation
    overlap = 1.0 - min(dist / (position_range + radius_range), 1.0)
    angle = angle_between(self.orientation_mean, other.orientation_mean)
    alignment = 1.0 - min(angle / orientation_range, 1.0)
    return overlap * alignment * (self.material == other.material)
  
  def probability(self, other: 'Feature', estimate: State) -> float:
    """
    Computes the probability P(yt is z | x, z),
    P(yt has size within sigma(z) of z | z) & P(yt has orientation within sigma(x) of z | z)
    where
    * `other` (yt) is the feature representing the current visual measurement
    * `self` (z) is a previously recognized feature
    * `estimate` (x) is the current state estimate
    """
    # NOTE: we are assuming that position, radius & orientation are independent
    # Compute deviations
    dp = self.deviation(other)
    delta_p = other.position_mean - self.position_mean
    r = np.linalg.norm(delta_p)
    position_probability = 1.0 if r == 0.0 else 0.0
    if dp > 0.0:
      # Compute the sigma value (num deviations from mean) of difference between features (based
      # on distributions projected onto the line subtending features)
      # Marginal probability along radial vector delta_p
      radial_deviation = (1 + self.radius_deviation) * self.radius_mean + (1 + other.radius_deviation) * other.radius_mean
      boundary = radial_deviation
      # Pr(r <= self.radius_mean) written in standard normal distribution
      upper_p = (boundary - r) / dp
      lower_p = (-boundary - r) / dp
      # Computes P(r is within boundaries of z) using CDF of standard normal distribution
      position_probability = st.norm.cdf(upper_p) - st.norm.cdf(lower_p)
    delta_r = abs(other.radius_mean - self.radius_mean)
    radius_probability = 1.0 if delta_r == 0.0 else 0.0
    if delta_r > 0.0:
      upper_r = delta_r / self.radius_deviation + 1
      lower_r = delta_r / self.radius_deviation - 1
      # Computes P(radius of yt is within 1 stddev of z) using CDF of standard normal distribution
      radius_probability = (st.norm.cdf(upper_r) - st.norm.cdf(lower_r)) / 0.68
    angle = angle_between(self.orientation_mean, other.orientation_mean)
    do = self.orientation_deviation + other.orientation_deviation
    upper_o = angle / do + 1
    lower_o = angle / do - 1
    orientation_probability = (st.norm.cdf(upper_o) - st.norm.cdf(lower_o)) / 0.68
    # print("B: {}, {}, {}".format(position_probability, radius_probability, orientation_probability))
    return position_probability * radius_probability * orientation_probability
  
  def angles(self) -> Tuple[float, float]:
    return spherical_angles(self.orientation_mean)
  
  def merge(self, other: 'Feature', probability: float) -> 'Feature':
    assert other.n == 1
    n = self.n + 1
    color = self.incremental_mean(self.color, other.color, n)
    return Feature(
      id=self.id,
      color=np.asarray((math.floor(color[0]), math.floor(color[1]), math.floor(color[2]))),
      n=n,
      position_mean=self.expectation(self.position_mean, other.position_mean, probability, n),
      position_deviation=self.incremental_deviation(self.position_deviation, self.position_mean, other.position_deviation, n),
      orientation_mean=self.incremental_mean(self.orientation_mean, other.orientation_mean, n),
      orientation_deviation=normalize(self.incremental_deviation(self.orientation_deviation, 0.0, other.orientation_deviation, n)),
      radius_mean=self.incremental_mean(self.radius_mean, other.radius_mean, n),
      radius_deviation=self.incremental_deviation(self.radius_deviation, self.radius_mean, other.radius_deviation, n),
      material=self.material
    )

  def measurement(self, other: 'Feature', state: State) -> VisualMeasurement:
    return VisualMeasurement(
      self.id,
      other.position_mean - state.position
    )
  
  def apply_basis(self, basis: State) -> 'Feature':
    # Apply affine transformation to feature from frame F(x, y, z, O) to
    # F(x, y, z, p) and apply rotation in world coordinates
    clone = deepcopy(self)
    angle = angle_between(basis.forward, Z_AXIS)
    axis = np.cross(basis.forward, Z_AXIS)
    rotation = angle_axis(axis, angle)
    clone.position_mean = np.dot(rotation, self.position_mean) + basis.position
    clone.orientation_mean = normalize(np.dot(rotation, clone.orientation_mean))
    p2 = np.dot(rotation, clone.position_deviation + self.position_mean)
    p1 = np.dot(rotation, self.position_mean)
    clone.position_deviation = p2 - p1
    return clone
  
  def delta(self, other: 'Feature') -> Delta:
    return Delta(
      other.position_mean - self.position_mean,
      other.orientation_mean - self.orientation_mean
    )
  
  def bbox(self, deviation: float) -> BoundingBox:
    radius = self.radius_mean * (1 + self.radius_deviation) + deviation
    return (
      self.position_mean[0] - radius, self.position_mean[0] + radius,
      self.position_mean[1] - radius, self.position_mean[1] + radius,
      self.position_mean[2] - radius, self.position_mean[2] + radius
    )
  
  def value(self) -> str:
    return '''
      {0.id}, {0.n}, {0.age}, {0.color[0]}, {0.color[1]}, {0.color[2]}, {0.position_mean[0]}, {0.position_mean[1]},
      {0.position_mean[2]}, {0.position_deviation[0]}, {0.position_deviation[1]}, {0.position_deviation[2]},
      {0.orientation_mean[0]}, {0.orientation_mean[1]}, {0.orientation_mean[2]}, {0.orientation_deviation},
      {0.radius_mean}, {0.radius_deviation}, {0.material.value}
    '''.format(self)
  
  def from_row(row) -> 'Feature':
    return Feature(
      row[0], row[1], row[2], np.asarray((row[3], row[4], row[5])), np.asarray((row[6], row[7], row[8])),
      np.asarray((row[9], row[10], row[11])), np.asarray((row[12], row[13], row[14])), row[15], row[16],
      row[17], Material(row[18])
    )

class SpatialIndex:
  def __init__(self) -> 'SpatialIndex':
    property = index.Property()
    property.dimension = 3
    property.dat_extension = 'data'
    property.idx_extension = 'index'
    self.rtree = index.Rtree(os.path.join(CONFIG.databasePath, 'spatialindex'), properties=property, interleaved=False)

  def insert(self, feature: Feature):
    self.rtree.insert(feature.id, feature.bbox(0.0))
  
  def delete(self, id: int, bbox: BoundingBox):
    self.rtree.delete(id, bbox)

  def count(self) -> int:
    return self.rtree.count()
  
  def intersection(self, box: BoundingBox) -> Generator[int, None, None]:
    return self.rtree.intersection(box)

@unique
class Observe(Enum):
  PROBABILITY = 0
  VISUAL_MEASUREMENT = 1
  PROCESSED = 2

@dataclass
class ProcessedFeatures:
  # (new_feature, correlated_feature, measurement)
  processed: List[Tuple[Feature, Optional[Feature], Optional[VisualMeasurement]]]
  # bbox: BoundingBox compute min & max along each axis for aging process

ObserveResult = Union[None, ProcessedFeatures, List[VisualMeasurement]]

class FeatureDatabase:
  SIMILARITY_THRESHOLD = 0.15
  # All fields except the primary key
  ALL = '''
    id, n, age, color_r, color_g, color_b, position_mean_x, position_mean_y, position_mean_z, position_deviation_x,
    position_deviation_y, position_deviation_z, orientation_mean_x, orientation_mean_y, orientation_mean_z,
    orientation_deviation, radius_mean, radius_deviation, material
  '''

  def __init__(self) -> 'FeatureDatabase':
    self.index = SpatialIndex()
    self.connection = sqlite3.connect(os.path.join(CONFIG.databasePath, 'recognition.sqlite'))
    cur = self.connection.cursor()
    cur.execute('''
      CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY,
        n INTEGER,
        age INTEGER,
        color_r INTEGER,
        color_g INTEGER,
        color_b INTEGER,
        position_mean_x REAL,
        position_mean_y REAL,
        position_mean_z REAL,
        position_deviation_x REAL,
        position_deviation_y REAL,
        position_deviation_z REAL,
        orientation_mean_x REAL,
        orientation_mean_y REAL,
        orientation_mean_z REAL,
        orientation_deviation REAL,
        radius_mean REAL,
        radius_deviation REAL,
        material INTEGER
      )
    ''')
    cur.execute('SELECT MAX(id) FROM features')
    row = cur.fetchone()
    self.id_counter = row[0] if row[0] is not None else 1
  
  def update_features(self, frame: List[Feature]):
    cur = self.connection.cursor()
    for feature in frame:
      if feature.id == 0:
        feature.id = self.id_counter
        self.id_counter += 1
      self.index.insert(feature)
      cur.execute('INSERT OR REPLACE INTO features ({all}) VALUES ({feature})'.format(
        all=FeatureDatabase.ALL,
        feature=feature.value()
      ))
    self.connection.commit()

  def batch_select(self, ids: List[int]) -> List[Feature]:
    cur = self.connection.cursor()
    cur.execute('SELECT * FROM features WHERE id in ({})'.format(','.join(['?'] * len(ids))), ids)
    rows = cur.fetchall()
    return [Feature.from_row(row) for row in rows]

  def clear(self):
    cur = self.connection.cursor()
    cur.execute('DELETE FROM features')
    self.connection.commit()
    self.index = None
    try:
      os.remove(os.path.join(CONFIG.databasePath, 'spatialindex.data'))
      os.remove(os.path.join(CONFIG.databasePath, 'spatialindex.index'))
    except OSError:
      pass
    self.index = SpatialIndex()
    self.id_counter = 1
  
  def cold_localize(self, frame: List[Feature]) -> State:
    # Robot is lost -> correlate feature frame to environment to determine starting location
    # Perform graph based probability search:
    # * Find a feature with similar properties to current feature than guess the estimate as the exact delta to this feature
    pass

  def observe(self, estimate: State, frame: List[Feature], what = Observe.PROBABILITY) -> Tuple[ObserveResult, float]:
    result = None
    if what is Observe.PROCESSED:
      result = ProcessedFeatures([])
    elif what is Observe.VISUAL_MEASUREMENT:
      result = []
    probability_accum = 0.0
    n = 0
    for feature in frame:
      transformed = feature.apply_basis(estimate)
      max_probability = 0.0
      max_feature = None
      intersection = [id for id in self.index.intersection(transformed.bbox(estimate.position_deviation))]
      intersection = self.batch_select(intersection)
      for intersect in intersection:
        probability = intersect.probability(transformed, estimate)
        # print(probability)
        if probability > max_probability:
          max_feature = intersect
          max_probability = probability
      if max_feature is not None:
        n += 1
        if what is not Observe.PROBABILITY:
          measurement = VisualMeasurement(transformed, max_feature.delta(transformed), max_probability)
          if what is Observe.PROCESSED:
            if max_probability >= FeatureDatabase.SIMILARITY_THRESHOLD:
              result.processed.append((transformed, max_feature, measurement))
            else:
              result.processed.append((transformed, None, measurement))
          elif what is Observe.VISUAL_MEASUREMENT:
            result.append(measurement)
        probability_accum += max_probability
      elif what is Observe.PROCESSED:
        result.processed.append((transformed, None, None))

    if n == 0:
      return result, 0.0
    else:
      return result, probability_accum / float(n)
  
  def apply_features(self, processed: ProcessedFeatures):
    add = []
    for feature, source, measurement in processed.processed:
      if source is not None:
        self.index.delete(source.id, source.bbox(0.0))
        merged = source.merge(feature, measurement.importance_weight)
        add.append(merged)
      else:
        add.append(feature)
    self.update_features(add)

@dataclass
class Voxel:
  id: int = 0
  n: int = 1
  position: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))
  color: Color = field(default_factory=lambda: np.asarray((0, 0, 0)))
  material: List[int] = field(default_factory=lambda: np.asarray([0.0] * 24))

  def hash(self) -> str:
    return "{},{}".format(self.position[0], self.position[1])
  
  def from_row(row) -> 'Voxel':
    return Voxel(
      row[0], row[1], (row[2], row[3], row[4]), np.asarray((row[5], row[6], row[7])),
      list(map(lambda x: float(x), row[8].split(',')))
    )
  
  def value(self) -> Tuple:
    return (
      self.id, self.n, self.position[0], self.position[1], self.position[2], int(self.color[0]),
      int(self.color[1]), int(self.color[2]), ','.join(str(mat) for mat in self.material)
    )

class OccupancyDatabase:
  VOXEL_SIZE = 0.1
  ALL = "id, n, x, y, z, color_r, color_g, color_b, material"

  def __init__(self) -> 'OccupancyDatabase':
    self.connection = sqlite3.connect(os.path.join(CONFIG.databasePath, 'occupancy.sqlite'))
    cur = self.connection.cursor()
    cur.execute('''
      CREATE TABLE IF NOT EXISTS occupancy (
        id INTEGER PRIMARY KEY,
        n INTEGER,
        x INTEGER,
        y INTEGER,
        z INTEGER,
        color_r INTEGER,
        color_g INTEGER,
        color_b INTEGER,
        material INTEGER,
        CONSTRAINT XY UNIQUE (x, y, z)
      )
    ''')
    cur.execute('SELECT MAX(id) FROM occupancy')
    row = cur.fetchone()
    self.id_counter = row[0] if row[0] is not None else 1
  
  def clear(self):
    cur = self.connection.cursor()
    cur.execute('DELETE FROM occupancy')
    self.connection.commit()
  
  def apply_voxels(self, image: Any, points3d: Any, disparity, state: State):
    THRESHOLD = 5
    SAMPLES = 1000
    angle = angle_between(state.forward, Z_AXIS)
    axis = np.cross(state.forward, Z_AXIS)
    rotation = angle_axis(axis, angle)
    voxels: List[Voxel] = []
    mins = np.asarray([0.0, 0.0, 0.0])
    maxes = np.asarray([0.0, 0.0, 0.0])
    total = 0
    previous = {}
    while total < SAMPLES:
      i = random.randint(0, CONFIG.width - 1)
      j = random.randint(0, CONFIG.height - 1)
      if not "{},{}".format(i, j) in previous:
        if disparity[j, i] > THRESHOLD and not isinf(disparity[j, i]):
          v = np.dot(rotation, pixel_ray(Z_AXIS, i, j)) * (points3d[j, i, 2] / 1000.0)
          position = np.floor((v + state.position) / OccupancyDatabase.VOXEL_SIZE)
          voxel = Voxel(
            position=(position[0], position[1], position[2]),
            color=image[j, i],
            material=np.asarray([0.0] * 23 + [1.0])
          )
          voxels.append(voxel)
          previous[voxel.hash()] = voxel
          mins = np.minimum(position, mins)
          maxes = np.maximum(position, maxes)
          total += 1 # What if image is entirely out of threshold
    cur = self.connection.cursor()
    cur.execute(
      'SELECT * FROM occupancy WHERE x >= ? AND x <= ? AND y >= ? AND y <= ? AND z >= ? AND z <= ?',
      [mins[0], maxes[0], mins[1], maxes[1], mins[2], maxes[2]]
    )
    previous = {}
    rows = cur.fetchall()
    for row in rows:
      voxel = Voxel.from_row(row)
      previous[voxel.hash()] = voxel
    values = []
    for voxel in voxels:
      if voxel.hash() in previous:
        original = previous[voxel.hash()]
        n = original.n + 1
        voxel.id = original.id
        voxel.n = n
        voxel.color = np.floor(original.color * (n - 1) / n + voxel.color / n).astype(np.uint8)
        voxel.material = original.material * (n - 1) / n + voxel.material / n
      else:
        voxel.id = self.id_counter
        self.id_counter += 1
      previous[voxel.hash()] = voxel
      values.append(voxel.value())
    cur.executemany('INSERT OR REPLACE INTO occupancy VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
    self.connection.commit()
  
  # def visualize(self) -> Any:

metadata_database = MetadataDatabase()
feature_database = FeatureDatabase()
occupancy_database = OccupancyDatabase()
