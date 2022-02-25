import json
import math
import os
from rtree import index
import sqlite3
import statistics
import scipy.stats as st
from typing import Any, Generator, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from vslam.camera import angle_between, normalize, pixel_ray, projection
from vslam.config import CONFIG
from vslam.segmentation import Material
from vslam.state import Color, Delta, Position, State, Vector

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
  color: Color = np.asarray((0, 0, 0))
  position_mean: Position = np.asarray((0.0, 0.0, 0.0))
  position_deviation: Vector = (math.inf, math.inf, math.inf)
  orientation_mean: Vector = np.asarray((0.0, 0.0, 1.0))
  orientation_deviation: float = math.inf
  radius_mean: float = 0.0
  radius_deviation: float = 0.0
  material: Material = Material.UNDEFINED

  def create(kp, image, points3d, disparity, state: State) -> Optional['Feature']:
    THRESHOLD = 0.001
    SIZE_THRESHOLD = 3.0
    x, y = round(kp.pt[0]), round(kp.pt[1])
    window_size = math.ceil(kp.size / 2.0)
    if kp.size < SIZE_THRESHOLD:
      return None
    color = np.asarray((0, 0, 0))
    n = 0.0
    depth = []
    for i in range(x - window_size, x + window_size):
      for j in range(x - window_size, y + window_size):
        if math.sqrt((i - x) ** 2 + (j - y) ** 2) <= window_size:
          n += 1.0
          color = color * (n - 1) / n + np.asarray(image[i][j]) / n
          if disparity[i][j] > THRESHOLD:
            depth.append(float(points3d[i][j][2] / 1000.0)) # Q matrix was computed using mm

    if len(depth) < 2:
      return None

    right = np.cross(state.forward, state.up)
    v = pixel_ray(state.forward, kp.pt[0], kp.pt[1])
    radius = kp.size / 2.0
    
    return Feature(
      color=np.asarray((math.floor(color[0]), math.floor(color[1]), math.floor(color[2]))),
      position_mean=state.position + v * statistics.mean(depth),
      position_deviation=(statistics.stdev(depth) ** 2) * v,
      orientation_mean=math.cos(kp.angle) * right + math.sin(kp.angle) * state.up,
      orientation_deviation=math.pi,
      radius_mean=radius,
      radius_deviation=1.0 - kp.response
    )
  
  def incremental_mean(self, mean, sample, n: int) -> Any:
    """
    Merge two means incrementally (self has n - 1 samples other has 1 sample).
    """
    return mean * float(n - 1) / n + sample / float(n)

  def incremental_deviation(self, deviation, mean, sample, n: int) -> Any:
    """
    Merge two deviations incrementally (self has n - 1 samples other has 1 sample).
    """
    return (float(n - 2) / (n - 1)) * deviation + np.power(sample - mean, 2) / float(n)

  def deviation(self, other: 'Feature') -> Tuple[float, float, float]:
    """
    Compute position, radius and orientation deviations between two features.
    """
    assert other.n == 1
    delta_p = other.position_mean - self.position_mean
    # Unit vector from feature 1 -> 2
    r = normalize(delta_p)

    # Incrementally average deviation components based on number of samples n with deviation
    # components computed as the projection of deviation vector onto r for both features
    position_deviation = self.incremental_mean(
      abs(projection(self.position_deviation, r)),
      abs(projection(other.position_deviation, r)),
      self.n + 1
    )
    radius_deviation = self.incremental_mean(self.radius_deviation, other.radius_deviation, self.n + 1)
    # TODO: properly compute orientation deviation
    orientation_deviation = self.orientation_deviation + other.orientation_deviation
    return position_deviation, radius_deviation, orientation_deviation
  
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
  
  def probability(self, other: 'Feature', estimate: State) -> Tuple[float, float, float]:
    """
    Computes the probability P(yt has position within sigma(x) of z | x, z),
    P(yt has size within sigma(z) of z | z) & P(yt has orientation within sigma(x) of z | z)
    where
    * `other` (yt) is the feature representing the current visual measurement
    * `self` (z) is a previously recognized feature
    * `estimate` (x) is the current state estimate
    """
    # NOTE: we are assuming that position, radius & orientation are independent
    # Compute deviations
    dp, dr, do = self.deviation(other)
    delta_p = other.position_mean - self.position_mean
    r = np.linalg.norm(delta_p)
    # Compute the sigma value (num deviations from mean) of difference between features (based
    # on distributions projected onto the line subtending features)
    upper_p = (r + estimate.position_deviation) / dp
    lower_p = (r - estimate.position_deviation) / dp
    # Computes P(-sigma(x) < delta_p < sigma(x)) using CDF of standard normal distribution
    position_probability = st.norm.cdf(upper_p) - st.norm.cdf(lower_p)
    delta_r = abs(other.radius_mean - self.radius_mean)
    upper_r = (delta_r + self.radius_deviation) / dr
    lower_r = (delta_r - self.radius_deviation) / dr
    # Computes P(-sigma(z) < delta_r < sigma(z)) using CDF of standard normal distribution
    radius_probability = st.norm.cdf(upper_r) - st.norm.cdf(lower_r)
    v1 = self.position_mean - estimate.position
    v2 = other.position_mean - estimate.position
    angle = angle_between(v1, v2)
    upper_o = (angle + estimate.orientation_deviation) / do
    lower_o = (angle - estimate.orientation_deviation) / do
    orientation_probability = st.norm.cdf(upper_o) - st.norm.cdf(lower_o)
    return position_probability, radius_probability, orientation_probability
  
  def merge(self, other: 'Feature') -> 'Feature':
    assert other.n == 1
    n = self.n + 1
    color = self.incremental_mean(self.color, other.color, n)
    return Feature(
      id=self.id,
      color=np.asarray((math.floor(color[0]), math.floor(color[1]), math.floor(color[2]))),
      n=n,
      position_mean=self.incremental_mean(self.position_mean, other.position_mean, n),
      position_deviation=self.incremental_deviation(self.position_deviation, self.position_mean, other.position_deviation, n),
      orientation_mean=self.incremental_mean(self.orientation_mean, other.orientation_mean, n),
      orientation_deviation=self.incremental_deviation(self.orientation_deviation, 0.0, other.orientation_deviation, n),
      radius_mean=self.incremental_mean(self.radius_mean, other.radius_mean, n),
      radius_deviation=self.incremental_deviation(self.radius_deviation, self.radius_mean, other.radius_deviation, n),
      material=self.material
    )

  def measurement(self, other: 'Feature', state: State) -> VisualMeasurement:
    return VisualMeasurement(
      self.id,
      other.position_mean - state.position
    )
  
  def adjust(self, delta: Delta):
    self.position_mean += delta.delta_position
    self.orientation_mean += delta.delta_orientation
  
  def bbox(self) -> BoundingBox:
    radius = self.radius_mean + self.radius_deviation
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
    self.rtree.insert(feature.id, feature.bbox())
  
  def delete(self, id: int, bbox: BoundingBox):
    self.rtree.delete(id, bbox)

  def count(self) -> int:
    return self.rtree.count()
  
  def intersection(self, box: BoundingBox) -> Generator[int, None, None]:
    return self.rtree.intersection(box)

@dataclass
class ProcessedFeatures:
  processed: List[Tuple[Feature, Optional[Feature]]]

class FeatureDatabase:
  SIMILARITY_THRESHOLD = 0.6
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
    for id in ids:
      cur.execute('SELECT * FROM features WHERE id={}'.format(id))
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
    pass
  
  def process_features(self, estimate: State, frame: List[Feature]) -> Tuple[List[VisualMeasurement], ProcessedFeatures]:
    measurements = []
    processed = []
    n = 0
    for feature in frame:
      max_similarity = 0.0
      max_feature = None
      intersection = [id for id in self.index.intersection(feature.bbox())]
      for intersect in self.batch_select(intersection):
        similarity = intersect.similarity(feature)
        if similarity > FeatureDatabase.SIMILARITY_THRESHOLD and similarity > max_similarity:
          max_feature = intersect
      if max_feature is not None:
        # TODO: compute P(measurement|state,action) using parametric solution to gaussian distribution P(nearest intersect point < N < opposite of intersection)
        processed.append((feature, max_feature))
        n += 1
        measurements.append(max_feature.measurement(feature, estimate))
      else:
        processed.append((feature, None))

    return measurements, ProcessedFeatures(processed)
  
  def apply_features(self, last_to_next: Delta, processed: ProcessedFeatures):
    add = []
    for feature, source in processed.processed:
      if source is not None:
        self.index.delete(source.id, source.bbox())
        feature.adjust(last_to_next)
        merged = source.merge(feature)
        add.append(merged)
      else:
        feature.adjust(last_to_next)
        add.append(feature)
    self.update_features(add)

metadata_database = MetadataDatabase()
feature_database = FeatureDatabase()
