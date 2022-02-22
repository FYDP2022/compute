import math
import os
from rtree import index
import sqlite3
import statistics
from typing import Any, Generator, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from vslam.camera import angle_between, pixel_ray
from vslam.config import CONFIG
from vslam.segmentation import Material
from vslam.state import Color, Delta, Position, State, Vector

# NOTE: bounding boxes are in the form (x1, x2, y1, y2, z1, z2)
BoundingBox = Tuple[float, float, float, float, float, float]

class MetadataDatabase:
  def __init__(self) -> 'MetadataDatabase':
    pass

@dataclass
class Feature:
  id: int = 0
  n: int = 1
  age: int = 0
  color: Color = (0, 0, 0)
  position_mean: Position = (0.0, 0.0, 0.0)
  position_deviation: Vector = (math.inf, math.inf, math.inf)
  orientation_mean: Vector = (0.0, 0.0, 1.0)
  orientation_deviation: float = math.inf
  radius_mean: float = 0.0
  radius_deviation: float = 0.0
  material: Material = Material.BACKGROUND

  def create(kp, image, points3d, disparity, state: State) -> Optional['Feature']:
    THRESHOLD = 0.001
    x, y = round(kp.pt.x), round(kp.pt.y)
    window_size = math.ceil(kp.size / 2.0)
    color = (0, 0, 0)
    n = 0.0
    depth = []
    for i in range(x - window_size, x + window_size):
      for j in range(x - window_size, y + window_size):
        if math.sqrt((i - x) ** 2 + (j - y) ** 2) <= window_size:
          color = (1.0 / n) * (color * n + image[i][j])
          if disparity[i][j] > THRESHOLD:
            depth.append(points3d[i][j])
          n += 1.0

    if len(depth) == 0:
      return None

    right = np.cross(state.forward, state.up)
    v = pixel_ray(state.forward, kp.pt.x, kp.pt.y)
    radius = kp.size / 2.0
    
    return Feature(
      color=color,
      position_mean=state.position + v * points3d[math.round(kp.pt.x)][math.round(kp.pt.y)],
      position_deviation=(statistics.stdev(depth) ** 2) * v,
      orientation_mean=math.cos(kp.angle) * right + math.sin(kp.angle) * state.up,
      orientation_deviation=math.pi,
      radius_mean=radius,
      radius_deviation=1.0 - kp.response
    )
  
  def incremental_mean(self, mean, sample, n: int) -> Any:
    return mean * float(n - 1) / n + sample / float(n)

  def incremental_deviation(self, deviation, mean, sample, n) -> Any:
    return (float(n - 2) / (n - 1)) * deviation + np.power(sample - mean, 2) / n
  
  def similarity(self, other: 'Feature') -> float:
    dist = np.linalg.norm(other.position_mean - self.position_mean)
    overlap = 1.0 - max(dist / (self.radius_mean + self.radius_deviation + other.radius_mean + other.radius_deviation), 1.0)
    angle = angle_between(self.orientation_mean, other.orientation_mean)
    alignment = 1.0 - max(angle / (self.orientation_deviation + other.orientation_deviation), 1.0)
    return overlap * alignment * (self.material == other.material)
  
  def merge(self, other: 'Feature') -> 'Feature':
    assert other.n == 1
    n = self.n + 1
    return Feature(
      id=self.id,
      color=self.incremental_mean(self.color, other.color, n),
      n=n,
      position_mean=self.incremental_mean(self.position_mean, other.position_mean, n),
      position_deviation=self.incremental_deviation(self.position_deviation, self.position_mean, other.position_deviation, n),
      orientation_mean=self.incremental_mean(self.orientation_mean, other.orientation_mean, n),
      orientation_deviation=self.incremental_deviation(self.orientation_deviation, self.orientation_mean, other.orientation_deviation, n),
      radius_mean=self.incremental_mean(self.radius_mean, other.radius_mean, n),
      radius_deviation=self.incremental_deviation(self.radius_deviation, self.radius_mean, other.radius_deviation, n),
      material=self.material
    )

  def delta(self, other: 'Feature') -> Delta:
    Delta(
      other.position_mean - self.position_mean,
      other.orientation_mean - self.orientation_mean
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
      {.n}, {.age}, {.color_r}, {.color_g}, {.color_b}, {.position_mean_x}, {.position_mean_y},
      {.position_mean_z}, {.position_deviation_x}, {.position_deviation_y}, {.position_deviation_z},
      {.orientation_mean_x}, {.orientation_mean_y}, {.orientation_mean_z}, {.orientation_deviation},
      {.radius_mean}, {.radius_deviation}, {.material}
    '''.format(self)
  
  def from_row(row) -> 'Feature':
    return Feature(
      row[0], row[1], row[2], (row[3], row[4], row[5]), (row[6], row[7], row[8]),
      (row[9], row[10], row[11]), (row[12], row[13], row[14]), row[15], row[16], row[17], Material(row[18])
    )

class SpatialIndex:
  def __init__(self) -> 'SpatialIndex':
    property = index.Property()
    property.dimension = 3
    property.dat_extension = 'data'
    property.idx_extension = 'index'
    self.rtree = index.Rtree(os.path.join(CONFIG.databasePath, 'spatialindex'), properties=property)

  def insert(self, feature: Feature):
    self.rtree.insert(feature.id, feature.bbox())

  def insert_all(self, generator: Generator[Tuple[int, BoundingBox]]):
    self.rtree.insert(generator)
  
  def delete(self, id: int, bbox: BoundingBox):
    self.rtree.delete(id, bbox)

  def count(self) -> int:
    return self.rtree.count()
  
  def intersection(self, box: BoundingBox) -> Generator[int]:
    return self.rtree.intersection(box)

@dataclass
class ProcessedFeatures:
  processed: List[Tuple[Feature, Optional[Feature]]] = []

class FeatureDatabase:
  SIMILARITY_THRESHOLD = 0.8
  # All fields except the primary key
  ALL = '''
    n, age, color_r, color_g, color_b, position_mean_x, position_mean_y, position_mean_z, position_deviation_x,
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
    self.last_frame: List[Feature] = None
  
  def update_features(self, frame: List[Feature]):
    cur = self.connection.cursor()
    cur.execute('BEGIN TRANSACTION')
    for feature in frame:
      cur.execute('INSERT OR REPLACE INTO features ({id_key}{all}) VALUES ({id_val}{feature})'.format(
        id_key='id, ' if feature.id != 0 else '',
        id_val='{}, '.format(feature.id) if feature.id != 0 else '',
        all=FeatureDatabase.ALL,
        feature=feature.value()
      ))
    cur.execute('COMMIT')

  def batch_select(self, ids: List[int]) -> List[Feature]:
    cur = self.connection.cursor()
    cur.execute('BEGIN TRANSACTION')
    for id in ids:
      cur.execute('SELECT * FROM features WHERE id={}'.format(id))
    cur.execute('COMMIT')
    rows = cur.fetchall()
    return [Feature.from_row(row) for row in rows]

  def localize(self, frame: List[Feature]) -> State:
    # Robot is lost -> correlate feature frame to environment to determine starting location
    pass
  
  def process_features(self, estimate: State, frame: List[Feature]) -> Tuple[Delta, ProcessedFeatures]:
    delta = Delta()
    processed = []
    n = 0
    for feature in frame:
      max_similarity = 0.0
      max_feature = None
      for intersect in self.batch_select(self.index.intersection(feature.bbox())):
        similarity = intersect.similarity(feature)
        if similarity > FeatureDatabase.SIMILARITY_THRESHOLD and similarity > max_similarity:
          max_feature = intersect
      if max_feature is not None:
        processed.append((feature, max_feature))
        n += 1
        feature_delta = max_feature.delta(feature)
        delta.delta_position = delta.delta_position * float(n - 1) / n + feature_delta.delta_position / n
        delta.delta_orientation = delta.delta_orientation * float(n - 1) / n + feature_delta.delta_orientation / n
      else:
        processed.append((feature, None))

    return delta, ProcessedFeatures(processed)
  
  def apply_features(self, adjust: Delta, processed: ProcessedFeatures):
    add = []
    for feature, source in processed.processed:
      feature.adjust(adjust)
      if source is not None:
        merged = source.merge(feature)
        add.append(merged)
      else:
        add.append(feature)
    self.update_features(add)

metadata_database = MetadataDatabase()
feature_database = FeatureDatabase()
