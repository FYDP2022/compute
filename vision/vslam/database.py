from enum import IntEnum, unique
import math
import sqlite3
from typing import Any, List
from dataclasses import dataclass

from vslam.state import Color, Delta, Position, State, Vector

connection = sqlite3.connect('recognition.db')

class MetadataDatabase:
  def __init__(self) -> 'MetadataDatabase':
    pass

@unique
class Material(IntEnum):
  BACKGROUND = 0

@dataclass
class Feature:
  id: int = 0
  n: int = 1
  color: Color = (0, 0, 0)
  position: Position = (0.0, 0.0, 0.0)
  position_variance: Vector = (0.0, 0.0, 0.0)
  radius_mean: float = 0.0
  radius_variance: float = 0.0
  material: Material = Material.BACKGROUND

  def create(kp, image, points3d, disparity, state: State) -> 'Feature':
    THRESHOLD = 0.001
    window_size = math.ceil(kp.size / 2)
    color = (0, 0, 0)
    n = 0.0
    n_depth = 0.0
    for i in range(kp.x - window_size, kp.x + window_size):
      for j in range(kp.y - window_size, kp.y + window_size):
        color = (1.0 / n) * (color * n + image[i][j])
        
        n += 1.0
    
    return Feature(
      color=color,
      position=points3d[math.round(kp.pt.x)][math.round(kp.pt.y)],
      position_variance=(1.0 - kp.response, 1.0 - kp.response,)
    )

class FeatureDatabase:
  def __init__(self) -> 'FeatureDatabase':
    cur = connection.cursor()
    cur.execute('''
      CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY,
        n INTEGER,
        color_r INTEGER,
        color_g INTEGER,
        color_b INTEGER,
        position_mean_x REAL,
        position_mean_y REAL,
        position_mean_z REAL,
        position_variance_x REAL,
        position_variance_y REAL,
        position_variance_z REAL,
        radius_mean REAL,
        radius_variance REAL,
        material INTEGER,
        CONSTRAINT Id_Frame UNIQUE (id, frame)
      )
    ''')
    self.last_frame: List[Feature] = None
  
  def process_features(self, state: State, features: List[Feature]) -> Delta:
    if self.last_frame is None:
      self.last_frame = features
      return Delta()
    delta = Delta()

    self.last_frame = features
    return delta
    

metadata_database = MetadataDatabase()
feature_database = FeatureDatabase()
