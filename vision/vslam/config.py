from enum import Enum, Flag, unique
from typing import Optional

class AppMode(Enum):
  FOLLOW = 0,
  BLADE = 1

@unique
class DebugWindows(Flag):
  NONE = 0
  CAMERA = 1
  REMAP = 2
  DEPTH = 4
  KEYPOINT = 8
  SEMANTIC = 16
  ALL = 31

  def __contains__(self, item: 'DebugWindows'):
    return (self.value & item.value) == item.value
  
  def parse(input: Optional[str]) -> 'DebugWindows':
    if input is None:
      return DebugWindows.NONE
    split = input.replace(' ', '').split('|')
    result = DebugWindows.NONE
    if 'camera' in split:
      result = result | DebugWindows.CAMERA
    if 'remap' in split:
      result = result | DebugWindows.REMAP
    if 'depth' in split:
      result = result | DebugWindows.DEPTH
    if 'keypoint' in split:
      result = result | DebugWindows.KEYPOINT
    if 'semantic' in split:
      result = result | DebugWindows.SEMANTIC
    if 'all' in split:
      return DebugWindows.ALL
    return result

class Config:
  def __init__(
    self,
    mode: AppMode = AppMode.FOLLOW,
    debug: bool = True,
    interval: float = 5.0,
    windows: DebugWindows = DebugWindows.NONE,
    dataPath: str = 'data',
    databasePath: str = 'com',
    width: int = 960,
    height: int = 540,
    map_width: int = 460,
    map_height: int = 634
  ) -> 'Config':
    self.mode = mode
    self.debug = debug
    self.interval = interval
    self.windows = windows
    self.dataPath = dataPath
    self.databasePath = databasePath
    self.width = width
    self.height = height
    self.map_width = map_width
    self.map_height = map_height

CONFIG = Config()
