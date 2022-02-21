from enum import Flag, unique
from typing import Optional

@unique
class DebugWindows(Flag):
  NONE = 0
  CAMERA = 1
  REMAP = 2
  DEPTH = 4
  KEYPOINT = 8
  ALL = 15

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
    if 'all' in split:
      return DebugWindows.ALL
    return result

class Config:
  def __init__(
    self,
    debug: bool = True,
    interval: float = 5.0,
    windows: DebugWindows = DebugWindows.NONE,
    dataPath: str = 'data',
    width: int = 960,
    height: int = 540
  ) -> 'Config':
    self.debug = debug
    self.interval = interval
    self.windows = windows
    self.dataPath = dataPath
    self.width = width
    self.height = height

CONFIG = Config()
