from enum import Flag, unique

@unique
class DebugWindows(Flag):
  NONE = 0
  CAMERA = 1
  REMAP = 2
  DEPTH = 4
  ALL = 7

  def __contains__(self, item: 'DebugWindows'):
    return (self.value & item.value) == item.value
  
  def parse(input: str) -> 'DebugWindows':
    split = input.replace(' ', '').split('|')
    result = DebugWindows.NONE
    if 'camera' in split:
      result = result | DebugWindows.CAMERA
    if 'remap' in split:
      result = result | DebugWindows.REMAP
    if 'depth' in split:
      result = result | DebugWindows.DEPTH
    if 'all' in split:
      return DebugWindows.ALL
    return result

class Config:
  def __init__(self, debug: bool = True, windows: DebugWindows = DebugWindows.NONE, dataPath: str = 'data') -> 'Config':
    self.debug = debug
    self.windows = windows
    self.dataPath = dataPath

CONFIG = Config()
