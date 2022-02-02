from enum import IntEnum, unique

@unique
class DebugWindows(IntEnum):
  NONE = 0
  DEPTH = 1
  ALL = 3

  def __contains__(self, item: 'DebugWindows'):
    return (self.value & item.value) == item.value

class Config:
  def __init__(self, debug: bool = True, windows: DebugWindows = DebugWindows.NONE, dataPath: str = 'data') -> 'Config':
    self.debug = debug
    self.windows = windows
    self.dataPath = dataPath

CONFIG = Config()
