from dataclasses import dataclass
from enum import IntEnum, unique

@unique
class DebugWindows(IntEnum):
  NONE = 0
  DEPTH = 1
  ALL = 1

  def __contains__(self, item: 'DebugWindows'):
    return (self.value & item.value) == item.value

@dataclass
class Config:
  debug: bool = True
  windows: DebugWindows = DebugWindows.NONE
  dataPath: str = 'data'


CONFIG = Config()
