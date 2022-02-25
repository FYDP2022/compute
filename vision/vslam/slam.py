from particlefilter import ParticleFilterLoc

from vslam.state import ControlState, Measurement, State

class SLAM:
  """
  Simultaneous mapping and localization module.
  """

  def __init__(self) -> 'SLAM':
    pass

  def step(self, estimate: State, action: ControlState, measurement: Measurement):
    pf = ParticleFilterLoc()