from vslam.state import ControlState, Delta, State

class DynamicsModel:
  """
  Physical model of robot movement. Takes current belief and supplements it
  with knowledge of the control state of the system.
  """

  def __init__(self) -> 'DynamicsModel':
    pass

  def step(self, estimate: State, action: ControlState) -> Delta:
    return Delta()
