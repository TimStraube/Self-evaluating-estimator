import numpy as np

class World():
  def __init__(self):
    self.world = np.array([0])

  def observe(self):
    return self.world

  def act(self, action):
    if action == 0:
      self.world = self.world - 1
    else:
      self.world = self.world + 1
