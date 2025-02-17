# World interface
import numpy as np

class Interface():
  def __init__(self):
    self.world = np.array([0, 0])

  def observe(self):
    return self.world

  def act(self, action):
    # print("World: " + str(self.world))
    if action == 0:
      self.world = self.world - 2
    elif action == 1:
      self.world = self.world - 1
    elif action == 2:
      self.world = self.world + 1
    else:
      self.world = self.world + 2
