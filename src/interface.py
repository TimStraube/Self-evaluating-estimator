# World interface
import numpy as np

class Interface():
  def __init__(self):
    self.world = np.array([0, 0])

  def observe(self):
    return self.world

  def act(self, action):
    # print("World: " + str(self.world))
    self.world = self.world + action - 3
