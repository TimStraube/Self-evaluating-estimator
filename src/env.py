import numpy as np

class World():
  def __init__(self):
    self.world = np.array([0])

  def observe(self):
    return self.world

  def act(self, action):
    # print("World: " + str(self.world))
    if action == 0:
      self.world -= 1
    elif action == 1:
      self.world += 1
