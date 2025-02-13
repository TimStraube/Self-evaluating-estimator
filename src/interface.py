# World interface

class Interface():
  def __init__(self, world):
    self.world = world

  async def observe(self):
    pass

  async def act(self, action):
    pass
