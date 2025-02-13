import random

class Memory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.thoughtPointer = 0
    self.rewards = []

  def update(self, stateOfMind):
    # For a new input a stateOfMind is forgotten
    self.memory.insert(0, stateOfMind)
    if len(self.memory) <= self.capacity:
      return
    index = random.randrange(len(self.memory))
    self.memory.pop(index)

  def getStateOfMind(self):
    thought = self.memory[self.thoughtPointer]
    self.thoughtPointer += 1
    return thought

  def setThoughtPointer(self, thoughtPointer):
    self.thoughtPointer = thoughtPointer
    

