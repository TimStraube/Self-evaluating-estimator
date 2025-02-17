import random
import numpy as np

class Memory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = np.zeros((capacity, 2))
    self.thoughtPointer = 0
    self.rewards = []

  def update(self, stateOfMind):
    # print("Updating memory with state of mind: " + str(stateOfMind))
    # For a new input a stateOfMind is forgotten
    self.memory[self.thoughtPointer] = stateOfMind

  def getStateOfMind(self):
    thought = self.memory[self.thoughtPointer]
    return thought

  def setThoughtPointer(self, thoughtPointer):
    self.thoughtPointer = thoughtPointer
    
  def getCapacity(self):
    return self.capacity
  
  def __str__(self):
    return str(self.memory)

