import random
import numpy as np

class Memory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = np.zeros((capacity, 2))
    self.thoughtPointer = 0
    self.rewards = np.zeros((capacity, 1))

  def update(self, stateOfMind, reward):
    # print("Updating memory with state of mind: " + str(stateOfMind))
    # For a new input a stateOfMind is forgotten
    self.memory[self.thoughtPointer] = stateOfMind
    self.rewards[self.thoughtPointer] = reward

  def getStateOfMind(self):
    thought = self.memory[self.thoughtPointer]
    return thought

  def setThoughtPointer(self, thoughtPointer):
    self.thoughtPointer = thoughtPointer
    
  def getCapacity(self):
    return self.capacity
  
  def getReward(self, thoughtPointer):
    return self.rewards[thoughtPointer]
  
  def getMeanReward(self):
    return np.mean(self.rewards)
  
  def __str__(self):
    return str(self.memory)

