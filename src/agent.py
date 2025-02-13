import gymnasium
import numpy as np
import stable_baselines3 as sb3
from src.stateOfMind import StateOfMind
from src.memory import Memory
from interface import Interface
import asyncio

class Agent(gymnasium.Env):
  def __init__(self):
    super().__init__()
    self.stateOfMind = StateOfMind()
    # Define a space for stateOfMind observations; for example, assume it outputs a vector of length 10
    state_dim = 2
    state_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
    # Define an additional box observation
    additional_box = gymnasium.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    # Combine both into a single observation as a Dict space
    self.observation_space = gymnasium.spaces.Dict({
      "stateOfMind": state_space,
      "additional": additional_box
    })
    self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    # Action on the world and thougth pointer
    self.action_space = gymnasium.spaces.Discrete(2)
    self.stateOfMind = StateOfMind()
    self.memory = Memory(1000)
    self.world = Interface(self)
    self.averageReward = 0

  def step(self, action):
    world_input = self.world.observe()

    # stateOfMind
    current_state_of_mind = self.memory.getStateOfMind()
    observation = {
      "stateOfMind": current_state_of_mind,
      "additional": world_input
    }

    self.memory.update(world_input)

    value = self.evaluate(current_state_of_mind, world_input)
    self.averageReward = (self.averageReward + value) / 2
    reward = value - self.averageReward

    # Machine act
    self.world.act(action[0])

    self.memory.setThoughtPointer(action[1])

    return observation, reward, False, {}

  def reset(self):
      return self.observation_space.sample()

  # The agent is rewarded based on the familarity between the state of mind and the world input  
  def evaluate(self, state_of_mind, world_input):
    reward_state_of_mind = 0
    reward_world_input = 0
    
    return np.sqrt(reward_state_of_mind ** 2 + reward_world_input ** 2)