import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from memory import Memory
from env import World

class Agent(gymnasium.Env):
  def __init__(self):
    super().__init__()
    # Define a space for stateOfMind observations; for example, assume it outputs a vector of length 10
    state_dim = 2
    self.capacity = 5
    self.memory = Memory(self.capacity)
    self.world = World()
    # Combine both into a single observation as a Dict space
    self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
    # Action on the world and thougth pointer
    self.action_space = gymnasium.spaces.MultiDiscrete([
      2, 
      self.memory.getCapacity()]
    )
    self.averageReward = 0
    self.rewards = []
    self.world_states = []

  def reset(self, seed=None):
    self.memory = Memory(self.capacity)
    self.world = World()
    self.averageReward = 0
    return self.memory.getStateOfMind()

  def step(self, action):
    # print("Action: " + str(action))
    action_world = action[0]
    # Select a observation from the observation memory
    thought_pointer = action[1]
    # print("Thought pointer: " + str(thought_pointer))
    self.memory.setThoughtPointer(thought_pointer)
    current_state_of_mind = self.memory.getStateOfMind()

    # Get an observation from the world
    world_input = self.world.observe()

    # print("State of mind: " + str(current_state_of_mind))
    # print("World input: " + str(world_input))
    perception_difference = current_state_of_mind - world_input
    # print("Perception difference: " + str(perception_difference))
    value = self.evaluate(perception_difference)
    # print("Value: " + str(value))
    # print("Average reward: " + str(self.averageReward))
    # No activation function on the reward gain
    # reward =  value - self.averageReward
    # Logistic
    reward = 1 / (1 + np.exp(-(value - self.memory.getMeanReward())))

    # Machine act
    self.world.act(action_world)

    # Update the memory with the new observation
    self.memory.update(world_input, reward)
    self.world_states.append(world_input)

    # self.render()
    # print("Reward: " + str(reward))

    self.rewards.append(reward)
    return perception_difference, reward, False, False, {}
  
  def render(self):
    print(self.memory)

  # The agent is rewarded based on the familarity between the state of mind and the world input  
  def evaluate(self, perception_difference):   
    return -np.sum(np.abs(perception_difference))
  
if __name__ == '__main__':
  env = Agent()
  # Parallel environments
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(
    total_timesteps=1000,
    progress_bar=True
  )
  import matplotlib.pyplot as plt

  plt.plot(env.rewards)
  window_size = 20
  if len(env.rewards) >= window_size:
    filtered_rewards = np.convolve(env.rewards, np.ones(window_size) / window_size, mode='valid')
  else:
    filtered_rewards = env.rewards

  plt.plot(filtered_rewards)
  plt.xlabel('Timesteps')
  plt.ylabel('Reward')
  plt.title('Rewards over time')
  plt.show()
  import matplotlib.pyplot as plt

  # Convert list of world states to a numpy array
  world_states_array = np.array(env.world_states)

  # Plot each dimension of the world state separately
  for i in range(world_states_array.shape[1]):
    plt.plot(world_states_array[:, i], label=f'World State Dimension {i}')

  plt.legend()
  plt.xlabel('Timesteps')
  plt.ylabel('World State Value')
  plt.title('World States over Time')
  plt.show()
