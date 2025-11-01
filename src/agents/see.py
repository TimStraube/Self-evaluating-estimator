import gymnasium as gym
import numpy as np
from src.envs import random
from src.agents.memory import Memory


class SEE(gym.Env):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.memory = Memory(capacity, (5,))
        self.environment = random.RandomEnv()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=10,
            shape=(5,),
            dtype=np.float32
        )
        # First action: environment action (5 discrete choices)
        # Next 5 actions: memory weights (discretized to 10 levels each)
        self.action_space = gym.spaces.MultiDiscrete([5, 10, 10, 10, 10, 10])

    def reset(self, seed=None, options=None):
        obs = self.environment.reset()
        obs = np.array(obs, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Action environment is a single integer
        a_u = action[0]
        # Action memory: convert discretized values (0-9) to continuous weights (0-1)
        # and normalize to sum to 1
        memory_actions = action[1:].astype(np.float32) / 9.0  # Scale to [0, 1]
        a_c = memory_actions / np.sum(memory_actions) if np.sum(memory_actions) > 0 else np.ones(5, dtype=np.float32) / 5

        o_u, done, info = self.environment.step(a_u)

        # Transform observation to the frequency domain
        O_u = np.fft.fft(o_u).real

        # ===========================
        # Step 1: Perception
        # ===========================

        # Compute predicted observation
        O_p = self.memory.getAllImages().T @ a_c.flatten()

        # Compute perceived observation in frequency domain
        O_y = O_p * O_u

        # Compute perceived reward
        r_y = np.var(O_y) * (np.sum(O_p ** 2) - np.sum(O_u ** 2))

        # Step 2: Evaluation

        # Compute evaluation metric
        r = r_y - np.sum(a_c * self.memory.getReward())

        # Step 3: Update

        # Update memory with new observation and reward
        O_c = (1 - r_y) * O_p + r_y * O_u
        # Update rewards depending on the memory action and the achieved reward
        r_c = 0.5 * (self.memory.getReward() + a_c * r_y)

        self.memory.update(
            O_c,
            r_c
        )

        truncated = False
        info = {}
        # Ensure correct data types for Stable Baselines3
        O_y = np.array(O_y, dtype=np.float32)
        r = float(r)
        return O_y, r, done, truncated, info

    def render(self):
        pass
