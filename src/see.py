import gymnasium as gym
import numpy as np
from src.envs import random, arc
from src.memory import Memory


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
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(5),
            gym.spaces.Box(low=0.0, high=1.0, shape=(
                5,), dtype=np.float32) 
        ))

    def reset(self, seed=None, options=None):
        obs = self.environment.reset()
        obs = np.array(obs, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Action environment is a single integer
        a_u = action[0]
        # Action memory is a distribution over memory slots for example [0.1, 0.2, 0.3, 0.2, 0.2]
        a_c = action[1:] / np.sum(action[1:])

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
        return O_y, r, done, truncated, info

    def render(self):
        pass
