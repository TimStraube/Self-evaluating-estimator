import gymnasium as gym
import numpy as np

class RandomEnv():
    """
    """
    def __init__(self):
        super().__init__()

    def reset(self):
        # seed und options werden aktuell nicht verwendet, aber für Gymnasium-Kompatibilität akzeptiert
        obs = np.random.rand(5).astype(np.float32)
        return obs

    def step(self, action):
        obs = np.random.rand(5).astype(np.float32)
        reward = float(np.random.rand())
        done = bool(np.random.rand() > 0.95)
        info = {}
        return obs, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
