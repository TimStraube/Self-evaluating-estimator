import gym
import numpy as np

class RandomEnv(gym.Env):
    """
    Ein einfaches Gym-Environment mit zufälligen Beobachtungen und Aktionen.
    Beobachtungsraum: Box mit Form (4,)
    Aktionsraum: Diskret mit 3 Aktionen
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        obs = self.observation_space.sample()
        return obs

    def step(self, action):
        obs = self.observation_space.sample()
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
