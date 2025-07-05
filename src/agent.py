import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.erinnerung import Erinnerung
from src.gedächtnis import Gedächtnis
from src.umwelt import Umwelt
from src.envs.test import Test

class Agent(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.gedächtnis = Gedächtnis(5)
        self.umwelt = Umwelt(Test())
        
        self.zustand_dimension = self.umwelt.observe().shape
        print(self.zustand_dimension)
        # Combine both into a single observation as a Dict space
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1.0, shape=self.zustand_dimension, dtype=np.int16)
        # Action on the world and thougth pointer
        self.action_space = gymnasium.spaces.MultiDiscrete([
            self.gedächtnis.getKapazität(),
            self.umwelt.umwelt.aktionsraum, 
        ])
        self.belohnungsErwartung = 0
        self.belohnungen = []
        self.umwelt_states = []

    def set_umwelt(self, umwelt):
        self.umwelt = umwelt
        self.zustand_dimension = self.umwelt.observe().shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1.0, shape=self.zustand_dimension, dtype=np.int16)
        self.action_space = gymnasium.spaces.MultiDiscrete([
            self.gedächtnis.getKapazität(),
            self.umwelt.umwelt.aktionsraum, 
        ])

    def reset(self, seed=None):
        self.gedächtnis = Gedächtnis(5)
        # self.umwelt = Umwelt()  # Entfernt, da falscher Aufruf
        # Stattdessen: Zustand der aktuellen Umwelt zurücksetzen
        if hasattr(self.umwelt, 'umwelt') and hasattr(self.umwelt.umwelt, 'restart'):
            self.umwelt.zustand_welt = self.umwelt.umwelt.restart()
        self.belohnungsErwartung = 0
        return self.gedächtnis.getBild(), {}

    def step(self, aktion):
        print("Aktion:" + str(aktion))
        # Aktion als Array sicherstellen
        aktion = np.array(aktion)
        if aktion.ndim == 0 or aktion.shape == ():
            raise ValueError(f"Aktion ist kein Array oder leer: {aktion}")
        # Aktion  0: Erinnerung auswählen
        # Aktion  1: Aktion auf die Umwelt
        self.gedächtnis.setSchreibZeiger(int(aktion[0]))
        gedanke = self.gedächtnis.getBild()
        beobachtung = self.umwelt.observe()
        # Typ- und Formkonsistenz sicherstellen
        gedanke = np.asarray(gedanke)
        beobachtung = np.asarray(beobachtung)
        if gedanke.shape != beobachtung.shape:
            raise ValueError(f"Shape mismatch: Gedanke {gedanke.shape}, Beobachtung {beobachtung.shape}")
        if gedanke.dtype != beobachtung.dtype:
            gedanke = gedanke.astype(np.float32)
            beobachtung = beobachtung.astype(np.float32)
        # Wahrnehmungsdissonanz und Reward nach Architektur
        wahrnehmungsdissonanz = gedanke - beobachtung
        self.wahrnehmungsdissonanzmatrix = wahrnehmungsdissonanz  # Für GUI-Zugriff
        r_delta = -np.sum(np.abs(wahrnehmungsdissonanz))
        # Evaluation: r_t = r_delta - 1/C * sum(r_G,t)
        mean_reward = self.gedächtnis.getMeanReward() if hasattr(self.gedächtnis, 'getMeanReward') else 0
        C = self.gedächtnis.getKapazität() if hasattr(self.gedächtnis, 'getKapazität') else 1
        eval_reward = r_delta - (1/C) * mean_reward
        reward = np.clip(eval_reward, -100, 100)
        # Aktion an Umwelt
        self.umwelt.act(int(aktion[1]))
        self.gedächtnis.update(Erinnerung(beobachtung, reward))
        self.umwelt_states.append(beobachtung)
        self.belohnungen.append(reward)
        print(f"Schreibzeiger: {self.gedächtnis.schreibZeiger}")
        print(f"Memory-Content: {[e.getBild() for e in self.gedächtnis.speicher]}")
        print(f"Umwelt-Zustand: {self.umwelt.zustand_welt}")
        print("Reward: " + str(reward))
        print("Perception difference: " + str(r_delta))
        return wahrnehmungsdissonanz, reward, False, False, {}

    def cross_entropy_reward(self, predicted, target):
        # Robust: negative L1-Distanz als Reward, damit keine NaN entstehen
        return -np.sum(np.abs(predicted - target))

    def render(self):
        print(self.gedächtnis)

if __name__ == '__main__':
    env = Agent()
    # Parallel environments
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(
        total_timesteps=50000,
        progress_bar=True
    )
    import matplotlib.pyplot as plt

    plt.plot(env.rewards)
    window_size = 500
    if len(env.rewards) >= window_size:
        filtered_rewards = np.convolve(env.rewards, np.ones(
            window_size) / window_size, mode='valid')
    else:
        filtered_rewards = env.rewards

    plt.plot(filtered_rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Rewards over time')
    plt.show()
    import matplotlib.pyplot as plt

    # Convert list of world states to a numpy array
    world_states_array = np.array(env.umwelt_states)

    # Plot each dimension of the world state separately
    for i in range(world_states_array.shape[1]):
        plt.plot(world_states_array[:, i], label=f'World State Dimension {i}')

    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('World State Value')
    plt.title('World States over Time')
    plt.show()
