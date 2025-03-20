import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from erinnerung import Erinnerung
from gedächtnis import Gedächtnis
from umwelt import Umwelt


class Agent(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.gedächtnis = Gedächtnis(250)
        self.umwelt = Umwelt()
        
        self.state_dim = self.umwelt.observe().shape
        print(self.state_dim)
        # Combine both into a single observation as a Dict space
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1.0, shape=self.state_dim, dtype=np.int16)
        # Action on the world and thougth pointer
        self.action_space = gymnasium.spaces.MultiDiscrete([
            self.gedächtnis.getKapazität(),
            2,
            2,
            81
        ])
        self.belohnungsErwartung = 0
        self.rewards = []
        self.umwelt_states = []

    def reset(self, seed=None):
        self.gedächtnis = Gedächtnis(250)
        self.umwelt = Umwelt()
        self.belohnungsErwartung = 0
        return self.gedächtnis.getBild(), {}

    def step(self, aktion):
        print("Aktion:" + str(aktion))
        # Aktion  0: Erinnerung auswählen
        # Aktion  1: Selbstbewusstsein der Aktion zuweisen
        # Aktion +2: Aktion auf die Umwelt
        
        # Der Agent wählt eine Erinnerung aus dem Gedächtnis
        self.gedächtnis.setSchreibZeiger(aktion[0])
        gedanke = self.gedächtnis.getBild()

        # Eine Beobachtung wird aus der Umwelt geladen
        beobachtung = self.umwelt.observe()
        
        # Der Agent vergleicht den Gedanken mit der Beobachtung
        wahrnehmungsdissonanz = gedanke - beobachtung
        wahrnehmungsdissonanzwert = self.cross_entropy_reward(gedanke, beobachtung)
        
        # print("Value: " + str(value))
        # print("Average reward: " + str(self.belohnungsErwartung))
        # No activation function on the reward gain
        # reward = value - self.belohnungsErwartung
        # Logistic    # The agent is rewarded based on the familarity between the state of mind and the world input
        reward = np.exp(wahrnehmungsdissonanzwert + self.gedächtnis.getMeanReward())

        # Der Agent führt nun die Aktion in der Welt aus
        # print("aktion" + aktion[2:])
        self.umwelt.act(aktion[2:])

        # Update the memory with the new observation
        self.gedächtnis.update(Erinnerung(beobachtung, reward))
        self.umwelt_states.append(beobachtung)

        # self.render()
        # print("Reward: " + str(reward))

        self.rewards.append(reward)

        print("Reward: " + str(reward))

        print("Perception difference: " + str(wahrnehmungsdissonanzwert))

        return wahrnehmungsdissonanz, reward, False, False, {}

    def cross_entropy_reward(self, predicted, target):
        # Clipping, um log(0) zu vermeiden
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        
        # Kreuzentropie
        cross_entropy = -np.sum(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
        
        # Umkehren, da niedrigere Kreuzentropie besser ist
        return -cross_entropy

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
