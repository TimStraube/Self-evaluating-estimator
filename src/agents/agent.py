import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.envs.arc_environment import ARCEnvironment
from src.erinnerung import Erinnerung
from src.gedächtnis import Gedächtnis
from src.umwelt import Umwelt
from src.envs.test import Test

class Agent(gymnasium.Env):
    def __init__(self):
        super().__init__()
        # Use original test environment
        self.umwelt = Umwelt(Test())

        # Memory system
        self.gedächtnis = Gedächtnis(5)

        self.zustand_dimension = self.umwelt.observe().shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1.0, shape=self.zustand_dimension, dtype=np.int16)
        self.action_space = gymnasium.spaces.MultiDiscrete([
            self.gedächtnis.getKapazität(),
            self.umwelt.umwelt.aktionsraum,
        ])

        # Gedächtnisfilter als interne Aktion des Agenten
        # Wertebereich: [0, 1], Summe der Filterwerte = 1
        # Größe: (1, Gedächtnis-Kapazität)
        self.gedächtnisfilter = np.zeros((1, self.gedächtnis.getKapazität()))
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
        super().reset(seed=seed)
        self.umwelt_states = []
        return self.umwelt.reset()[0]

    def step(self, aktion):
        # Aktionen des Agenten
        self.gedächtnisfilter = aktion[0]
        self.umweltaktion = aktion[1]

        # Gedächtnisvorhersage berechnen:
        # Alle gespeicherten Umweltzustände (shape: [kapazität, *zustand_dimension])
        umweltzustände = np.array(self.gedächtnis.getAlle())
        # Gedächtnisfilter transponieren (shape: [kapazität, 1])
        filter_t = np.array(self.gedächtnisfilter).reshape(-1, 1)
        # Gedächtnisvorhersage: gewichtete Summe der Umweltzustände
        # (Broadcasting über die Zustandsdimension)
        gedächtnisvorhersage = np.sum(umweltzustände * filter_t, axis=0)

        # Wahrnehmungsdissonanz berechnen (z.B. L2-Norm)
        aktueller_zustand = self.umwelt.observe()
        # Wahrnehmungsdissonanz: gedächtnisvorhersage mal FFT(aktueller_zustand)
        fft_zustand = np.fft.fft(aktueller_zustand)
        wahrnehmungsdissonanz = gedächtnisvorhersage * fft_zustand

        # Aktion in der Umwelt ausführen
        obs, belohnung, terminated, truncated, info = self.umwelt.step(self.umweltaktion)

        # Gedächtnis aktualisieren
        # Für jeden gespeicherten Zustand O_c im Gedächtnis:
        # O_c,t+1 = (1 - r_Δ) * a_c * O_c + r_Δ * (1 - a_c) * FFT(O_U,t)
        r_delta = 0.1  # Beispielwert, ggf. anpassen oder als Parameter übergeben
        alle_zustände = self.gedächtnis.getAlle()
        a_c = np.array(self.gedächtnisfilter).flatten()  # Form: [kapazität]
        fft_aktueller_zustand = np.fft.fft(aktueller_zustand)
        neue_zustände = []
        for i, O_c in enumerate(alle_zustände):
            O_c_neu = (1 - r_delta) * a_c[i] * O_c + r_delta * (1 - a_c[i]) * fft_aktueller_zustand
            neue_zustände.append(O_c_neu)
        self.gedächtnis.setAlle(neue_zustände)
        
        # Evaluation
        # Belohnung berechnen: belohnung = r_delta - gedächtnisfilter_c * r_c für alle C Kapazität
        # Annahme: r_c ist die Wahrnehmungsdissonanz für jeden Speicherplatz (Kapazität)
        # Da wahrnehmungsdissonanz ein Array sein kann, berechnen wir r_c für jede Kapazität
        # und summieren die Einzelbelohnungen auf
        belohnung = np.full(self.gedächtnis.getKapazität(), np.linalg.norm(wahrnehmungsdissonanz))

        gedächtnisfilter_c = np.array(self.gedächtnisfilter).flatten()
        belohnung = np.sum(r_delta - gedächtnisfilter_c * Gedächtnis.getReward())

        return wahrnehmungsdissonanz, belohnung, terminated, truncated, info

    def render(self):
        print(self.gedächtnis)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train an agent in ARC or original environment.")
    parser.add_argument('--env', type=str, choices=['arc', 'original'], default='original',
                        help='Environment to use: "arc" for ARCEnvironment, "original" for Test environment.')
    parser.add_argument('--memory-size', type=int, default=5,
                        help='Size of the memory (Gedächtnis).')
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Number of training timesteps.')
    parser.add_argument('--save-path', type=str, default='agent_model.zip',
                        help='Path to save the trained model.')
    args = parser.parse_args()

    # Initialize environment
    if args.env == 'arc':
        env = ARCEnvironment()
    else:
        env = Test()

    umwelt = Umwelt(env)
    agent = Agent()
    agent.set_umwelt(umwelt)
    agent.gedächtnis = Gedächtnis(args.memory_size)

    # Create vectorized environment
    vec_env = make_vec_env(lambda: agent, n_envs=1)

    # Initialize PPO model
    model = PPO('MlpPolicy', vec_env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=args.timesteps)

    # Save the trained model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    main()
