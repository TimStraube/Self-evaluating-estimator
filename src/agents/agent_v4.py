import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.envs.arc_environment import ARCEnvironment
from src.erinnerung import Erinnerung
from src.gedächtnis import Gedächtnis
from src.umwelt import Umwelt

class Agent(gymnasium.Env):
    def __init__(self, use_arc=False):
        super().__init__()
        self.use_arc = use_arc

        self.env = make_vec_env(ARCEnvironment if use_arc else lambda: Umwelt(...), n_envs=1)

        # Memory system
        self.gedächtnis = Gedächtnis(5)

        # Determine observation space based on environment
        if self.use_arc:
            self.observation_space = self.arc_env.observation_space
            self.action_space = self.arc_env.action_space
            self.zustand_dimension = (2, 30, 30)  # ARC observation: input + prediction channels
        else:
            self.zustand_dimension = self.umwelt.observe().shape
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=1.0, shape=self.zustand_dimension, dtype=np.int16)
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
        if self.use_arc:
            # Reset ARC environment
            observation, info = self.arc_env.reset(seed=seed)
            self.belohnungsErwartung = 0
            return observation, info
        else:
            # Original reset logic
            self.gedächtnis = Gedächtnis(10)
            if hasattr(self.umwelt, 'umwelt') and hasattr(self.umwelt.umwelt, 'restart'):
                self.umwelt.zustand_welt = self.umwelt.umwelt.restart()
            self.belohnungsErwartung = 0
            return self.gedächtnis.getBild(), {}

    def step(self, aktion):
        if self.use_arc:
            # For ARC environment, action is [memory_index, grid_position]
            # But we need to handle it in the SEE framework
            aktion = np.array(aktion)
            if aktion.ndim == 0 or aktion.shape == ():
                # If single action, assume it's just grid position, use memory slot 0
                aktion = np.array([0, aktion])
            
            memory_slot = int(aktion[0])
            grid_action = int(aktion[1])
            
            # Delegate to ARC environment with the full action tuple
            observation, ext_reward, terminated, truncated, info = self.arc_env.step((memory_slot, grid_action))
            # Wahrnehmungsdissonanz: Differenz zwischen Vorhersage und Ziel
            prediction = self.arc_env.current_prediction
            target = self.arc_env.current_output
            r_delta = -np.sum(np.abs(prediction - target))
            # Mittelwert der Rewards im Gedächtnis
            mean_reward = self.gedächtnis.getMeanReward() if hasattr(self.gedächtnis, 'getMeanReward') else 0
            C = self.gedächtnis.getKapazität() if hasattr(self.gedächtnis, 'getKapazität') else 1
            # SEE-Prinzip: Reward = r_delta - 1/C * mean_reward + externer Reward
            reward = r_delta - (1/C) * mean_reward + ext_reward
            self.belohnungsErwartung = reward
            # Gedächtnis aktualisieren
            self.gedächtnis.update(Erinnerung(prediction, reward))
            self.belohnungen.append(reward)
            return observation, reward, terminated, truncated, info
        else:
            # Original SEE step logic
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

def main():
    """Main training function for SEE agent"""
    import argparse
    import matplotlib
    import os
    import json
    matplotlib.use('Agg')  # Use non-interactive backend
    
    parser = argparse.ArgumentParser(description='Train SEE agent')
    parser.add_argument('--arc', action='store_true', help='Use ARC environment instead of original')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--save', type=str, help='Path to save trained model')
    
    args = parser.parse_args()
    
    print(f"Training SEE agent with {'ARC' if args.arc else 'original'} environment...")
    
    # Create agent
    env = Agent(use_arc=args.arc)
    
    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    
    # Train the model
    print(f"Starting training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    except ImportError:
        print("Progress bar not available, training without progress bar...")
        model.learn(total_timesteps=args.timesteps)
    
    # Save model if path provided
    if args.save:
        model.save(args.save)
        print(f"Model saved to {args.save}")
    
    # Plot rewards if available
    if hasattr(env, 'rewards') and env.rewards:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(env.rewards)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Raw Rewards')
        
        # Plot smoothed rewards
        plt.subplot(1, 2, 2)
        window_size = min(500, len(env.rewards))
        if len(env.rewards) >= window_size:
            filtered_rewards = np.convolve(env.rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(filtered_rewards)
        else:
            plt.plot(env.rewards)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title(f'Smoothed Rewards (window={window_size})')
        
        plt.tight_layout()
        plt.savefig('training_rewards.png', dpi=150, bbox_inches='tight')
        print("Training plot saved as 'training_rewards.png'")
        print("Training completed successfully!")
    
    # Evaluation mit Test-Datensatz (nur für ARC)
    if args.arc:
        print("Starte Evaluation mit Test-Datensatz...")
        from src.envs.arc_environment import ARCEnvironment
        test_data_path = "/home/timstraube/Programme/SEE/data/arc-prize-2025"
        test_env = ARCEnvironment(data_path=test_data_path)
        # Testdaten laden
        test_env.tasks = {}
        test_file = os.path.join(test_data_path, "arc-agi_test_challenges.json")
        try:
            with open(test_file, 'r') as f:
                test_env.tasks = json.load(f)
                test_env.task_ids = list(test_env.tasks.keys())[:10]
                print(f"Geladene Testaufgaben: {len(test_env.task_ids)}")
        except FileNotFoundError:
            print(f"Testdatei nicht gefunden: {test_file}")
            test_env.tasks = test_env._create_dummy_task()
            test_env.task_ids = list(test_env.tasks.keys())

        # Evaluation: Führe für jede Testaufgabe eine Episode aus
        num_episodes = len(test_env.task_ids)
        num_success = 0
        total_reward = 0.0
        for task_id in test_env.task_ids:
            test_env.current_task_id = task_id
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = test_env.step(action)
                episode_reward += reward
            # Erfolg, wenn die Lösung am Ende exakt stimmt
            if test_env._is_correct_solution():
                num_success += 1
            print(f"Testaufgabe {task_id}: {'Erfolgreich' if test_env._is_correct_solution() else 'Fehlgeschlagen'}, Reward = {episode_reward:.3f}")
            total_reward += episode_reward
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        print(f"Erfolgreich gelöste Aufgaben: {num_success} von {num_episodes}")
        print(f"Accuracy: {num_success / num_episodes:.2%}")
        print(f"Durchschnittlicher Reward auf Test-Datensatz: {avg_reward:.3f}")

    return model, env


if __name__ == '__main__':
    main()
