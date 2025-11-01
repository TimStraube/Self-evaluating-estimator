import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.agents.see import SEE

if __name__ == "__main__":
    env = SEE(capacity=5)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    obs, _ = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(f"Reward: {reward}")
        if done:
            obs, _ = env.reset()
