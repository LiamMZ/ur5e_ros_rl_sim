import rospy
from stable_baselines3 import PPO
import gym

class PPOAgent:
    def __init__(self, env, total_timesteps=10000):
        self.env = env
        self.total_timesteps = total_timesteps
        self.model = PPO('MlpPolicy', env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

    def predict(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
