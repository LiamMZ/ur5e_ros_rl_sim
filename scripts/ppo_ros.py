from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import rospy
from ur5e_gym_env import UR5eEnv


# Instantiate your custom Gym environment
env = DummyVecEnv([lambda: UR5eEnv()])

# Optional: normalize the environment
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Define the RL agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained agent
model.save("ppo_ur5e_pick_and_place")

# Don't forget to save your VecNormalize statistics when you save your agent:
env.save("ppo_ur5e_pick_and_place_norm")
