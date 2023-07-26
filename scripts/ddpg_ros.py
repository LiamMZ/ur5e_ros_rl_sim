from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from ur5e_gym_env import UR5eEnv

print("Setting up training environment.")
# Instantiate your custom Gym environment
env = DummyVecEnv([lambda: UR5eEnv()])

print("Normalizing training environment.")
# Optional: normalize the environment
env = VecNormalize(env, norm_obs=True, norm_reward=True)

print("Setting up ddpg model.")
# Define the RL agent
model = DDPG("MlpPolicy", env, verbose=1)

print(env.buf_obs)
print("Training Model.")
# Train the agent
model.learn(total_timesteps=10000)

print("Finished Training Model.")
# Save the trained agent
model.save("dqn_ur5e_pick_and_place")

print("Saved Trained Model.")
# Don't forget to save your VecNormalize statistics when you save your agent:
env.save("dqn_ur5e_pick_and_place_norm")
