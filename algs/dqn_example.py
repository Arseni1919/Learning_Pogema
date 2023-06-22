from stable_baselines3 import DQN
from pogema import GridConfig
import gym

env = gym.make("Pogema-v0", grid_config=GridConfig(size=8, density=0.3, num_agents=1, max_episode_steps=30))

dqn_agent = DQN('MlpPolicy', env, verbose=1)