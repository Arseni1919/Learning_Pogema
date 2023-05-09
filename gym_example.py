# import gymnasium as gym
# import pogema
#
# # This interface provides experience only for agent with id=0,
# # other agents will take random actions.
# env = gym.make("Pogema-v0")


from pogema import pogema_v0, GridConfig

# Create Pogema environment with PettingZoo interface
env = pogema_v0(GridConfig(integration="PettingZoo"))

env.reset()

for agent in range(100):
  env.step(env.sample_actions())

