import numpy.random

from globals import *
from plot_functions.plot_objects import Plotter


class PoSdsAgent:
    def __init__(self, num, obs_radius):
        self.num = num
        self.obs_radius = obs_radius
        self.name = f'agent_{num}'
        self.nei_list = []
        self.nei_dict = {}
        # obs
        self.obstacles = None
        self.agents_obs = None
        self.xy = None
        self.target_xy = None
        self.global_obstacles = None
        self.global_xy = None
        self.global_target_xy = None

    def reset_all_nei(self):
        self.nei_list = []
        self.nei_dict = {}

    def add_nei(self, new_nei):
        self.nei_list.append(new_nei)
        self.nei_dict[new_nei.name] = new_nei

    def update(self, i_obs):
        self.obstacles = i_obs['obstacles']
        self.agents_obs = i_obs['agents']
        self.xy = i_obs['xy']
        self.target_xy = i_obs['target_xy']
        self.global_obstacles = i_obs['global_obstacles']
        self.global_xy = i_obs['global_xy']
        self.global_target_xy = i_obs['global_target_xy']


def update_all_agents_their_obs(agents, obs):
    for i_obs_index, i_obs in enumerate(obs):
        agents[i_obs_index].update(i_obs)


def remove_all_agents_their_nei(agents):
    for agent in agents:
        agent.reset_all_nei()


def set_all_agents_their_nei(agents):
    for curr_agent in agents:
        for agent_2 in agents:
            if curr_agent.name != agent_2.name:
                x_dist = np.abs(curr_agent.global_xy[0] - agent_2.global_xy[0])
                y_dist = np.abs(curr_agent.global_xy[1] - agent_2.global_xy[1])
                if x_dist < curr_agent.obs_radius and y_dist < curr_agent.obs_radius:
                    curr_agent.add_nei(agent_2)
        # print(f'{curr_agent.name} nei: {[nei.name for nei in curr_agent.nei_list]}')


def get_actions(agents, obs, small_iters):
    # update obs
    update_all_agents_their_obs(agents, obs)
    # reset neighbours
    remove_all_agents_their_nei(agents)
    set_all_agents_their_nei(agents)
    # calc initial path
    pass
    for i_iter in range(small_iters):
        # exchange paths with neighbors
        pass
        # detect collision + recalculate path
        pass
        # termination condition
        pass

    # decide on the next action
    actions = []
    for i_obs_index, i_obs in enumerate(obs):
        # action = agents[i_obs_index].act(i_obs)
        action = numpy.random.randint(0, 5)
        actions.append(action)
    return actions


def main():
    num_agents = 5
    max_episode_steps = 1000
    small_iters = 3
    plotter = Plotter()
    # seed = 10
    seed = random.randint(0, 100)
    obs_radius = 3

    # Define random configuration
    grid_config = GridConfig(
        num_agents=num_agents,  # number of agents
        size=30,  # size of the grid
        density=0.2,  # obstacle density
        seed=seed,  # set to None for random
        # obstacles, agents and targets
        # positions at each reset
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type='MAPF'
    )
    # env = pogema_v0(grid_config=Hard8x8())
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env)

    # create agents
    agents = []
    for i in range(num_agents):
        agent = PoSdsAgent(num=i, obs_radius=obs_radius)
        agents.append(agent)

    obs = env.reset()

    # while True:
    for i in range(max_episode_steps):
        actions = get_actions(agents, obs, small_iters)
        obs, reward, terminated, info = env.step(actions)
        # env.render()
        print(f'iter: {i}')
        plotter.render(info={
            'i_step': i,
            'obs': obs,
            'num_agents': num_agents
        })
        if all(terminated):
            break

    # env.save_animation("render.svg")
    # env.save_animation("render_agent_0.svg", AnimationConfig(egocentric_idx=0))


if __name__ == '__main__':
    main()