import numpy.random
from globals import *
from plot_functions.plot_objects import Plotter
from alg_temporal_a_star import build_graph_nodes, build_heuristic_for_multiple_targets
from alg_temporal_a_star import h_func_creator, a_star


class PoSdsAgent:
    def __init__(self, num, obs_radius, small_iters):
        self.num = num
        self.obs_radius = obs_radius
        self.small_iters = small_iters
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
        self.path = None
        self.beliefs = None

    def agent_reset(self):
        self.nei_list = []
        self.nei_dict = {}
        self.beliefs = {i_small_iter: [] for i_small_iter in range(self.small_iters)}

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

    def plan(self, nodes, nodes_dict, h_func):
        node_start = nodes_dict[f'{self.global_xy[0]}_{self.global_xy[1]}']
        node_goal = nodes_dict[f'{self.global_target_xy[0]}_{self.global_target_xy[1]}']
        v_constr_dict = {node.xy_name: [] for node in nodes}
        perm_constr_dict = {node.xy_name: [] for node in nodes}
        self.path, info = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func,
                                 v_constr_dict=v_constr_dict, perm_constr_dict=perm_constr_dict,
                                 plotter=None, middle_plot=True, nodes_dict=nodes_dict,
                                 )

    def send_path_to_nei(self, i_iter):
        for nei in self.nei_list:
            nei.beliefs[i_iter].append(self.path)

    def recalc_path(self):
        pass


def update_all_agents_their_obs(agents, obs):
    for i_obs_index, i_obs in enumerate(obs):
        agents[i_obs_index].update(i_obs)


def reset_agents(agents):
    for agent in agents:
        agent.agent_reset()


def set_all_agents_their_nei(agents):
    for curr_agent in agents:
        for agent_2 in agents:
            if curr_agent.name != agent_2.name:
                x_dist = np.abs(curr_agent.global_xy[0] - agent_2.global_xy[0])
                y_dist = np.abs(curr_agent.global_xy[1] - agent_2.global_xy[1])
                if x_dist < curr_agent.obs_radius and y_dist < curr_agent.obs_radius:
                    curr_agent.add_nei(agent_2)
        # print(f'{curr_agent.name} nei: {[nei.name for nei in curr_agent.nei_list]}')


def get_actions(agents, obs, small_iters, nodes, nodes_dict, h_func):
    # update obs
    update_all_agents_their_obs(agents, obs)
    # reset neighbours
    reset_agents(agents)
    set_all_agents_their_nei(agents)

    # calc initial path
    for agent in agents:
        agent.plan(nodes, nodes_dict, h_func)
    for i_iter in range(small_iters):
        # exchange paths with neighbors
        for agent in agents:
            agent.send_path_to_nei(i_iter)
        # detect collision + recalculate path
        for agent in agents:
            agent.recalc_path()
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
        agent = PoSdsAgent(num=i, obs_radius=obs_radius, small_iters=small_iters)
        agents.append(agent)

    obs = env.reset()
    update_all_agents_their_obs(agents, obs)
    # prebuild map
    img_np = obs[0]['global_obstacles']
    img_np = 1 - img_np
    map_dim = img_np.shape
    nodes, nodes_dict = build_graph_nodes(img_np=img_np, show_map=False)
    # heuristic
    node_goals = [nodes_dict[f'{agent.global_xy[0]}_{agent.global_xy[1]}'] for agent in agents]
    h_dict = build_heuristic_for_multiple_targets(node_goals, nodes, map_dim)
    h_func = h_func_creator(h_dict)

    # while True:
    for i in range(max_episode_steps):
        actions = get_actions(agents, obs, small_iters, nodes, nodes_dict, h_func)
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
