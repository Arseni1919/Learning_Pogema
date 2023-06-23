import numpy.random
from globals import *
from plot_functions.plot_objects import Plotter
from alg_temporal_a_star import build_graph_nodes, build_heuristic_for_multiple_targets
from alg_temporal_a_star import h_func_creator, a_star


def c_v_check_for_agent(agent_1: str, path_1, results, immediate=False):
    """
    c_v_for_agent_list: (agents name 1, agents name 2, x, y, t)
    """
    if type(agent_1) is not str:
        raise RuntimeError('type(agent_1) is not str')
    c_v_for_agent_list = []
    if len(path_1) < 1:
        return c_v_for_agent_list
    for agent_2, path_2 in results.items():
        # if type(agent_2) is not str:
        #     raise RuntimeError('type(agent_2) is not str')
        if agent_2 != agent_1:
            for t in range(max(len(path_1), len(path_2))):
                node_1 = path_1[min(t, len(path_1) - 1)]
                node_2 = path_2[min(t, len(path_2) - 1)]
                if (node_1.x, node_1.y) == (node_2.x, node_2.y):
                    c_v_for_agent_list.append((agent_1, agent_2, node_1.x, node_1.y, t))
                    if immediate:
                        return c_v_for_agent_list
    return c_v_for_agent_list


def c_e_check_for_agent(agent_1: str, path_1, results, immediate=False):
    """
    c_e_check_for_agent: (agents name 1, agents name 2, x, y, x, y, t)
    """
    if type(agent_1) is not str:
        raise RuntimeError('type(agent_1) is not str')
    c_e_for_agent_list = []
    if len(path_1) <= 1:
        return c_e_for_agent_list
    for agent_2, path_2 in results.items():
        if agent_2 != agent_1:
            if len(path_2) > 1:
                prev_node_1 = path_1[0]
                prev_node_2 = path_2[0]
                for t in range(1, min(len(path_1), len(path_2))):
                    node_1 = path_1[t]
                    node_2 = path_2[t]
                    if (prev_node_1.x, prev_node_1.y, node_1.x, node_1.y) == (node_2.x, node_2.y, prev_node_2.x, prev_node_2.y):
                        c_e_for_agent_list.append((agent_1, agent_2, prev_node_1.x, prev_node_1.y, node_1.x, node_1.y, t))
                        if immediate:
                            return c_e_for_agent_list
                    prev_node_1 = node_1
                    prev_node_2 = node_2
    return c_e_for_agent_list


def build_constraints(nodes, other_paths):
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict = {node.xy_name: [] for node in nodes}

    for agent_name, path in other_paths.items():
        if len(path) > 0:

            # correct times
            for indx_node, node in enumerate(path):
                node.t = indx_node

            final_node = path[-1]
            perm_constr_dict[final_node.xy_name].append(final_node.t)

            prev_node = path[0]
            for node in path:
                # vertex
                v_constr_dict[f'{node.x}_{node.y}'].append(node.t)
                # edge
                if prev_node.xy_name != node.xy_name:
                    e_constr_dict[f'{prev_node.x}_{prev_node.y}'].append((node.x, node.y, node.t))
                prev_node = node
    return v_constr_dict, e_constr_dict, perm_constr_dict


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
        self.a_star_iter_limit = 1e100
        self.arrived = False

    def agent_reset(self):
        self.nei_list = []
        self.nei_dict = {}
        self.beliefs = {i_small_iter: {} for i_small_iter in range(self.small_iters)}

    def add_nei(self, new_nei):
        if new_nei.name not in self.nei_dict:
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
        if self.global_xy == self.global_target_xy:
            self.arrived = True

    def plan(self, nodes, nodes_dict, h_func):
        if self.arrived:
            self.path = []
            return
        node_start = nodes_dict[f'{self.global_xy[0]}_{self.global_xy[1]}']
        node_goal = nodes_dict[f'{self.global_target_xy[0]}_{self.global_target_xy[1]}']
        v_constr_dict = {node.xy_name: [] for node in nodes}
        perm_constr_dict = {node.xy_name: [] for node in nodes}
        self.path, info = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func,
                                 v_constr_dict=v_constr_dict, perm_constr_dict=perm_constr_dict,
                                 plotter=None, middle_plot=True, nodes_dict=nodes_dict,
                                 )
        if self.path is None:
            raise RuntimeError('self.path is None')

    def send_path_to_nei(self, i_iter):
        for nei in self.nei_list:
            nei.beliefs[i_iter][self.name] = self.path

    def decision_bool(self, i_iter):
        # A MORE SMART VERSION
        path_lngths = [len(nei_path) for nei_name, nei_path in self.beliefs[i_iter].items()]
        max_n = max(path_lngths)
        min_n = min(path_lngths)
        # priority on smaller paths
        if len(self.path) > max_n and random.random() < 0.9:
            return True
        elif len(self.path) < min_n and random.random() < 0.1:
            return True
        else:
            path_lngths.append(len(self.path))
            path_lngths.sort()
            my_order = path_lngths.index(len(self.path))
            my_alpha = 0.1 + 0.8 * (my_order / len(path_lngths))
            if random.random() < my_alpha:
                return True
        return False

    def recalc_path(self, nodes, nodes_dict, h_func, i_iter):
        if self.arrived:
            return True
        c_v_list = c_v_check_for_agent(self.name, self.path, self.beliefs[i_iter])
        c_e_list = c_e_check_for_agent(self.name, self.path, self.beliefs[i_iter])
        if len(self.path) == 0:
            raise RuntimeError('len(self.path) == 0')
        if len(c_v_list) == 0 and len(c_e_list) == 0:
            return True

        to_change = self.decision_bool(i_iter)
        if to_change:
            node_start = nodes_dict[f'{self.global_xy[0]}_{self.global_xy[1]}']
            node_goal = nodes_dict[f'{self.global_target_xy[0]}_{self.global_target_xy[1]}']
            v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(nodes, self.beliefs[i_iter])
            # perm_constr_dict = {node.xy_name: [] for node in nodes}  # disappear at the end
            new_path, info = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func,
                                    v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                    perm_constr_dict=perm_constr_dict,
                                    nodes_dict=nodes_dict, iter_limit=self.a_star_iter_limit
                                    )
            if new_path is not None:
                self.path = new_path
        return False

    def get_next_action(self):
        """
        ACTIONS:
        0 - idle
        1 - left (down in numbers), 2 - right (up in numbers), 3 - down (down in numbers), 4 - up (up in numbers)
        """
        # action = numpy.random.randint(0, 5)
        # if arrived
        if self.arrived:
            return 0
        next_pos_node = self.path[1]
        curr_x, curr_y = self.global_xy
        next_x, next_y = next_pos_node.x, next_pos_node.y
        action = 0
        # LEFT
        if next_x < curr_x:
            action = 1
        # RIGHT
        if next_x > curr_x:
            action = 2
        # DOWN
        if next_y < curr_y:
            action = 3
        # UP
        if next_y > curr_y:
            action = 4
        # action = 2
        # PRINT:
        if self.num == 0:
            print(f"\n{self.name}'s action: {action}")
            print(f"{self.name}'s pos: {self.global_xy}")
        return action


def update_all_agents_their_obs(agents, obs):
    for i_obs_index, i_obs in enumerate(obs):
        agents[i_obs_index].update(i_obs)


def reset_agents(agents):
    for agent in agents:
        agent.agent_reset()


def set_all_agents_their_nei(agents):
    for curr_agent in agents:
        if not curr_agent.arrived:
            for agent_2 in agents:
                if curr_agent.name != agent_2.name:
                    if not agent_2.arrived:
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
        no_collision_list = []
        for agent in agents:
            no_collision = agent.recalc_path(nodes, nodes_dict, h_func, i_iter)
            no_collision_list.append(no_collision)
        # termination condition
        if all(no_collision_list):
            break

    # decide on the next action
    actions = []
    for agent in agents:
        actions.append(agent.get_next_action())
        # actions.append(0)
    # not_arrived_agents = [agent for agent in agents if not agent.arrived]
    # print(not_arrived_agents)
    return actions


def main():
    num_agents = 20
    max_episode_steps = 1000
    small_iters = 3
    obs_radius = 3
    plotter = Plotter()
    seed = 10
    # seed = random.randint(0, 100)

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
    obs = env.reset()

    # prebuild map
    img_np = obs[0]['global_obstacles']
    img_np = 1 - img_np
    map_dim = img_np.shape
    nodes, nodes_dict = build_graph_nodes(img_np=img_np, show_map=False)

    # create agents
    agents = []
    for i in range(num_agents):
        agent = PoSdsAgent(num=i, obs_radius=obs_radius, small_iters=small_iters)
        agents.append(agent)
    update_all_agents_their_obs(agents, obs)

    # heuristic
    node_goals = [nodes_dict[f'{agent.global_xy[0]}_{agent.global_xy[1]}'] for agent in agents]
    h_dict = build_heuristic_for_multiple_targets(node_goals, nodes, map_dim)
    h_func = h_func_creator(h_dict)

    # while True:
    for i in range(max_episode_steps):
        actions = get_actions(agents, obs, small_iters, nodes, nodes_dict, h_func)
        obs, reward, terminated, info = env.step(actions)
        # env.render()
        print(f'\n[PO-SDS] step: {i}')
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
