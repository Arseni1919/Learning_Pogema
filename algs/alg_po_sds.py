import random

import numpy.random
from globals import *
from plot_functions.plot_objects import Plotter
from algs.alg_temporal_a_star import build_graph_nodes, build_heuristic_for_multiple_targets
from algs.alg_temporal_a_star import h_func_creator, a_star


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
                    if (prev_node_1.x, prev_node_1.y, node_1.x, node_1.y) == (
                    node_2.x, node_2.y, prev_node_2.x, prev_node_2.y):
                        c_e_for_agent_list.append(
                            (agent_1, agent_2, prev_node_1.x, prev_node_1.y, node_1.x, node_1.y, t))
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
            perm_constr_dict[final_node.xy_name] = [min(perm_constr_dict[final_node.xy_name])]

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
    def __init__(self, num, obs_radius, small_iters, global_nodes, global_nodes_dict, po_field=False):
        self.num = num
        self.obs_radius = obs_radius
        self.small_iters = small_iters
        self.global_nodes = global_nodes
        self.global_nodes_dict = global_nodes_dict
        self.po_field = po_field
        self.name = f'agent_{num}'
        self.nei_list = []
        self.nei_dict = {}
        # obs
        self.agents_obs = None
        self.xy = None
        self.target_xy = None
        self.obstacles = None
        self.global_xy = None
        self.global_target_xy = None
        self.global_obstacles = None
        self.path = None
        self.a_star_iter_limit = 1e100
        self.arrived = False
        self.no_collisions = False
        # believed known data:
        self.beliefs = None  # other agents' paths
        self.b_nodes, self.b_nodes_dict = [], {}
        self.b_start = None
        self.b_goal = None
        self.b_near_goal_pool = None

    def agent_reset(self):
        self.nei_list = []
        self.nei_dict = {}
        self.beliefs = {i_small_iter: {} for i_small_iter in range(self.small_iters)}

    def add_nei(self, new_nei):
        if new_nei.name not in self.nei_dict:
            self.nei_list.append(new_nei)
            self.nei_dict[new_nei.name] = new_nei

    def update_b_nodes(self):
        if not self.arrived:
            if not self.po_field:
                self.b_start = self.global_nodes_dict[f'{self.global_xy[0]}_{self.global_xy[1]}']
                self.b_goal = self.global_nodes_dict[f'{self.global_target_xy[0]}_{self.global_target_xy[1]}']
                self.b_nodes = self.global_nodes
                self.b_nodes_dict = self.global_nodes_dict
            else:
                # set b_start
                self.b_start = self.global_nodes_dict[f'{self.global_xy[0]}_{self.global_xy[1]}']
                # set b_nodes and b_nodes_dict
                for node in self.global_nodes:
                    if node.xy_name not in self.b_nodes_dict:
                        x_dist = np.abs(self.b_start.x - node.x)
                        y_dist = np.abs(self.b_start.y - node.y)
                        if x_dist <= self.obs_radius and y_dist <= self.obs_radius:
                            self.b_nodes.append(node)
                            self.b_nodes_dict[node.xy_name] = node
                # set b_goal
                global_target_x, global_target_y = self.global_target_xy
                min_dist = np.abs(self.b_nodes[0].x - global_target_x) + np.abs(self.b_nodes[0].y - global_target_y)
                self.b_goal = self.b_nodes[0]
                for node in self.b_nodes[1:]:
                    curr_dist = np.abs(node.x - global_target_x) + np.abs(node.y - global_target_y)
                    if curr_dist <= min_dist:
                        min_dist = curr_dist
                        self.b_goal = node
                # set b_near_goal_pool
                self.b_near_goal_pool = []
                for node in self.b_nodes:
                    curr_dist = np.abs(node.x - self.b_goal.x) + np.abs(node.y - self.b_goal.y)
                    if curr_dist <= 2 * self.obs_radius:
                        self.b_near_goal_pool.append(node)

    def update(self, i_obs):
        self.agents_obs = i_obs['agents']

        self.xy = i_obs['xy']
        self.target_xy = i_obs['target_xy']
        self.obstacles = i_obs['obstacles']

        self.global_xy = i_obs['global_xy']
        self.global_target_xy = i_obs['global_target_xy']
        self.global_obstacles = i_obs['global_obstacles']

        if self.global_xy == self.global_target_xy:
            self.arrived = True

        # beliefs updates
        self.update_b_nodes()

    def plan(self, h_func, v_constr_dict=None, e_constr_dict=None, perm_constr_dict=None):
        if self.arrived:
            self.path = []
            return
        if self.po_field:
            if self.no_collisions and len(self.path) > 2:
                self.path = self.path[1:]
                return
        if v_constr_dict is None:
            v_constr_dict = {node.xy_name: [] for node in self.global_nodes}
            e_constr_dict = {node.xy_name: [] for node in self.global_nodes}
            perm_constr_dict = {node.xy_name: [] for node in self.global_nodes}
        new_path = None
        while new_path is None:
            new_path, info = a_star(start=self.b_start, goal=self.b_goal, nodes=self.b_nodes, h_func=h_func,
                                    v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                    perm_constr_dict=perm_constr_dict,
                                    nodes_dict=self.b_nodes_dict,
                                    )
            if new_path is not None:
                self.path = new_path
                return
            else:
                if not self.po_field:
                    return
                self.b_goal = np.random.choice(self.b_near_goal_pool, 1)[0]
                print(f'\n[{self.name}] New Goal: {self.b_goal.xy_name} out of {len(self.b_nodes)}')
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

    def recalc_path(self, h_func, i_iter):
        if self.arrived:
            return True
        c_v_list = c_v_check_for_agent(self.name, self.path, self.beliefs[i_iter])
        c_e_list = c_e_check_for_agent(self.name, self.path, self.beliefs[i_iter])
        if len(self.path) == 0:
            raise RuntimeError('len(self.path) == 0')

        if len(c_v_list) == 0 and len(c_e_list) == 0:
            self.no_collisions = True
            return self.no_collisions
        self.no_collisions = False

        to_change = self.decision_bool(i_iter)
        if to_change:
            v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(self.global_nodes, self.beliefs[i_iter])
            self.plan(h_func, v_constr_dict, e_constr_dict, perm_constr_dict)
            # new_path, info = a_star(start=self.b_start, goal=self.b_goal, nodes=self.b_nodes, h_func=h_func,
            #                         v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
            #                         perm_constr_dict=perm_constr_dict,
            #                         nodes_dict=self.b_nodes_dict)
            # if new_path is not None:
            #     self.path = new_path
        return self.no_collisions

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
        if len(self.path) == 1:
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
        return action


def update_all_agents_their_obs(agents, obs):
    for i_obs_index, i_obs in enumerate(obs):
        agent = agents[i_obs_index]
        agent.update(i_obs)


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


def get_actions(agents, obs, small_iters, h_func):
    # update obs
    update_all_agents_their_obs(agents, obs)
    # reset neighbours
    reset_agents(agents)
    set_all_agents_their_nei(agents)

    # calc initial path
    for agent in agents:
        agent.plan(h_func)
    for i_iter in range(small_iters):
        # exchange paths with neighbors
        for agent in agents:
            agent.send_path_to_nei(i_iter)
        # detect collision + recalculate path
        no_collision_list = []
        for agent in agents:
            no_collision = agent.recalc_path(h_func, i_iter)
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


def run_po_sds(env, num_agents, max_episode_steps, obs_radius, plotter, *args, **kwargs):
    step_counter = 0
    soc_counter = 0
    succeeded = True

    obs = env.reset()
    small_iters = kwargs['small_iters']
    po_field = kwargs['po_field']
    plot_per = kwargs['plot_per']

    # prebuild map
    img_np = obs[0]['global_obstacles']
    img_np = 1 - img_np
    map_dim = img_np.shape
    nodes, nodes_dict = build_graph_nodes(img_np=img_np, show_map=False)

    # create agents
    agents = []
    for i in range(num_agents):
        agent = PoSdsAgent(num=i, obs_radius=obs_radius, small_iters=small_iters,
                           global_nodes=nodes, global_nodes_dict=nodes_dict, po_field=po_field)
        agents.append(agent)
    update_all_agents_their_obs(agents, obs)

    # heuristic
    node_goals = [nodes_dict[f'{agent.global_xy[0]}_{agent.global_xy[1]}'] for agent in agents]
    h_dict = build_heuristic_for_multiple_targets(node_goals, nodes, map_dim)
    h_func = h_func_creator(h_dict)

    # while True:
    for i in range(max_episode_steps):
        step_counter += 1
        actions = get_actions(agents, obs, small_iters, h_func)
        obs, reward, terminated, info = env.step(actions)
        # env.render()
        print(f'\r[PO-SDS] step: {i}', end='')
        if plotter:
            if i % plot_per == 0:
                plotter.render(info={
                    'i_step': i,
                    'obs': obs,
                    'num_agents': num_agents,
                    'agents': agents,
                })
        if all(terminated):
            break
        else:
            soc_counter += sum(terminated)
        if step_counter >= max_episode_steps - 1:
            succeeded = False

    # env.save_animation("render.svg")
    # env.save_animation("render_agent_0.svg", AnimationConfig(egocentric_idx=0))

    stat_info = {'steps': step_counter, 'soc': soc_counter, 'succeeded': succeeded}
    return stat_info


def main():
    num_agents = 5
    max_episode_steps = 1000
    small_iters = 3
    obs_radius = 3
    seed = 59
    # seed = random.randint(0, 100)
    print(f'[SEED]: {seed}')
    po_field = True
    # po_field = False

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

    plotter = Plotter()

    run_po_sds(env, num_agents, max_episode_steps, obs_radius, plotter, small_iters=small_iters, po_field=po_field,
               plot_per=1)


if __name__ == '__main__':
    main()
