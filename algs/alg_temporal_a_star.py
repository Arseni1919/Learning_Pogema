from simulator_objects import Node, ListNodes
from globals import *


# funct_graph

def dist_heuristic(from_node, to_node):
    return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)


def h_func_creator(h_dict):
    def h_func(from_node, to_node):
        if to_node.xy_name in h_dict:
            h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
            if h_value > 0:
                return h_value
        return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
        # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)
    return h_func


def h_get_node(successor_xy_name, node_current, nodes_dict):
    if node_current.xy_name == successor_xy_name:
        return None
    return nodes_dict[successor_xy_name]


def build_heuristic_for_one_target(target_node, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    # print('Started to build heuristic...')
    copy_nodes = nodes
    nodes_dict = {node.xy_name: node for node in copy_nodes}
    target_name = target_node.xy_name
    target_node = nodes_dict[target_name]
    # target_node = [node for node in copy_nodes if node.xy_name == target_node.xy_name][0]
    # open_list = []
    # close_list = []
    open_nodes = ListNodes(target_name=target_node.xy_name)
    closed_nodes = ListNodes(target_name=target_node.xy_name)
    # open_list.append(target_node)
    open_nodes.add(target_node)
    iteration = 0
    # while len(open_list) > 0:
    while len(open_nodes) > 0:
        iteration += 1
        # node_current = get_node_from_open(open_list, target_name)
        node_current = open_nodes.pop()
        # if node_current.xy_name == '30_12':
        #     print()
        for successor_xy_name in node_current.neighbours:
            node_successor = h_get_node(successor_xy_name, node_current, nodes_dict)
            if node_successor:
                successor_current_g = node_current.g_dict[target_name] + 1  # h(now, next)

                # INSIDE OPEN LIST
                if node_successor.xy_name in open_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    open_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # INSIDE CLOSED LIST
                elif node_successor.xy_name in closed_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    closed_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # NOT IN CLOSED AND NOT IN OPEN LISTS
                else:
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # node_successor.g_dict[target_name] = successor_current_g
                # node_successor.parent = node_current

        # open_nodes.remove(node_current, target_name=target_node.xy_name)
        closed_nodes.add(node_current)

        if plotter and middle_plot and iteration % 1000 == 0:
            plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                               closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)
        if iteration % 100 == 0:
            print(f'\riter: {iteration}', end='')

    if plotter and middle_plot:
        plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                           closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)

    h_table = np.zeros(map_dim)
    for node in copy_nodes:
        h_table[node.x, node.y] = node.g_dict[target_name]
    # h_dict = {target_node.xy_name: h_table}
    # print(f'\rFinished to build heuristic at iter {iteration}.')
    return h_table


def build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    print('Started to build heuristic...')
    h_dict = {}
    _ = [node.reset(target_nodes) for node in nodes]
    iteration = 0
    for node in target_nodes:
        h_table = build_heuristic_for_one_target(node, nodes, map_dim, to_save, plotter, middle_plot)
        h_dict[node.xy_name] = h_table

        print(f'\nFinished to build heuristic for node {iteration}.')
        iteration += 1
    return h_dict


def make_neighbours(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)
        for node_2 in nodes:
            if node_1.xy_name != node_2.xy_name:
                dist = math.sqrt((node_1.x - node_2.x)**2 + (node_1.y - node_2.y)**2)
                if dist == 1.0:
                    node_1.neighbours.append(node_2.xy_name)


def distance_nodes(node1, node2, h_func: dict = None):
    if h_func is None:
        # print('regular distance')
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    else:
        heuristic_dist = h_func[node1.x][node1.y][node2.x][node2.y]
        # direct_dist = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return heuristic_dist


def set_nei(name_1, name_2, nodes_dict):
    if name_1 in nodes_dict and name_2 in nodes_dict and name_1 != name_2:
        node1 = nodes_dict[name_1]
        node2 = nodes_dict[name_2]
        dist = distance_nodes(node1, node2)
        if dist == 1:
            node1.neighbours.append(node2.xy_name)
            node2.neighbours.append(node1.xy_name)


def make_self_neighbour(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)


def build_graph_from_np(img_np, show_map=False):
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    # make_neighbours(nodes)

    name_1, name_2 = '', ''
    for i_x in range(x_size):
        for i_y in range(y_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2

    print('finished rows')

    for i_y in range(y_size):
        for i_x in range(x_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2
    make_self_neighbour(nodes)
    print('finished columns')

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict


def build_graph_nodes(img_np, show_map=False):
    return build_graph_from_np(img_np, show_map)


# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #


def check_future_constr(node_current, v_constr_dict, e_constr_dict, perm_constr_dict, ignore_dict, start):
    # NO NEED FOR wasted waiting
    if node_current.xy_name in ignore_dict:
        return False
    new_t = node_current.t + 1
    time_window = node_current.t - start.t
    if time_window < 20:
        return False

    future_constr = False
    for nei_xy_name in node_current.neighbours:
        if v_constr_dict and new_t in v_constr_dict[nei_xy_name]:
            future_constr = True
            break
        if e_constr_dict and (node_current.x, node_current.y, new_t) in e_constr_dict[nei_xy_name]:
            future_constr = True
            break
        if perm_constr_dict:
            if len(perm_constr_dict[nei_xy_name]) > 0:
                if new_t >= perm_constr_dict[nei_xy_name][0]:
                    future_constr = True
                    break
    return future_constr


def get_max_final(perm_constr_dict):
    if perm_constr_dict:
        final_list = [v[0] for k, v in perm_constr_dict.items() if len(v) > 0]
        max_final_time = max(final_list) if len(final_list) > 0 else 1
        return max_final_time
    return 1


def get_node(successor_xy_name, node_current, nodes, nodes_dict, open_nodes, closed_nodes, v_constr_dict, e_constr_dict,
             perm_constr_dict, max_final_time, **kwargs):
    new_t = node_current.t + 1

    if 'short_a_star_list' in kwargs and len(kwargs['short_a_star_list']) > 0:
        if successor_xy_name not in kwargs['short_a_star_list']:
            return None, ''

    if v_constr_dict:
        if new_t in v_constr_dict[successor_xy_name]:
            return None, ''

    if e_constr_dict:
        if (node_current.x, node_current.y, new_t) in e_constr_dict[successor_xy_name]:
            return None, ''

    if perm_constr_dict:
        if len(perm_constr_dict[successor_xy_name]) > 0:
            if len(perm_constr_dict[successor_xy_name]) != 1:
                raise RuntimeError('len(perm_constr_dict[successor_xy_name]) != 1')
            final_time = perm_constr_dict[successor_xy_name][0]
            if new_t >= final_time:
                return None, ''

    if max_final_time:
        if node_current.t >= max_final_time:
            new_t = max_final_time + 1

    new_ID = f'{successor_xy_name}_{new_t}'
    if new_ID in open_nodes.dict:
        return open_nodes.dict[new_ID], 'open_nodes'
    if new_ID in closed_nodes.dict:
        return closed_nodes.dict[new_ID], 'closed_nodes'

    node = nodes_dict[successor_xy_name]
    return Node(x=node.x, y=node.y, t=new_t, neighbours=node.neighbours), 'new'


def reset_nodes(start, goal, nodes, **kwargs):
    _ = [node.reset() for node in nodes]
    start.reset(**kwargs)
    return start, goal, nodes


def a_star(start, goal, nodes, h_func,
           v_constr_dict=None, e_constr_dict=None, perm_constr_dict=None,
           plotter=None, middle_plot=False,
           iter_limit=1e100, nodes_dict=None, **kwargs):
    """
    new_t in v_constr_dict[successor_xy_name]
    """
    start_time = time.time()
    # start, goal, nodes = deepcopy_nodes(start, goal, nodes)  # heavy!
    start, goal, nodes = reset_nodes(start, goal, nodes, **kwargs)
    print('\rStarted A*...', end='')
    open_nodes = ListNodes()
    closed_nodes = ListNodes()
    node_current = start
    node_current.h = h_func(node_current, goal)
    open_nodes.add(node_current)
    max_final_time = get_max_final(perm_constr_dict)
    future_constr = False
    iteration = 0
    while len(open_nodes) > 0:
        iteration += 1
        if iteration > iter_limit:
            print(f'\n[ERROR]: out of iterations (more than {iteration})')
            return None, {'runtime': time.time() - start_time, 'n_open': len(open_nodes.heap_list), 'n_closed': len(closed_nodes.heap_list)}
        node_current = open_nodes.pop()

        if node_current.xy_name == goal.xy_name:
            # break
            # if there is a future constraint of a goal
            if len(v_constr_dict[node_current.xy_name]) > 0:
                # we will take the maximum time out of all constraints
                max_t = max(v_constr_dict[node_current.xy_name])
                # and compare to the current time
                # if it is greater, we will continue to expand the search tree
                if node_current.t > max_t:
                    # otherwise break
                    break
            else:
                break
        if 'df_dict' in kwargs:
            future_constr = check_future_constr(node_current, v_constr_dict, e_constr_dict, perm_constr_dict, kwargs['df_dict'], start)
            if future_constr:
                goal = node_current
                break
        for successor_xy_name in node_current.neighbours:
            node_successor, node_successor_status = get_node(
                successor_xy_name, node_current, nodes, nodes_dict, open_nodes, closed_nodes,
                v_constr_dict, e_constr_dict, perm_constr_dict, max_final_time, **kwargs
            )

            successor_current_time = node_current.t + 1  # h(now, next)
            if node_successor is None:
                continue

            # INSIDE OPEN LIST
            if node_successor_status == 'open_nodes':
                if node_successor.t <= successor_current_time:
                    continue
                open_nodes.remove(node_successor)

            # INSIDE CLOSED LIST
            elif node_successor_status == 'closed_nodes':
                if node_successor.t <= successor_current_time:
                    continue
                closed_nodes.remove(node_successor)

            # NOT IN CLOSED AND NOT IN OPEN LISTS
            else:
                node_successor.h = h_func(node_successor, goal)
            node_successor.t = successor_current_time
            node_successor.g = node_successor.t
            node_successor.parent = node_current
            open_nodes.add(node_successor)

        # open_nodes.remove(node_current)
        closed_nodes.add(node_current)

        if plotter and middle_plot and iteration % 10 == 0:
            plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                               closed_list=closed_nodes.get_nodes_list(),
                               start=start, goal=goal, nodes=nodes, a_star_run=True)
        print(f'\r(a_star) iter: {iteration}, closed: {len(closed_nodes.heap_list)}', end='')

    path = None
    if node_current.xy_name == goal.xy_name:
        path = []
        while node_current is not None:
            path.append(node_current)
            node_current = node_current.parent
        path.reverse()

    if plotter and middle_plot:
        plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                           closed_list=closed_nodes.get_nodes_list(),
                           start=start, goal=goal, path=path, nodes=nodes, a_star_run=True)
    # print('\rFinished A*.', end='')
    # if path is None:
    #     print()
    return path, {'runtime': time.time() - start_time, 'n_open': len(open_nodes.heap_list), 'n_closed': len(closed_nodes.heap_list), 'future_constr': future_constr}


def main():
    nodes = [
        Node(x=1, y=1, neighbours=[]),
        Node(x=1, y=2, neighbours=[]),
        Node(x=1, y=3, neighbours=[]),
        Node(x=1, y=4, neighbours=[]),
        Node(x=2, y=1, neighbours=[]),
        Node(x=2, y=2, neighbours=[]),
        Node(x=2, y=3, neighbours=[]),
        Node(x=2, y=4, neighbours=[]),
        Node(x=3, y=1, neighbours=[]),
        Node(x=3, y=2, neighbours=[]),
        Node(x=3, y=3, neighbours=[]),
        Node(x=3, y=4, neighbours=[]),
        Node(x=4, y=1, neighbours=[]),
        Node(x=4, y=2, neighbours=[]),
        Node(x=4, y=3, neighbours=[]),
        Node(x=4, y=4, neighbours=[]),
    ]
    make_neighbours(nodes)
    node_start = nodes[0]
    node_goal = nodes[-1]
    # plotter = Plotter(map_dim=(5, 5), subplot_rows=1, subplot_cols=3)
    result = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=dist_heuristic, plotter=None,
                    middle_plot=True)

    plt.show()
    print(result)
    plt.close()


def try_a_map_from_pic():
    num_agents = 5
    max_episode_steps = 1000
    seed = 10
    # seed = random.randint(0, 100)
    obs_radius = 3
    size = 30

    # Define random configuration
    grid_config = GridConfig(
        num_agents=num_agents,  # number of agents
        size=size,  # size of the grid
        density=0.2,  # obstacle density
        seed=seed,  # set to None for random
        # obstacles, agents and targets
        # positions at each reset
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type='MAPF'
    )
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env)
    obs = env.reset()
    img_np = obs[0]['global_obstacles']
    img_np = 1 - img_np
    map_dim = img_np.shape
    nodes, nodes_dict = build_graph_nodes(img_np=img_np, show_map=False)
    # ------------------------- #
    # x_start, y_start = 97, 99
    # x_goal, y_goal = 38, 89
    # node_start = [node for node in nodes if node.x == x_start and node.y == y_start][0]
    # node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    # ------------------------- #
    # node_start = nodes[100]
    # node_goal = nodes[-1]
    # ------------------------- #
    node_start = random.choice(nodes)
    node_goal = random.choice(nodes)
    print(f'start: {node_start.x}, {node_start.y} -> goal: {node_goal.x}, {node_goal.y}')
    # ------------------------- #
    # ------------------------- #
    # plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    # plotter = None
    # ------------------------- #
    # ------------------------- #
    # target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False
    h_dict = build_heuristic_for_multiple_targets([node_goal], nodes, map_dim)
    h_func = h_func_creator(h_dict)
    # ------------------------- #
    # h_func = dist_heuristic
    # ------------------------- #
    # ------------------------- #
    # constraint_dict = None
    v_constr_dict = {node.xy_name: [] for node in nodes}
    # v_constr_dict = {'30_12': [69], '29_12': [68, 69]}
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict['4_4'].append(10)
    # ------------------------- #
    # ------------------------- #
    # result = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func, plotter=plotter, middle_plot=False)
    profiler.enable()
    result, info = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func,
                          v_constr_dict=v_constr_dict, perm_constr_dict=perm_constr_dict,
                          plotter=None, middle_plot=True, nodes_dict=nodes_dict,
                          )
    profiler.disable()
    if result:
        print('The result is:', *[node.xy_name for node in result], sep='->')
        print('The result is:', *[node.ID for node in result], sep='->')
    # ------------------------- #
    # ------------------------- #
    plt.show()
    # plt.close()


if __name__ == '__main__':
    # random_seed = True
    random_seed = False
    seed = random.choice(range(1000)) if random_seed else 121
    random.seed(seed)
    np.random.seed(seed)
    print(f'SEED: {seed}')
    # main()
    profiler = cProfile.Profile()
    # profiler.enable()
    try_a_map_from_pic()
    # profiler.disable()
    # stats.print_stats()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('../stats/results_a_star.pstat')

# def deepcopy_nodes(start, goal, nodes):
#     copy_nodes_dict = {node.xy_name: copy.deepcopy(node) for node in nodes}
#     copy_start = copy_nodes_dict[start.xy_name]
#     copy_goal = copy_nodes_dict[goal.xy_name]
#     copy_nodes = heap_list(copy_nodes_dict.values())
#     return copy_start, copy_goal, copy_nodes

# for open_node in open_list:
#     if open_node.ID == new_ID:
#         return open_node
# for closed_node in close_list:
#     if closed_node.ID == new_ID:
#         return closed_node


# def get_node_from_open(open_nodes):
#     # v_list = open_list
#     # f_dict = {}
#     # f_vals_list = []
#     # for node in open_nodes.heap_list:
#     #     curr_f = node.f()
#     #     f_vals_list.append(curr_f)
#     #     if curr_f not in f_dict:
#     #         f_dict[curr_f] = []
#     #     f_dict[curr_f].append(node)
#     #
#     # smallest_f_nodes = f_dict[min(f_vals_list)]
#     #
#     # h_dict = {}
#     # h_vals_list = []
#     # for node in smallest_f_nodes:
#     #     curr_h = node.h
#     #     h_vals_list.append(curr_h)
#     #     if curr_h not in h_dict:
#     #         h_dict[curr_h] = []
#     #     h_dict[curr_h].append(node)
#     #
#     # smallest_h_from_smallest_f_nodes = h_dict[min(h_vals_list)]
#     # next_node = random.choice(smallest_h_from_smallest_f_nodes)
#
#     next_node = open_nodes.pop()
#     return next_node


# NO NEED FOR wasted waiting
# if 'a_star_mode' in kwargs and kwargs['a_star_mode'] == 'fast':
#     if successor_xy_name == node_current.xy_name:
#         no_constraints = True
#         for nei_xy_name in node_current.neighbours:
#             if v_constr_dict and new_t in v_constr_dict[nei_xy_name]:
#                 no_constraints = False
#                 break
#
#             if no_constraints and e_constr_dict and (node_current.x, node_current.y, new_t) in e_constr_dict[nei_xy_name]:
#                 no_constraints = False
#                 break
#         if no_constraints:
#             return None, 'future_constr'