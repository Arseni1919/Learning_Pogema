import numpy as np

from globals import *


def plot_example(ax, info):
    ax.cla()
    ax.plot([1, 2, 10])


def plot_csr(ax, info):
    """
    info={
        'to_save_dict': to_save_dict,
        'num_agents': num_agents,
        'is_json': is_json
    }
    """
    ax.cla()
    stats_dict = info['to_save_dict']['stats_dict']
    n_agents_list = info['to_save_dict']['n_agents_list']
    algs_to_test_list = info['to_save_dict']['algs_to_test_list']
    num_agents = info['num_agents']
    is_json = info['is_json']

    for alg_name in algs_to_test_list:
        csr_list = []
        x_list = []
        for i_num_agents in n_agents_list:
            i_num_agents_index = i_num_agents
            if is_json:
                i_num_agents_index = f'{i_num_agents_index}'
            succeeded_list = stats_dict[alg_name][i_num_agents_index]['succeeded_list']
            if len(succeeded_list) > 0:
                csr_v = sum(succeeded_list) / len(succeeded_list)
                csr_list.append(csr_v)
                x_list.append(i_num_agents)
            if i_num_agents == num_agents:
                break
        ax.plot(x_list, csr_list, label=f'{alg_name}', marker='.')
    ax.legend()
    ax.set_title(f'CSR')
    ax.set_ylabel('CSR')
    ax.set_xlabel('N agents')
    ax.set_xticks(n_agents_list)


def plot_soc(ax, info):
    """
    info={
        'to_save_dict': to_save_dict,
        'num_agents': num_agents,
        'is_json': is_json
    }
    """
    ax.cla()
    stats_dict = info['to_save_dict']['stats_dict']
    n_agents_list = info['to_save_dict']['n_agents_list']
    algs_to_test_list = info['to_save_dict']['algs_to_test_list']
    num_agents = info['num_agents']
    is_json = info['is_json']

    for alg_name in algs_to_test_list:
        soc_metric_list = []
        x_list = []
        for i_num_agents in n_agents_list:
            i_num_agents_index = i_num_agents
            if is_json:
                i_num_agents_index = f'{i_num_agents_index}'
            soc_list = stats_dict[alg_name][i_num_agents_index]['soc_list']
            if len(soc_list) > 0:
                soc_v = np.mean(soc_list)
                soc_metric_list.append(soc_v)
                x_list.append(i_num_agents)
            if i_num_agents == num_agents:
                break
        ax.plot(x_list, soc_metric_list, label=f'{alg_name}', marker='.')
    ax.legend()
    ax.set_title(f'SoC')
    ax.set_ylabel('SoC')
    ax.set_xlabel('N agents')
    ax.set_xticks(n_agents_list)


def plot_field(ax, info):
    ax.cla()
    obs_agents = info.obs
    num_agents = info.num_agents

    # plot field
    obs_agent = obs_agents[0]
    global_obstacles = obs_agent['global_obstacles'] * -1
    global_obstacles = np.transpose(global_obstacles)
    # ax.imshow(global_obstacles, origin='lower', cmap='gray')
    ax.imshow(global_obstacles, cmap='gray', origin='lower')

    # plot agents
    for i in range(num_agents):
        obs_agent = obs_agents[i]

        # target
        global_target_xy = obs_agent['global_target_xy']
        color = 'yellow' if i == 0 else 'red'
        target_circle = plt.Circle((global_target_xy[0], global_target_xy[1]), 1, color=color, alpha=1)
        # target_circle = plt.Circle((global_target_xy[1], global_target_xy[0]), 1, color='red', alpha=1)
        ax.add_patch(target_circle)

        # agent
        global_xy = obs_agent['global_xy']
        color = 'purple' if i == 0 else 'blue'
        agent_circle = plt.Circle((global_xy[0], global_xy[1]), 0.5, color=color, alpha=1)
        # agent_circle = plt.Circle((global_xy[1], global_xy[0]), 0.5, color='blue', alpha=1)
        ax.add_patch(agent_circle)

    ax.set_title(f'field')


def plot_nei_agents(ax, info):
    ax.cla()
    obs_agents = info.obs
    num_agents = info.num_agents
    obs_agent = obs_agents[0]
    other_agents = obs_agent['agents'] * -1
    other_agents = np.transpose(other_agents)
    ax.imshow(other_agents, cmap='Blues')

    ax.set_title(f'agents')


