from globals import *


def plot_example(ax, info):
    ax.cla()
    ax.plot([1, 2, 10])


def plot_field(ax, info):
    ax.cla()
    obs_agents = info.obs
    num_agents = info.num_agents

    # plot field
    obs_agent = obs_agents[0]
    global_obstacles = obs_agent['global_obstacles'] * -1
    global_obstacles = np.transpose(global_obstacles)
    # ax.imshow(global_obstacles, origin='lower', cmap='gray')
    ax.imshow(global_obstacles, cmap='gray')

    # plot agents
    for i in range(num_agents):
        obs_agent = obs_agents[i]

        # target
        global_target_xy = obs_agent['global_target_xy']
        target_circle = plt.Circle((global_target_xy[0], global_target_xy[1]), 1, color='red', alpha=1)
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


