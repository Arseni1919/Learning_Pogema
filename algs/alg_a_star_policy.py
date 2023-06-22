import random

import numpy as np
from pogema import pogema_v0, Hard8x8, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from IPython.display import SVG, display
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from plot_functions.plot_objects import Plotter

INF = 1e7


class GridMemory:
    def __init__(self, start_r=64):
        self._memory = np.zeros(shape=(start_r * 2 + 1, start_r * 2 + 1), dtype=np.bool_)

    @staticmethod
    def _try_to_insert(x, y, source, target):
        r = source.shape[0] // 2
        try:
            target[x - r:x + r + 1, y - r:y + r + 1] = source
            return True
        except ValueError:
            return False

    def _increase_memory(self):
        m = self._memory
        r = self._memory.shape[0]
        self._memory = np.zeros(shape=(r * 2 + 1, r * 2 + 1))
        assert self._try_to_insert(r, r, m, self._memory)

    def update(self, x, y, obstacles):
        while True:
            r = self._memory.shape[0] // 2
            if self._try_to_insert(r + x, r + y, obstacles, self._memory):
                break
            self._increase_memory()

    def is_obstacle(self, x, y):
        r = self._memory.shape[0] // 2
        if -r <= x <= r and -r <= y <= r:
            return self._memory[r + x, r + y]
        else:
            return False


class Node:
    def __init__(self, coord: (int, int) = (INF, INF), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        elif self.g != other.g:
            return self.g < other.g
        else:
            return self.i < other.i or self.j < other.j


def h(node, target):
    nx, ny = node
    tx, ty = target
    return abs(nx - tx) + abs(ny - ty)


def a_star(start, target, grid: GridMemory, max_steps=10000):
    open_ = list()
    closed = {start: None}

    heappush(open_, Node(start, 0, h(start, target)))

    for step in range(int(max_steps)):
        u = heappop(open_)

        for n in [(u.i - 1, u.j), (u.i + 1, u.j), (u.i, u.j - 1), (u.i, u.j + 1)]:
            if not grid.is_obstacle(*n) and n not in closed:
                heappush(open_, Node(n, u.g + 1, h(n, target)))
                closed[n] = (u.i, u.j)

        if step >= max_steps or (u.i, u.j) == target or len(open_) == 0:
            break

    next_node = target if target in closed else None
    path = []
    while next_node is not None:
        path.append(next_node)
        next_node = closed[next_node]

    return list(reversed(path))


class AStarAgent:
    def __init__(self, seed=0):
        self._moves = GridConfig().MOVES
        self._reverse_actions = {tuple(self._moves[i]): i for i in range(len(self._moves))}

        self._gm = None
        self._saved_xy = None
        self.clear_state()
        self._rnd = np.random.default_rng(seed)

    def act(self, obs):
        xy, target_xy, obstacles, agents = obs['xy'], obs['target_xy'], obs['obstacles'], obs['agents']


        if self._saved_xy is not None and h(self._saved_xy, xy) > 1:
            raise IndexError("Agent moved more than 1 step. Please, call clear_state method before new episode.")
        if self._saved_xy is not None and h(self._saved_xy, xy) == 0 and xy != target_xy:
            return self._rnd.integers(len(self._moves))
        self._gm.update(*xy, obstacles)
        path = a_star(xy, target_xy, self._gm, )
        if len(path) <= 1:
            action = 0
        else:
            (x, y), (tx, ty), *_ = path
            action = self._reverse_actions[tx - x, ty - y]

        self._saved_xy = xy
        return action

    def clear_state(self):
        self._saved_xy = None
        self._gm = GridMemory()


class BatchAStarAgent:
    def __init__(self):
        self.astar_agents = {}

    def act(self, observations):
        actions = []
        for idx, obs in enumerate(observations):
            if idx not in self.astar_agents:
                self.astar_agents[idx] = AStarAgent()
            actions.append(self.astar_agents[idx].act(obs))
        return actions

    def reset_states(self):
        self.astar_agents = {}


def get_actions(agents, obs):
    actions = []
    for i_obs_index, i_obs in enumerate(obs):
        # agent_obs = {
        #     'xy': (5, 5),
        #     'target_xy': (np.where(i_obs[2] == 1)[0][0], np.where(i_obs[2] == 1)[1][0]),
        #     'obstacles': i_obs[0],
        #     'agents': i_obs[1],
        # }
        # action = agents[i_obs_index].act(agent_obs)
        action = agents[i_obs_index].act(i_obs)
        actions.append(action)
    return actions


def main():
    num_agents = 10
    max_episode_steps = 1000
    plotter = Plotter()
    # seed = 10
    seed = random.randint(0, 100)

    # Define random configuration
    grid_config = GridConfig(
        num_agents=num_agents,  # number of agents
        size=30,  # size of the grid
        density=0.2,  # obstacle density
        seed=seed,  # set to None for random
        # obstacles, agents and targets
        # positions at each reset
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=3,  # defines field of view
        observation_type='MAPF'
    )
    # env = pogema_v0(grid_config=Hard8x8())
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env)

    # create agents
    agents = []
    for i in range(num_agents):
        agent = AStarAgent()
        agents.append(agent)

    obs = env.reset()

    # while True:
    for i in range(max_episode_steps):
        # Using random policy to make actions
        # TODO: here message exchange
        actions = get_actions(agents, obs)
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

    env.save_animation("render.svg")
    env.save_animation("render_agent_0.svg", AnimationConfig(egocentric_idx=0))


if __name__ == '__main__':
    main()
