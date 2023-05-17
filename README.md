# Learning  [Pogema](https://github.com/AIRI-Institute/pogema)


## Quick Start

Just install from PyPI:

```
pip install pogema
```

Run example: 

```python
import numpy as np
from pogema import pogema_v0, Hard8x8, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from IPython.display import SVG, display

def main():
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Configure grid
    grid = """
    .....#.....
    .....#.....
    ...........
    .....#.....
    .....#.....
    #.####.....
    .....###.##
    .....#.....
    .....#.....
    ...........
    .....#.....
    """
    num_agents = 8

    # Define new configuration with 8 randomly placed agents
    grid_config = GridConfig(map=grid, num_agents=num_agents)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # or:
    # grid_config = GridConfig(
    #     num_agents=4,  # number of agents
    #     size=32,  # size of the grid
    #     density=0.2,  # obstacle density
    #     seed=1,  # set to None for random
    #     # obstacles, agents and targets
    #     # positions at each reset
    #     max_episode_steps=128,  # horizon
    #     obs_radius=5,  # defines field of view
    # )
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    env = pogema_v0(grid_config=grid_config)
    # env = pogema_v0(grid_config=Hard8x8())
    env = AnimationMonitor(env)
    obs = env.reset()  

    while True:
        # Using random policy to make actions
        actions = env.sample_actions()
        print(actions)
        obs, reward, terminated, info = env.step(actions)  
        if all(terminated):
            break

    env.save_animation("render.svg")
    display(SVG('render.svg'))

if __name__ == '__main__':
    main()
```

## Actions

0 - idle

1 - up

2 - down

3 - left

4 - right

## Configurations and Observations

There are two types of configurations:

### 'POMAPF'

`obs` is three matrices in size of agent's window: 
- obstacles
- other agents around
- target position/direction

### 'MAPF'

`obs` is the following: 
- obstacles
- other agents around
- relative xy from the start position
- target position/direction
- global map of obstacles
- global xy of the agent
- global target position

## Rewards

1 if arrived to target, 0 otherwise.

## Render Options

To get `.svg` of the run of the environment do:

```python
from pogema.animation import AnimationMonitor, AnimationConfig
...
env = ...
env = AnimationMonitor(env)
...
env.save_animation("render.svg")
env.save_animation("render_agent_0.svg", AnimationConfig(egocentric_idx=0))
```

## Credits

- [pogema | github](https://github.com/AIRI-Institute/pogema)
- [pogema | Pogema animation SVG](https://colab.research.google.com/drive/19dSEGTQeM3oVJtVjpC162t1XApmv6APc?usp=sharing)
- [pogema | a_star_policy](https://github.com/AIRI-Institute/pogema/blob/main/pogema/a_star_policy.py)
- [pogema | DQN example](https://colab.research.google.com/drive/1vPwTd0PnzpWrB-bCHqoLSVwU9G9Lgcmv?usp=sharing)
- [pogema | APPO](https://github.com/Tviskaron/pogema-baselines/tree/main/appo)
- [RL algs | stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [RL algs | RLLib](https://docs.ray.io/en/master/rllib/rllib-algorithms.html)

