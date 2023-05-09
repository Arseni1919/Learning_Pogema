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

    # Define new configuration with 8 randomly placed agents
    grid_config = GridConfig(map=grid, num_agents=8)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # or:
    grid_config = GridConfig(
        num_agents=4,  # number of agents
        size=32,  # size of the grid
        density=0.2,  # obstacle density
        seed=1,  # set to None for random
        # obstacles, agents and targets
        # positions at each reset
        max_episode_steps=128,  # horizon
        obs_radius=5,  # defines field of view
    )
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    env = pogema_v0(grid_config=grid_config)
    # env = pogema_v0(grid_config=Hard8x8())
    env = AnimationMonitor(env)
    obs = env.reset()  # here

    while True:
        # Using random policy to make actions
        obs, reward, terminated, info = env.step(env.sample_actions())  # and here
        # env.render()
        if all(terminated):
            break

    env.save_animation("render.svg")
    display(SVG('render.svg'))


if __name__ == '__main__':
    main()
