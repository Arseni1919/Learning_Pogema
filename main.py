from pogema import pogema_v0, Hard8x8
import gym
from IPython.display import SVG, display
from pogema.animation import AnimationMonitor, AnimationConfig


def main():
    env = pogema_v0(grid_config=Hard8x8())
    env = AnimationMonitor(env)
    # env = gym.make('Pogema-8x8-easy-v0', integration='gym')
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
