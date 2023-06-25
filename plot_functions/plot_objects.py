import time

import matplotlib.pyplot as plt

from globals import *
from plot_functions.plot_functions import *


class Plotter:
    def __init__(self, to_render=True, plot_every=1, for_big_experiments=False):
        self.name = 'Plotter'
        self.to_render = to_render
        self.plot_every = plot_every
        if self.to_render:
            if for_big_experiments:
                self.fig, self.ax = plt.subplot_mosaic("CD;CD;CD", figsize=(12, 8))
            else:
                self.fig, self.ax = plt.subplot_mosaic("AAB;AAC;AAD", figsize=(12, 8))

    def render(self, info):
        if self.to_render:
            info = AttributeDict(info)
            if info.i_step % self.plot_every == 0:
                info.update({})

                # plot_mst_field(self.ax['A'], info)
                plot_field(self.ax['A'], info)
                plot_nei_agents(self.ax['B'], info)
                plot_b_nodes(self.ax['C'], info)

                plt.pause(0.001)
                # plt.show()

    def plot_big_test(self, to_save_dict, num_agents=None, is_json=False):
        if self.to_render:
            plot_csr(self.ax['C'], info={'to_save_dict': to_save_dict,
                                         'num_agents': num_agents,
                                         'is_json': is_json})
            plot_soc(self.ax['D'], info={'to_save_dict': to_save_dict,
                                         'num_agents': num_agents,
                                         'is_json': is_json})
            plt.pause(0.001)


def main():
    plotter = Plotter()

    plotter.render(info={'i_step': 1})

    plt.show()


if __name__ == '__main__':
    main()
