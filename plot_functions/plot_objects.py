import time

import matplotlib.pyplot as plt

from globals import *
from plot_functions import *


class Plotter:
    def __init__(self, to_render=True, plot_every=1):
        self.name = 'Plotter'
        self.to_render = to_render
        self.plot_every = plot_every
        if self.to_render:
            self.fig, self.ax = plt.subplot_mosaic("AAB;AAC;AAD", figsize=(12, 8))

    def render(self, info):
        if self.to_render:
            info = AttributeDict(info)
            if info.i_step % self.plot_every == 0:
                info.update({})

                # plot_mst_field(self.ax['A'], info)
                plot_example(self.ax['A'], info)
                plot_example(self.ax['B'], info)
                plot_example(self.ax['C'], info)

                plt.pause(0.001)
                # plt.show()


def main():
    plotter = Plotter()

    plotter.render(info={'i_step': 1})

    plt.show()


if __name__ == '__main__':
    main()





