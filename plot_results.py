from globals import *
from plot_functions.plot_objects import Plotter


def show_results(file_dir, plotter):
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        n_agents_list = json_object['n_agents_list']
    plotter.plot_big_test(json_object, is_json=True, num_agents=max(n_agents_list))
    plt.show()


def main():
    file_dir = 'logs_for_graphs/2023-06-24--20-17_ALGS-2_RUNS-3.json'

    plotter = Plotter(for_big_experiments=True)
    show_results(file_dir, plotter=plotter)


if __name__ == '__main__':
    main()
