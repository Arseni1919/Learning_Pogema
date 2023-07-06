import random

from globals import *

from plot_functions.plot_objects import Plotter
from pogema import pogema_v0, Hard8x8, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig

from algs.alg_po_sds import run_po_sds, run_full_sds
from algs.alg_a_star_policy import run_a_star_policy


def save_and_show_results(to_save_dict, file_dir, plotter=None, n_agents_list=None):
    # Serializing json
    json_object = json.dumps(to_save_dict, indent=4)
    with open(file_dir, "w") as outfile:
        outfile.write(json_object)
    # Results saved.
    if plotter:
        with open(f'{file_dir}', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        plotter.plot_big_test(json_object, is_json=True, num_agents=max(n_agents_list))


def create_to_save_dict(algs_to_test_dict, n_agents_list, runs_per_n_agents, **kwargs):
    stats_dict = {
        alg_name: {
            n_agents: {
                'succeeded_list': [],
                'soc_list': [],
                'steps_list': [],
            } for n_agents in n_agents_list
        } for alg_name, _ in algs_to_test_dict.items()
    }
    to_save_dict = {
        'stats_dict': stats_dict,
        'runs_per_n_agents': runs_per_n_agents,
        'n_agents_list': n_agents_list,
        'algs_to_test_list': [alg_name for alg_name, _ in algs_to_test_dict.items()]
    }
    to_save_dict.update(kwargs)
    return to_save_dict


def update_statistics_dict(stats_dict, alg_name, n_agents, alg_info):
    succeeded = alg_info['succeeded']
    stats_dict[alg_name][n_agents]['succeeded_list'].append(succeeded)
    if succeeded:
        stats_dict[alg_name][n_agents]['soc_list'].append(alg_info['soc'])
        stats_dict[alg_name][n_agents]['steps_list'].append(alg_info['steps'])


def set_seed(random_seed, seed, seeds):
    if random_seed:
        seed = seeds.pop()

    random.seed(seed)
    np.random.seed(seed)
    print(f'SEED: {seed}')
    return seed


def big_test(
        algs_to_test_dict: dict,
        n_agents_list: list,
        runs_per_n_agents: int,
        time_per_alg_limit,
        random_seed: bool,
        seed: int,
        seeds,
        plotter,
        plot_per,
        to_save_results,
        file_dir,
        profiler=None,
        obs_radius=3,
):
    print(f'\nTest started at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    # for plotter
    to_save_dict = create_to_save_dict(
        algs_to_test_dict=algs_to_test_dict,
        n_agents_list=n_agents_list,
        runs_per_n_agents=runs_per_n_agents,
        time_per_alg_limit=time_per_alg_limit,
    )
    stats_dict = to_save_dict['stats_dict']

    # for num of agents
    for num_agents in n_agents_list:

        # for same starts and goals
        for i_run in range(runs_per_n_agents):

            curr_seed = set_seed(random_seed, seed, seeds)

            # for at max 5 minutes
            for alg_name, (alg, params) in algs_to_test_dict.items():

                # Define random configuration
                grid_config = GridConfig(
                    num_agents=num_agents,  # number of agents
                    size=30,  # size of the grid
                    density=0.2,  # obstacle density
                    seed=curr_seed,  # set to None for random
                    # obstacles, agents and targets
                    # positions at each reset
                    max_episode_steps=time_per_alg_limit,  # horizon
                    obs_radius=obs_radius,  # defines field of view
                    observation_type='MAPF'
                )
                # env = pogema_v0(grid_config=Hard8x8())
                env = pogema_v0(grid_config=grid_config)
                env = AnimationMonitor(env)

                params['plot_per'] = plot_per
                # alg_info = alg(env, num_agents, time_per_alg_limit, obs_radius, plotter, **params)
                alg_info = alg(env, num_agents, time_per_alg_limit, obs_radius, None, **params)

                # plot + print
                print(f'\n#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'\r[{num_agents} agents][{i_run + 1} run][{alg_name}] -> succeeded: {alg_info["succeeded"]}, steps: {alg_info["steps"]}\n')
                update_statistics_dict(stats_dict, alg_name, num_agents, alg_info)
                if i_run % 1 == 0:
                    plotter.plot_big_test(to_save_dict, num_agents=num_agents)

        if to_save_results:
            save_and_show_results(to_save_dict, file_dir, plotter, n_agents_list)
            print('Results are saved.')

    print(f'\nTest finished at: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


def main():
    obs_radius = 3

    algs_to_test_dict = {
        'PO-SDS': (run_po_sds, {
            'small_iters': 3,
            'color': 'tab:orange',
            'po_field': False,
        }),
        'FO-SDS': (run_full_sds, {
            'color': 'tab:blue',
        }),
        # 'PO-SDS (agents and map)': (run_po_sds, {
        #     'small_iters': 3,
        #     'color': 'tab:orange',
        #     'po_field': True,
        # }),
        'A*-Policy': (run_a_star_policy, {
            'color': 'tab:purple',
        })
    }

    # n_agents_list = [2, 3, 4, 5]
    # n_agents_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # n_agents_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # n_agents_list = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_agents_list = [5, 10, 15, 20, 25, 30, 35, 40]  # !!!!!!!!!!!!!!!!!
    # n_agents_list = [10, 20, 30, 40]
    # n_agents_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # runs_per_n_agents = 50
    runs_per_n_agents = 40
    # runs_per_n_agents = 20  # !!!!!!!!!!!!!!!!!
    # runs_per_n_agents = 10
    # runs_per_n_agents = 5
    # runs_per_n_agents = 1
    # runs_per_n_agents = 3

    random_seed = True
    # random_seed = False
    seed = 116
    seeds = list(range(10000))
    random.shuffle(seeds)

    # time_per_alg_limit = 50
    time_per_alg_limit = 500

    plotter = Plotter(for_big_experiments=True)
    plot_per = 20

    to_save_results = True
    # to_save_results = False
    file_dir = f'logs_for_graphs/{datetime.now().strftime("%Y-%m-%d--%H-%M")}_ALGS-{len(algs_to_test_dict)}_RUNS-{runs_per_n_agents}.json'

    # profiler = None
    profiler = cProfile.Profile()

    if profiler:
        profiler.enable()
    big_test(
        algs_to_test_dict=algs_to_test_dict,
        n_agents_list=n_agents_list,
        runs_per_n_agents=runs_per_n_agents,
        time_per_alg_limit=time_per_alg_limit,
        random_seed=random_seed,
        seed=seed,
        seeds=seeds,
        plotter=plotter,
        plot_per=plot_per,
        to_save_results=to_save_results,
        file_dir=file_dir,
        profiler=profiler,
        obs_radius=obs_radius,
    )
    if profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats('stats/results_scale_experiments.pstat')
        print('Profile saved to stats/results_scale_experiments.pstat.')
    plt.show()


if __name__ == '__main__':
    main()
