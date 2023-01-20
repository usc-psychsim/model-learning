import argparse
import logging
import os
import pandas as pd
from typing import List, Dict

from model_learning import TeamTrajectory
from model_learning.bin.sar import create_sar_world, add_common_arguments, create_mental_models, add_agent_arguments, \
    add_trajectory_arguments
from model_learning.features.counting import empirical_feature_counts
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.trajectory import get_trajectory_action_counts
from model_learning.util.cmd_line import save_args
from model_learning.util.io import create_clear_dir, save_object
from model_learning.util.logging import change_log_handler
from model_learning.util.plot import plot_bar, dummy_plotly
from psychsim.action import Action

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'
__description__ = 'Collects a set of fixed-length trajectories in the search-and-rescue domain. ' \
                  'Saves the trajectories to a gif animation and pickle file.'

TRAJECTORIES_FILE = 'trajectories.pkl.gz'


def main():
    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    change_log_handler(os.path.join(output_dir, 'collect.log'), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)

    # create and set agent mental models
    create_mental_models(env, team_config, profiles)

    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    logging.info('========================================')
    logging.info(f'Generating {args.trajectories} trajectories of length {args.length}...')

    # generate trajectories using agents' policies and models
    team_trajectories: List[TeamTrajectory] = env.generate_team_trajectories(
        team,
        trajectory_length=args.length,
        n_trajectories=args.trajectories,
        horizon=args.horizon,
        selection=args.selection,
        processes=args.processes,
        threshold=args.prune,
        seed=args.seed)

    output_dir = args.output

    logging.info('========================================')
    logging.info(f'Saving results to {output_dir}...')

    # save trajectories to an animation (gif) file
    anim_path = os.path.join(output_dir, 'animations')
    create_clear_dir(anim_path, clear=False)
    logging.info(f'Saving animations for each trajectory to {anim_path}...')
    env.play_team_trajectories(team_trajectories, anim_path)

    # save all trajectories to pickle file
    file_path = os.path.join(output_dir, TRAJECTORIES_FILE)
    logging.info(f'Saving trajectories to {file_path}...')
    save_object(team_trajectories, file_path, compress_gzip=True)

    dummy_plotly()  # to clear plotly import message

    # save trajectory statistics
    stats_path = os.path.join(output_dir, 'stats')
    create_clear_dir(stats_path, clear=False)
    logging.info(f'Saving statistics of trajectories to {stats_path}...')

    # gets weighted reward feature counts across all trajectories
    for role in team_config.keys():
        reward_vector = SearchRescueRewardVector(env, env.world.agents[role])
        feature_func = lambda s: reward_vector.get_values(s)
        efc = empirical_feature_counts(team_trajectories, feature_func)
        plot_bar(pd.DataFrame(efc.reshape(1, -1), columns=reward_vector.names),
                 'Empirical Feature Counts',
                 os.path.join(stats_path, f'feature-counts-{role.lower()}.{args.img_format}'),
                 x_label='Reward Features', y_label='Mean Count')

    # gets actions counts for each agent
    action_counts: Dict[str, Dict[str, List[int]]] = {}
    for t, trajectory in enumerate(team_trajectories):
        _action_counts = get_trajectory_action_counts(trajectory, env.world)
        for agent in _action_counts.keys():
            if agent not in action_counts:
                action_counts[agent] = {}
            for action, count in _action_counts[agent].items():
                # filter action name
                action = str(Action({k: v for k, v in dict(next(iter(action))).items()
                                     if k not in ['subject', 'world']})).title()
                if action not in action_counts[agent]:
                    action_counts[agent][action] = []
                action_counts[agent][action].append(count)

    for agent, counts in action_counts.items():
        plot_bar(pd.DataFrame(counts), f'Average Action Counts for {agent}',
                 os.path.join(stats_path, f'action-counts-{agent.lower()}.{args.img_format}'),
                 x_label='Action', y_label='Mean Count', plot_mean=True)

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)
    add_agent_arguments(parser)
    add_trajectory_arguments(parser)

    args = parser.parse_args()

    main()
