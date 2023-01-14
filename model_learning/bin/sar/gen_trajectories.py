import argparse
import logging
import os
from typing import get_args

from model_learning import SelectionType
from model_learning.bin.sar import create_sar_world, add_common_arguments, create_mental_models, DISCOUNT, HORIZON, \
    ACT_SELECTION, RATIONALITY
from model_learning.util.cmd_line import save_args
from model_learning.util.io import create_clear_dir, save_object
from model_learning.util.logging import change_log_handler

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'
__description__ = 'Collects a set of fixed-length trajectories in the search-and-rescue domain. ' \
                  'Saves the trajectories to a gif animation and pickle file.'

NUM_TRAJECTORIES = 1  # 5  # 10
TRAJ_LENGTH = 25  # 30
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
    team_trajectories = env.generate_team_trajectories(
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

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)

    parser.add_argument('--selection', '-as', type=str, choices=get_args(SelectionType), default=ACT_SELECTION,
                        help='Agents\' action selection criterion, to untie equal-valued actions.')
    parser.add_argument('--horizon', '-hz', type=int, default=HORIZON, help='Agents\' planning horizon.')
    parser.add_argument('--rationality', '-r', type=float, default=RATIONALITY,
                        help='Agents\' rationality when selecting actions under a probabilistic criterion.')
    parser.add_argument('--discount', '-d', type=float, default=DISCOUNT, help='Agents\' planning discount factor.')

    parser.add_argument('--trajectories', '-t', type=int, default=NUM_TRAJECTORIES,
                        help='Number of trajectories to generate.')
    parser.add_argument('--length', '-l', type=int, default=TRAJ_LENGTH, help='Length of the trajectories to generate.')

    args = parser.parse_args()

    main()
