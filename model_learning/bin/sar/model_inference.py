import argparse
import logging
import os
import tqdm
from typing import List, Optional

from model_learning import TeamTrajectory, TeamModelDistTrajectory
from model_learning.bin.sar import add_common_arguments, create_sar_world, create_observers
from model_learning.inference import plot_team_model_inference, team_trajectories_model_inference
from model_learning.util.cmd_line import save_args
from model_learning.util.io import load_object, create_clear_dir, save_object
from model_learning.util.logging import change_log_handler, MultiProcessLogger
from model_learning.util.plot import dummy_plotly

__author__ = 'Pedro Sequeira, Haochen Wu'
__email__ = 'pedrodbs@gmail.com, hcaawu@gmail.com'
__description__ = 'Performs reward model inference in the search-and-rescue domain.' \
                  'First, loads a set of team trajectories corresponding to observed behavior between two ' \
                  'collaborative agents. ' \
                  'An observer agent then maintains beliefs (a probability distribution over reward models) about ' \
                  'each other agent, which are updated via PsychSim inference according to the agents\' actions' \
                  'in the given trajectories. Trajectories with model inference are saved into a .pkl file.'

TRAJECTORIES_FILE = 'trajectories.pkl.gz'


def main():
    assert os.path.isfile(args.traj_file), f'Could not find the team trajectories file at {args.traj_file}'

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    save_args(args, os.path.join(output_dir, 'args.json'))
    log_file = os.path.join(output_dir, 'model_inference.log')
    mp_log: Optional[MultiProcessLogger] = None
    if args.processes > 1 or args.processes <= -1:
        mp_log = MultiProcessLogger(log_file, level=args.verbosity)
    else:
        change_log_handler(log_file, level=args.verbosity)

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)

    # create and set observers' mental models
    observers = create_observers(env, team_config, profiles)

    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    # load trajectories
    logging.info('========================================')
    logging.info(f'Loading team trajectories from {args.traj_file}...')
    trajectories: List[TeamTrajectory] = load_object(args.traj_file)
    logging.info(f'Loaded {len(trajectories)} team trajectories of length {len(trajectories[0])}')

    logging.info('========================================')
    logging.info(f'Performing model inference over {len(trajectories)} trajectories...')
    team_model_dist_trajs: List[TeamModelDistTrajectory] = team_trajectories_model_inference(
        env.world, team, trajectories, observers,
        models_dists=team_config.get_models_distributions(),
        threshold=args.prune,
        processes=args.processes,
        seed=args.seed,
        verbose=args.verbosity <= logging.INFO)

    logging.info('========================================')
    logging.info(f'Saving results to {output_dir}...')

    # save all trajectories to pickle file
    file_path = os.path.join(output_dir, TRAJECTORIES_FILE)
    logging.info(f'Saving trajectories to {file_path}...')
    save_object(team_model_dist_trajs, file_path, compress_gzip=True)

    dummy_plotly()  # to clear plotly import message

    plots_dir = os.path.join(output_dir, 'model_inference')
    create_clear_dir(plots_dir, clear=False)
    logging.info(f'Generating model inference plots, saving to {plots_dir}...')
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        for other_role, models in ag_conf.mental_models.items():
            output_img = os.path.join(plots_dir, f'{role.lower()}_{other_role.lower()}_inference.{args.img_format}')
            plot_team_model_inference(team_model_dist_trajs, role, other_role, output_img)

    logging.info('Done!')

    if mp_log is not None:
        mp_log.close()  # close multiprocess logger


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)

    parser.add_argument(
        '--traj-file', '-tf', type=str, required=True,
        help='Path to the .pkl.gz file containing the team trajectories over which to perform model inference.')

    args = parser.parse_args()

    main()
