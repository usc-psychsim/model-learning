import argparse
import logging
import numpy as np
import os
import pandas as pd
import tqdm
from typing import List, Dict

from model_learning import MultiagentTrajectory
from model_learning.bin.sar import create_sar_world, add_common_arguments, add_agent_arguments, \
    add_trajectory_arguments, plot_feature_counts, _get_estimated_feature_counts
from model_learning.features.counting import empirical_feature_counts, mean_feature_counts
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.trajectory import copy_world
from model_learning.util.cmd_line import save_args
from model_learning.util.io import create_clear_dir, load_object
from model_learning.util.logging import change_log_handler
from model_learning.util.plot import dummy_plotly, plot_bar

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Compares *empirical* feature counts obtained from demonstrated trajectories against feature ' \
                  'counts *estimated* via Monte-Carlo sampling from a stochastic policy using some reward function ' \
                  '(e.g., resulting from IRL).'


def main():
    # checks files
    assert os.path.isfile(args.traj_file), f'Could not find the team trajectories file at {args.traj_file}'
    np.set_printoptions(precision=2)

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    change_log_handler(os.path.join(output_dir, 'fc_est_diff.log'), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)
    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    # load trajectories
    logging.info('========================================')
    logging.info(f'Loading team trajectories from {args.traj_file}...')
    trajectories: List[MultiagentTrajectory] = load_object(args.traj_file)
    logging.info(f'Loaded {len(trajectories)} trajectories of length {len(trajectories[0])}')

    logging.info('----------------------------------------')

    dummy_plotly()  # to clear plotly import message
    output_dir = os.path.join(args.output, 'results')
    create_clear_dir(output_dir, clear=False)
    logging.info(f'Computing feature counts empirically and via Monte-Carlo estimation, '
                 f'saving results to:\n\t{output_dir}')

    logging.info(f'Computing empirical feature counts for {len(trajectories)} trajectories...')
    empirical_fcs: Dict[str, np.ndarray] = {}
    for agent in tqdm.tqdm(team):
        agent_lrv = SearchRescueRewardVector(env, agent)
        feature_func = lambda s: agent_lrv.get_values(s)
        features, probs = empirical_feature_counts(trajectories, feature_func, unweighted=True)
        plot_feature_counts(features, probs, title=f'Expert {agent.name} Empirical FCs',
                            labels=agent_lrv.names, output_dir=output_dir, img_format=args.img_format)
        empirical_fcs[agent.name] = mean_feature_counts(features, probs)  # shape: (num_features, )
    logging.info(f'Empirical feature counts: {empirical_fcs}')

    logging.info('----------------------------------------')

    traj_len = len(trajectories[0])
    logging.info(f'Estimating feature counts from {len(trajectories)} starting states with stochastic policies '
                 f'sampled for {args.trajectories} Monte-Carlo trajectories of length {traj_len}...')

    # estimate feature counts for each agent
    estimated_fcs: Dict[str, np.ndarray] = {}
    orig_world = env.world
    for agent in team:
        env.world = copy_world(orig_world)  # replace world in env
        efc = _get_estimated_feature_counts(
            trajectories, env, env.world.agents[agent.name], team_config, profiles, output_dir, args)
        estimated_fcs[agent.name] = efc  # stores mean, shape: (num_features, )
    env.world = orig_world

    logging.info(f'Estimated feature counts: {estimated_fcs}')

    logging.info('----------------------------------------')

    # compute and plot difference between empirical (demo) and estimated (via MC) feature counts
    fc_diffs: Dict[str, float] = {}
    for agent in team:
        diff = np.abs(empirical_fcs[agent.name] - estimated_fcs[agent.name])
        norm_diff = np.sum(diff).item()
        agent_lrv = SearchRescueRewardVector(env, agent)
        plot_bar(pd.DataFrame(diff.reshape(1, -1), columns=agent_lrv.names),
                 'Feature Count Difference',
                 os.path.join(output_dir, f'fc-diff-{agent}.{args.img_format}'),
                 x_label='Reward Features', y_label='Abs. Difference')
        logging.info(f'Feature count difference for agent {agent.name}: {diff} (norm: {norm_diff:.2f})')
        fc_diffs[agent.name] = norm_diff

    plot_bar(fc_diffs, 'Feature Count Differences',
             os.path.join(output_dir, f'fc-diffs.{args.img_format}'),
             x_label='Agent', y_label='FC Diff.')

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)
    add_agent_arguments(parser)
    add_trajectory_arguments(parser)

    parser.add_argument('--traj-file', '-tf', type=str, required=True,
                        help='Path to the .pkl.gz file containing the demonstrated trajectories (possibly with model '
                             'inference) from which to compute the empirical feature counts.')

    args = parser.parse_args()

    main()
