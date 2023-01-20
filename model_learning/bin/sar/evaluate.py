import argparse
import logging
import os
from typing import List, Dict

from model_learning import TeamTrajectory
from model_learning.algorithms import ModelLearningResult
from model_learning.algorithms.max_entropy import THETA_STR
from model_learning.bin.sar import create_sar_world, add_common_arguments, add_agent_arguments, add_trajectory_arguments
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.trajectory import copy_world
from model_learning.util.cmd_line import save_args
from model_learning.util.io import create_clear_dir, load_object
from model_learning.util.logging import change_log_handler

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'
__description__ = 'Evaluates reward functions resulting from MIRL with ground-truth in the search-and-rescue domain.'


def compute_policy_divergence(expert_trajectories: List[TeamTrajectory], learner_trajectories: List[TeamTrajectory]):
    pass


def main():
    # checks results
    assert len(args.results_files) > 0, 'No  model learning results files were specified'
    for res_file in args.results_files:
        assert os.path.isfile(res_file), f'Could not find a model learning results file at {res_file}'

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    change_log_handler(os.path.join(output_dir, 'evaluate.log'), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    # loads MIRL results files for each agent
    logging.info('========================================')
    logging.info(f'Loading MIRL-ToM results from {len(args.results_files)} files...')
    results: Dict[str, ModelLearningResult] = {}
    for res_file in args.results_files:
        ag_results: ModelLearningResult = load_object(res_file)
        results[ag_results.agent] = ag_results
        logging.info(f'Loaded learned reward function for agent {ag_results.agent}: {ag_results.stats[THETA_STR]}')

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)

    # checks that we have results for all agents
    assert set(team_config.keys()) == set(results.keys()), \
        f'The model learning results for some agents were not provided, ' \
        f'expecting: {set(team_config.keys())}, provided: {set(results.keys())}'

    # generate expert and learner trajectories
    logging.info('========================================')
    logging.info(f'Generating {args.trajectories} trajectories of length {args.length} using the expert/GT team...')
    world = copy_world(env.world)  # make copy of world for learner team
    expert_trajectories: List[TeamTrajectory] = env.generate_team_trajectories(
        team,
        trajectory_length=args.length,
        n_trajectories=args.trajectories,
        horizon=args.horizon,
        selection=args.selection,
        processes=args.processes,
        threshold=args.prune,
        seed=args.seed)

    logging.info('========================================')
    logging.info(f'Generating {args.trajectories} trajectories of length {args.length} using the learner/IRL team...')

    # set old world back and modify agents' reward functions from IRL results
    env.world = world
    team = [world.agents[agent.name] for agent in team]
    for agent in team:
        agent_lrv = SearchRescueRewardVector(env, agent)
        agent_lrv.set_rewards(agent, results[agent.name].stats[THETA_STR])

    learner_trajectories: List[TeamTrajectory] = env.generate_team_trajectories(
        team,
        trajectory_length=args.length,
        n_trajectories=args.trajectories,
        horizon=args.horizon,
        selection=args.selection,
        processes=args.processes,
        threshold=args.prune,
        seed=args.seed)

    # TODO get "policies" from trajectories then compute divergence


    output_dir = args.output

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)
    add_agent_arguments(parser)
    add_trajectory_arguments(parser)

    parser.add_argument('--results-files', '-rf', nargs='+', type=str, required=True,
                        help='List of paths to the .pkl.gz files containing the model learning via MIRL-ToM results '
                             'for each agent.')

    args = parser.parse_args()

    main()
