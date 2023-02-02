import argparse
import logging
import os
from typing import Optional, List

from model_learning import TeamModelDistTrajectory
from model_learning.algorithms.mirl_tom import MIRLToM
from model_learning.bin.sar import add_common_arguments, create_sar_world, add_agent_arguments, \
    setup_modeling_agent
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.util.cmd_line import save_args, str2bool
from model_learning.util.io import create_clear_dir, load_object, save_object
from model_learning.util.logging import change_log_handler, MultiProcessLogger
from model_learning.util.plot import dummy_plotly
from psychsim.agent import Agent
from psychsim.reward import null_reward

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'
__description__ = 'Performs Multiagent IRL (reward model learning) with ToM in the Search-and-Rescue World using ' \
                  'MaxEnt IRL.'

# default learning params
NORM_THETA = True
LEARNING_RATE = 5e-2
MAX_EPOCHS = 30
THRESHOLD = 5e-3
DECREASE_RATE = True
EXACT = False
NUM_MC_TRAJECTORIES = 16  # 10
HORIZON = 2

RESULTS_FILE = 'results.pkl.gz'


def main():
    assert os.path.isfile(args.traj_file), f'Could not find the team trajectories file at {args.traj_file}'

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    save_args(args, os.path.join(output_dir, 'args.json'))
    log_file = os.path.join(output_dir, 'mirl_tom.log')
    mp_log: Optional[MultiProcessLogger] = None
    if args.processes > 1 or args.processes <= -1:
        mp_log = MultiProcessLogger(log_file, level=args.verbosity)
    else:
        change_log_handler(log_file, level=args.verbosity)

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)

    # load trajectories
    logging.info('========================================')
    logging.info(f'Loading team trajectories from {args.traj_file}...')
    trajectories: List[TeamModelDistTrajectory] = load_object(args.traj_file)
    logging.info(f'Loaded {len(trajectories)} team trajectories of length {len(trajectories[0])}')

    # checks trajectories against team config
    assert all(role in team_config for role in trajectories[0][0].models_dists.keys()), \
        f'Mismatch encountered between agents in the trajectories and the ones defined ' \
        f'in the configuration: {list(team_config.keys())}'

    # checks agent
    learner_ag: str = args.agent
    assert learner_ag in team_config, \
        f'The provided agent role: {learner_ag} is not defined in the configuration:  {list(team_config.keys())}'

    # checks model distributions in trajectories (considered consistent)
    models_dists = trajectories[0][0].models_dists
    for other, models in team_config[learner_ag].mental_models.items():
        assert other in models_dists[learner_ag], \
            f'Agent {other} not present in trajectories\' mental model distribution ' \
            f'of agent {learner_ag}: {list(models_dists.keys())}'
        for model in models.keys():
            assert f'{other}_{model}' in models_dists[learner_ag][other], \
                f'Model {model} of agent {other} not present in trajectories\' mental model distribution ' \
                f'of agent {learner_ag}: {list(models_dists[other].keys())}'

    # creates mental models for agent and set same agent params to models since they will only be used to simulate the
    # other agents' actions, on which the learner agent's actions will be then conditioned
    logging.info(f'Setting up agent for IRL: {learner_ag}')
    setup_modeling_agent(learner_ag, env, team_config, profiles,
                         rationality=args.rationality,
                         horizon=args.horizon,
                         selection=args.selection)

    # set learner's reward to 0 to ensure no GT reward leakage from team config file
    learner_ag: Agent = env.world.agents[learner_ag]
    learner_ag.setReward(null_reward(learner_ag.name))

    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    logging.info('========================================')
    logging.info(f'Performing decentralized IRL for agent: {learner_ag}...')

    learner_rwd_vector = SearchRescueRewardVector(env, learner_ag)
    alg = MIRLToM(
        'max-ent', learner_ag, learner_rwd_vector,
        normalize_weights=args.normalize,
        learning_rate=args.learning_rate,
        decrease_rate=args.decrease_rate,
        max_epochs=args.epochs,
        diff_threshold=args.threshold,
        exact=args.exact,
        num_mc_trajectories=args.monte_carlo,
        prune_threshold=args.prune,
        horizon=args.horizon,
        processes=args.processes,
        seed=args.seed)

    results = alg.learn(trajectories, verbose=args.verbosity <= logging.INFO)

    logging.info('========================================')
    logging.info(f'Saving results to {output_dir}...')

    dummy_plotly()  # to clear plotly import message

    # creates and saves plots with IRL stats
    results_dir = os.path.join(output_dir, 'results')
    create_clear_dir(results_dir, clear=False)
    logging.info(f'Saving IRL stats to {results_dir}...')
    alg.save_results(results, results_dir, img_format=args.img_format)

    # save results to pickle file
    file_path = os.path.join(output_dir, RESULTS_FILE)
    logging.info(f'Saving results to {file_path}...')
    save_object(results, file_path, compress_gzip=True)

    logging.info('Done!')

    if mp_log is not None:
        mp_log.close()  # close multiprocess logger


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)
    add_agent_arguments(parser)

    parser.add_argument('--traj-file', '-tf', type=str, required=True,
                        help='Path to the .pkl.gz file containing the team trajectories with mental model inference '
                             'with which to perform decentralized Multiagent IRL.')
    parser.add_argument('--agent', '-a', type=str, required=True,
                        help='The name or role associated with the "learner" agent, for which to perform '
                             'decentralized Multiagent IRL.')

    parser.add_argument('--learning-rate', '-lr', type=float, default=LEARNING_RATE,
                        help='IRL gradient descent learning/update rate.')
    parser.add_argument('--decrease-rate', '-dr', type=str2bool, default=DECREASE_RATE,
                        help='Whether to exponentially decrease the IRL learning rate over time.')
    parser.add_argument('--normalize', '-nw', type=str2bool, default=NORM_THETA,
                        help='Whether to normalize reward weights at each step of IRL.')
    parser.add_argument('--epochs', '-me', type=int, default=MAX_EPOCHS,
                        help='Maximum number of IRL gradient descent steps.')
    parser.add_argument('--threshold', '-dt', type=float, default=THRESHOLD,
                        help='IRL termination threshold for the weight vector difference.')
    parser.add_argument('--exact', '-ex', type=str2bool, default=EXACT,
                        help='Whether the computation of the MaxEnt IRL distribution over paths should be exact '
                             '(expand all stochastic branches) or not. If False, Monte Carlo sample trajectories '
                             'will be generated to estimate the feature counts.')
    parser.add_argument('--monte-carlo', '-mc', type=int, default=NUM_MC_TRAJECTORIES,
                        help='Number of Monte Carlo trajectories to be sampled during IRL if `exact=False`.')

    args = parser.parse_args()

    main()
