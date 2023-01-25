import argparse
import logging
import numpy as np
import os
import pandas as pd
import tqdm
from typing import List, Dict, Callable

from model_learning import StateActionProbTrajectory, State
from model_learning.algorithms import ModelLearningResult
from model_learning.algorithms.max_entropy import THETA_STR
from model_learning.bin.sar import create_sar_world, add_common_arguments, add_agent_arguments, add_trajectory_arguments
from model_learning.evaluation.metrics import policy_divergence, policy_mismatch_prob
from model_learning.features.counting import empirical_feature_counts, estimate_feature_counts
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.planning import get_states_policy
from model_learning.trajectory import copy_world
from model_learning.util.cmd_line import save_args
from model_learning.util.io import create_clear_dir, load_object
from model_learning.util.logging import change_log_handler
from model_learning.util.plot import dummy_plotly, plot_bar
from psychsim.agent import Agent
from psychsim.probability import Distribution

__author__ = 'Pedro Sequeira, Haochen Wu'
__email__ = 'pedrodbs@gmail.com, hcaawu@gmail.com'
__description__ = 'Evaluates reward functions resulting from MIRL with ground-truth in the search-and-rescue domain.'


def _get_team_policy(states: List[State], team: List[Agent], label: str) -> Dict[str, List[Distribution]]:
    logging.info('========================================')
    logging.info(f'Computing the state policies for the {label} team...')
    team_policy: Dict[str, List[Distribution]] = {}
    for agent in team:
        logging.info(f'\tComputing the state policies for agent {agent.name}...')
        team_policy[agent.name] = get_states_policy(
            agent, states, horizon=args.horizon, selection='distribution',
            processes=args.processes, seed=args.seed, use_tqdm=True)
    return team_policy


def _compare_policies(expert_policy: Dict[str, List[Distribution]],
                      learner_policy: Dict[str, List[Distribution]],
                      metric: Callable[[List[Distribution], List[Distribution]], float],
                      title: str, output_dir: str, y_label: str):
    # compute metric comparing the policies and create a bar plot with results
    result = {agent: metric(expert_policy[agent], learner_policy[agent]) for agent in expert_policy.keys()}
    logging.info(f'{title}: {result}')
    plot_bar(result, title,
             os.path.join(output_dir, f'{title.lower().replace(" ", "-")}.{args.img_format}'),
             x_label='Agent', y_label=y_label, y_min=0, y_max=1)


def _plot_feature_counts(fcs: np.ndarray, probs: np.ndarray, title: str, col_names: List[str], output_dir: str):
    efc = np.sum(fcs * probs, axis=1)  # shape: (num_traj, num_features)
    plot_bar(pd.DataFrame(efc, columns=col_names), title,
             os.path.join(output_dir, f'{title.lower().replace(" ", "-")}.{args.img_format}'),
             x_label='Reward Features', y_label='Mean Count')


def main():
    # checks files
    assert os.path.isfile(args.traj_file), f'Could not find the team trajectories file at {args.traj_file}'
    assert len(args.results) > 0, 'No  model learning results files were specified'
    for res_file in args.results:
        assert os.path.isfile(res_file), f'Could not find a model learning results file at {res_file}'

    np.set_printoptions(precision=2)

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    change_log_handler(os.path.join(output_dir, 'evaluate.log'), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    # loads MIRL results files for each agent
    logging.info('========================================')
    logging.info(f'Loading MIRL-ToM results from {len(args.results)} files...')
    results: Dict[str, ModelLearningResult] = {}
    for res_file in args.results:
        ag_results: ModelLearningResult = load_object(res_file)
        results[ag_results.agent] = ag_results
        logging.info(f'Loaded learned reward function for agent {ag_results.agent}: {ag_results.stats[THETA_STR]}')

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)
    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    # checks that we have results for all agents
    assert set(team_config.keys()) == set(results.keys()), \
        f'The model learning results for some agents were not provided, ' \
        f'expecting: {set(team_config.keys())}, provided: {set(results.keys())}'

    # load trajectories
    logging.info('========================================')
    logging.info(f'Loading team trajectories from {args.traj_file}...')
    expert_trajectories: List[StateActionProbTrajectory] = load_object(args.traj_file)
    logging.info(f'Loaded {len(expert_trajectories)} trajectories of length {len(expert_trajectories[0])}')

    # collect all states from the trajectories
    states: List[State] = [sap.state for trajectory in expert_trajectories for sap in trajectory]
    logging.info(f'Collected a total of {len(states)} states from the loaded trajectories')

    # TODO
    states = states[:4]

    logging.info('========================================')
    logging.info('Computing policies for expert and learner teams...')

    # compute expert and learner policies from the collected states
    world = copy_world(env.world)
    learner_team = [world.agents[agent.name] for agent in team]  # make copy of world for learner team
    expert_policy = _get_team_policy(states, team, 'experts / ground-truth')

    logging.info('----------------------------------------')

    # modify agents' reward functions from IRL results
    for agent in learner_team:
        agent_lrv = SearchRescueRewardVector(env, agent)
        agent_lrv.set_rewards(agent, results[agent.name].stats[THETA_STR])
    learner_policy = _get_team_policy(states, learner_team, 'learners / IRL')

    dummy_plotly()  # to clear plotly import message
    output_dir = args.output

    logging.info('========================================')
    logging.info('Computing policies divergence and mismatch...')
    _compare_policies(expert_policy, learner_policy,
                      metric=policy_divergence, title='Policy Divergence', output_dir=output_dir,
                      y_label='JSD')

    _compare_policies(expert_policy, learner_policy,
                      metric=policy_mismatch_prob, title='Policy Mismatch', output_dir=output_dir,
                      y_label='Mismatch Prob.')

    logging.info('========================================')
    logging.info('Comparing feature counts between expert and learner teams policies...')

    logging.info(f'Computing the empirical feature counts for the expert / ground-truth team '
                 f'for {len(expert_trajectories)} trajectories...')
    empirical_fcs: Dict[str, np.ndarray] = {}
    for agent in tqdm.tqdm(team):
        agent_lrv = SearchRescueRewardVector(env, agent)
        feature_func = lambda s: agent_lrv.get_values(s)
        fcs, probs = empirical_feature_counts(expert_trajectories, feature_func, unweighted=True)
        _plot_feature_counts(fcs, probs, title=f'Expert {agent.name} Empirical FCs',
                             col_names=agent_lrv.names, output_dir=output_dir)
        empirical_fcs[agent.name] = np.mean(np.sum(fcs * probs, axis=1), axis=0)  # stores mean, shape: (num_features, )
    logging.info(f'Empirical feature counts: {empirical_fcs}')

    logging.info('----------------------------------------')

    traj_len = len(expert_trajectories[0])
    logging.info(f'Estimating feature counts for the learner/IRL team for {len(expert_trajectories)} trajectories '
                 f'using {args.trajectories} Monte-Carlo trajectories of length {traj_len}...')

    # set old world back and estimate feature counts for each agent
    init_states: List[State] = [trajectory[0].state for trajectory in expert_trajectories]
    estimated_fcs: Dict[str, np.ndarray] = {}
    for agent in learner_team:
        logging.info(f'\tEstimating feature counts for agent {agent.name}...')
        agent_lrv = SearchRescueRewardVector(env, agent)
        feature_func = lambda s: agent_lrv.get_values(s)
        fcs, probs = estimate_feature_counts(
            agent, init_states, traj_len,
            feature_func=feature_func,
            exact=False,
            num_mc_trajectories=args.trajectories,
            horizon=args.horizon,
            threshold=args.prune,
            unweighted=True,
            processes=args.processes,
            seed=args.seed,
            verbose=False,
            use_tqdm=True)
        _plot_feature_counts(fcs, probs, title=f'Learner {agent.name} Estimated FCs',
                             col_names=agent_lrv.names, output_dir=output_dir)
        estimated_fcs[agent.name] = np.mean(np.sum(fcs * probs, axis=1), axis=0)  # stores mean, shape: (num_features, )
    logging.info(f'Estimated feature counts: {estimated_fcs}')

    logging.info('----------------------------------------')

    # compute and plot difference between empirical (GT) and estimated (IRL) feature counts
    fc_diffs: Dict[str, float] = {}
    for agent in empirical_fcs.keys():
        diff = np.abs(empirical_fcs[agent] - estimated_fcs[agent])
        norm_diff = np.sum(diff).item()
        agent_lrv = SearchRescueRewardVector(env, world.agents[agent])
        plot_bar(pd.DataFrame(diff.reshape(1, -1), columns=agent_lrv.names),
                 'Feature Count Difference',
                 os.path.join(output_dir, f'fc-diff-{agent}.{args.img_format}'),
                 x_label='Reward Features', y_label='Abs. Difference')
        logging.info(f'Feature count difference for agent {agent}: {diff} (norm: {norm_diff:.2f})')
        fc_diffs[agent] = norm_diff

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
                        help='Path to the .pkl.gz file containing the expert trajectories used to compute the metrics.')
    parser.add_argument('--results', '-rf', nargs='+', type=str, required=True,
                        help='List of paths to the .pkl.gz files containing the model learning via MIRL-ToM results '
                             'for each agent.')

    args = parser.parse_args()

    main()
