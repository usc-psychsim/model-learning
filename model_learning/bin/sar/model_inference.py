from timeit import default_timer as timer
import plotly.graph_objs as go
import argparse
import logging
import os
import pandas as pd
import random
import string
import tqdm
from typing import List, Optional, Dict

from model_learning import TeamTrajectory, TeamStateActionModelDist, TeamModelDistTrajectory, ModelsDistributions
from model_learning.bin.sar import add_common_arguments, create_sar_world, create_observers, TeamConfig
from model_learning.trajectory import copy_world
from model_learning.util.cmd_line import save_args
from model_learning.util.io import load_object, create_clear_dir, save_object
from model_learning.util.logging import change_log_handler, MultiProcessLogger, create_mp_log_handler
from model_learning.util.mp import run_parallel
from model_learning.util.plot_new import plot_timeseries, dummy_plotly
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, VectorDistributionSet, isSpecialKey, makeTree, setToConstantMatrix, actionKey
from psychsim.world import World

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'
__description__ = 'Performs reward model inference in the search-and-rescue domain.' \
                  'First, it loads a set of team trajectories corresponding to observed behavior between two ' \
                  'collaborative agents. ' \
                  'An observer agent then maintains beliefs (a probability distribution over reward models) about ' \
                  'each agent, which are updated via PsychSim inference according to the agents\' actions  .' \
                  'Trajectories with model inference are saved into a .pkl file.'

# common params
MODEL_RATIONALITY = 1  # .5
MODEL_SELECTION = 'distribution'  # 'random'  # 'distribution'

TRAJ_NUM_COL = 'Trajectory'


def _update_state(state: VectorDistributionSet, new_state: VectorDistributionSet):
    """
    Updates a state based on the value sof another state (overrides values).
    :param VectorDistributionSet state: the state to be updated.
    :param VectorDistributionSet new_state: the state from which to get the values used to update the first state.
    """
    # TODO the update() method of PsychSim state appears not to be working
    for key in new_state.keys():
        if key in state and not isSpecialKey(key):
            val = new_state.certain[key] if new_state.keyMap[key] is None else new_state[key]
            state.join(key, val)


def _plot_model_inference(trajectories: List[TeamModelDistTrajectory],
                          role: str,
                          other_role: str,
                          models: List[str],
                          plots_dir: str,
                          img_format: str) -> go.Figure:
    # collect model distribution data from each step of each trajectory
    dfs: List[pd.DataFrame] = []
    for i, trajectory in enumerate(trajectories):
        traj_dist: Dict[str, List[float]] = {model: [] for model in models}
        for _, _, dist, _ in trajectory:
            for model, prob in dist[role][other_role].items():
                traj_dist[model.replace(f'{other_role}_', '')].append(prob)

        # create dataframe and rearrange columns
        df = pd.DataFrame(traj_dist)
        df[TRAJ_NUM_COL] = i
        df = df[[TRAJ_NUM_COL] + models]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # save plot to file
    return plot_timeseries(df, f'{role.title()}\'s Model Inference of {other_role.title()}',
                           os.path.join(plots_dir, f'{role.lower()}_{other_role.lower()}_inference.{img_format}'),
                           x_label='Timesteps', y_label='Model Likelihood', var_label='Model',
                           average=True, group_by=TRAJ_NUM_COL, y_min=0, y_max=1,
                           legend=dict(x=0.01, y=0.99, yanchor='top'))


def get_model_distributions(observers: Dict[str, Agent],
                            team_config: TeamConfig) -> ModelsDistributions:
    """
    Gets the distribution over models of other agents for all agents given the observers' current beliefs.
    :param dict[str, Agent] observers: the observer for each agent/role.
    :param TeamConfig team_config: the team configuration with references to the desired mental models.
    :return: a distribution over other agents' model names for each agent.
    """
    model_dists: ModelsDistributions = {}
    for role, observer in observers.items():
        world = observer.world

        # gets agent observer's beliefs about all other agents' models
        ag_model_dists: Dict[str, Distribution] = {}
        model_dists[role] = ag_model_dists

        for other, models in team_config[role].mental_models.items():
            # gets observer's current belief over the other agent's models
            dist = Distribution({f'{other}_{model}': 0 for model in models.keys()})
            ag_model_dists[other] = dist
            obs_dist = world.getFeature(modelKey(other), state=observer.getBelief(model=observer.get_true_model()))
            for model, prob in obs_dist.items():
                model = model.rstrip(string.digits)  # strip digits to get original model name
                dist[model] = prob
            dist.normalize()
    return model_dists


def team_trajectory_model_inference(world: World,
                                    team: List[Agent],
                                    trajectory: TeamTrajectory,
                                    observers: Dict[str, Agent],
                                    team_config: TeamConfig,
                                    threshold: Optional[float] = None,
                                    seed: int = 0,
                                    verbose: bool = False) -> TeamModelDistTrajectory:
    """
    Performs and records model inference given a team trajectory.
    :param World world: the PsychSim world used to perform the model inference.
    :param list[Agent] team: the list of agents over which to perform model inference.
    :param TeamTrajectory trajectory: the trajectory with the agent's actions used to perform model inference.
    :param dict[str, Agent] observers: the observer agent's containing the beliefs over models to be updated.
    :param TeamConfig team_config: the team configuration with references to the desired mental models.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: TeamModelDistTrajectory
    :return: a trajectory containing the state-team action-model distribution tuples.
    """
    random.seed(seed)

    # reset to initial state in the given trajectory
    init_state = trajectory[0].state
    world = copy_world(world)
    _update_state(world.state, init_state)
    team = [world.agents[agent.name] for agent in team]  # get new world's agents
    observers = {ag_name: world.agents[observer.name] for ag_name, observer in observers.items()}

    models_dists = get_model_distributions(observers, team_config)  # initial beliefs
    n_step = len(trajectory)
    team_trajectory: TeamModelDistTrajectory = []
    total = 0.
    for i in range(n_step):
        start = timer()
        # append state-action-models dists tuple
        state, team_action, prob = trajectory[i]
        team_trajectory.append(TeamStateActionModelDist(state, team_action, models_dists, prob))

        # synchronize all agents' beliefs with state in trajectory
        world.modelGC()
        for agent in team + list(observers.values()):
            for model in agent.models.values():
                _update_state(model['beliefs'], state)  # also reset agents' beliefs to match init state

        if i == n_step - 1:
            break

        # get teams' actions as a (stochastic) dynamics tree
        team_action = {ag: makeTree(Distribution(
            {setToConstantMatrix(actionKey(ag), world.value2float(actionKey(ag), action)): prob
             for action, prob in action_dist.items()}))
            for ag, action_dist in team_action.items()}
        world.step(actions=team_action, threshold=threshold)

        models_dists = get_model_distributions(observers, team_config)

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'[Seed {seed}] Step {i} took {step_time:.2f}')

    if verbose:
        logging.info(f'[Seed {seed}] Total time: {total:.2f}s')

    return team_trajectory


def team_trajectories_model_inference(world: World,
                                      team: List[Agent],
                                      trajectories: List[TeamTrajectory],
                                      observers: Dict[str, Agent],
                                      team_config: TeamConfig,
                                      threshold: Optional[float] = None,
                                      processes: Optional[int] = -1,
                                      seed: int = 0,
                                      verbose: bool = False) -> List[TeamModelDistTrajectory]:
    """
    Performs and records model inference given a set of team trajectories.
    :param World world: the PsychSim world used to perform the model inference.
    :param list[Agent] team: the list of agents over which to perform model inference.
    :param list[TeamTrajectory] trajectories: the trajectories with the agent's actions used to perform model inference.
    :param dict[str, Agent] observers: the observer agent's containing the beliefs over models to be updated.
    :param TeamConfig team_config: the team configuration with references to the desired mental models.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: list[TeamModelDistTrajectory]
    :return: a list of trajectories containing the state-team action-model distribution tuples.
    """
    args = [(world, team, trajectories[t], observers, team_config, threshold, seed + t, verbose)
            for t in range(len(trajectories))]
    team_trajectories_with_model_dist: List[TeamModelDistTrajectory] = \
        run_parallel(team_trajectory_model_inference, args, processes=processes, use_tqdm=True, mp_logging=verbose)
    return team_trajectories_with_model_dist


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
    team_model_dist_trajs = team_trajectories_model_inference(
        env.world, team, trajectories, observers, team_config,
        threshold=args.prune,
        processes=args.processes,
        seed=args.seed,
        verbose=logging.NOTSET < args.verbosity <= logging.INFO)

    output_dir = args.output

    # save all trajectories to pickle file
    file_path = os.path.join(output_dir, 'trajectories.pkl.gz')
    logging.info(f'Saving trajectories to {file_path}...')
    save_object(team_model_dist_trajs, file_path, compress_gzip=True)

    # TODO
    file_path = os.path.join(output_dir, 'trajectories.pkl.gz')
    trajectories: List[TeamModelDistTrajectory] = load_object(file_path)

    dummy_plotly()  # to clear plotly import message

    plots_dir = os.path.join(output_dir, 'model_inference')
    create_clear_dir(plots_dir, clear=False)
    logging.info(f'Generating model inference plots, saving to {plots_dir}...')
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        for other_role, models in ag_conf.mental_models.items():
            _plot_model_inference(trajectories, role, other_role, list(models), plots_dir, args.img_format)

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
