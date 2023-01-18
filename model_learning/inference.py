import copy
from timeit import default_timer as timer

import itertools as it
import logging
import numpy as np
import pandas as pd
import random
import string
from plotly import graph_objs as go
from typing import List, Optional, Dict

from model_learning import Trajectory, TeamModelsDistributions, TeamModelDistTrajectory, TeamTrajectory, \
    TeamStateActionModelDist, ModelDistTrajectory, StateActionModelDist
from model_learning.trajectory import copy_world, update_state
from model_learning.util.mp import run_parallel
from model_learning.util.plot import plot_timeseries
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, makeTree, setToConstantMatrix, actionKey
from psychsim.world import World

__author__ = 'Pedro Sequeira, Haochen Wu'
__email__ = 'pedrodbs@gmail.com, hcaawu@gmail.com'

MODEL_SELECTION = 'distribution'  # needed to include al possible actions and avoid impossible observations
MODEL_RATIONALITY = 1  # for smoother modeling
MODEL_HORIZON = 1  # TODO HORIZON
OBSERVER_NAME = 'Observer'
OBSERVER_MODEL_NAME = 'OBSERVER'
TRAJ_NUM_COL = 'Trajectory'


def model_inference(trajectory: Trajectory,
                    models: List[str],
                    agent: Agent,
                    observer: Agent,
                    features: Optional[List[str]] = None,
                    verbose: bool = True) -> ModelDistTrajectory:
    """
    Updates and tracks the evolution of an observer agent's posterior distribution over the possible reward models of
    an "actor" agent.
    :param Trajectory trajectory: the trajectory of (world, action) pairs containing the "actor"
    agent's actions. It is assumed that agents `agent` and `observer` are present in each "world" of the trajectory so
    that we can recover the latter's distribution over reward models.
    :param list[str] models: a list of model names of the `agent` agent containing the different reward functions.
    :param Agent agent: the "actor" agent performing actions in the trajectory and containing the reward models.
    :param Agent observer: the observer agent modeling the actor whose beliefs we want to track.
    :param list[str] features: a list of world features to print at each timestep, if `verbose` is `True`.
    :param bool verbose: whether to print info messages to the logger at each timestep.
    :rtype: np.ndarray
    :return: a trajectory containing the state-action-model distribution tuples.
    """
    # gets the reward models the posterior over which we want to update and track
    rwd_models = [agent.getReward(model) for model in models]

    models_dist = ModelDistTrajectory()
    for t, sa in enumerate(trajectory):
        world, action, prob = sa

        if verbose and features is not None:
            logging.info('===============================================')
            logging.info(f'{t}.\t({",".join([str(world.getFeature(f, unique=True)) for f in features])})')

        # get actual agents from the trajectory world
        observer = world.agents[observer.name]
        agent = world.agents[agent.name]

        # get observer's beliefs about agent
        beliefs = observer.getBelief(world.state)
        assert len(beliefs) == 1  # Because we are dealing with a known-identity agent
        belief = next(iter(beliefs.values()))

        # update posterior distribution over models
        model_dist = world.getFeature(modelKey(agent.name), belief)
        dist = {}
        for model in model_dist.domain():
            rwd_model = agent.getReward(model)
            dist[rwd_models.index(rwd_model)] = model_dist[model]
        models_dist.append(StateActionModelDist(world.state, action, Distribution(dist), sa.prob))

        if verbose:
            logging.info('Observer models agent as:')
            logging.info(model_dist)
            logging.info(action if len(action) > 1 else action.first())

    return models_dist


def plot_model_inference(trajectories: List[ModelDistTrajectory], agent: str, output_img: str) -> go.Figure:
    """
    Creates plots that show the mean curves of an agent models' likelihoods over time given a set of trajectories.
    :param list[ModelDistTrajectory] trajectories: a list of trajectories with distributions over the agent's
    models for each timestep.
    :param str agent:the name of the modelee, i.e., the agent being modeled.
    :param str output_img: the path to the file in which to save the plot.
    :return:
    """
    models = [model.replace(f'{agent}_', '') for model in trajectories[0][0].models_dist.keys()]

    # collect model distribution data from each step of each trajectory
    dfs: List[pd.DataFrame] = []
    for i, trajectory in enumerate(trajectories):
        traj_dist: Dict[str, List[float]] = {model: [] for model in models}
        for _, _, dist, _ in trajectory:
            for model, prob in dist.items():
                traj_dist[model.replace(f'{agent}_', '')].append(prob)

        # create dataframe and rearrange columns
        df = pd.DataFrame(traj_dist)
        df[TRAJ_NUM_COL] = i
        df = df[[TRAJ_NUM_COL] + models]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # save plot to file
    return plot_timeseries(df, f'{agent.title()} Model Inference', output_img,
                           x_label='Timesteps', y_label='Model Likelihood', var_label='Model',
                           average=True, group_by=TRAJ_NUM_COL, y_min=0, y_max=1,
                           legend=dict(x=0.01, y=0.99, yanchor='top'))


def team_model_inference(world: World,
                         team: List[Agent],
                         trajectory: TeamTrajectory,
                         observers: Dict[str, Agent],
                         models_dists: TeamModelsDistributions,
                         threshold: Optional[float] = None,
                         seed: int = 0,
                         verbose: bool = False) -> TeamModelDistTrajectory:
    """
    Performs and records model inference given a team trajectory.
    :param World world: the PsychSim world used to perform the model inference.
    :param list[Agent] team: the list of agents over which to perform model inference.
    :param TeamTrajectory trajectory: the trajectory with the agent's actions used to perform model inference.
    :param dict[str, Agent] observers: the observer agent's containing the beliefs over models to be updated.
    :param TeamModelsDistributions models_dists: the models distributions with references to the desired mental models.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: TeamModelDistTrajectory
    :return: a trajectory containing the state-team action-models distributions tuples.
    """
    random.seed(seed)

    # reset to initial state in the given trajectory
    init_state = trajectory[0].state
    world = copy_world(world)
    update_state(world.state, init_state)
    team = [world.agents[agent.name] for agent in team]  # get new world's agents
    observers = {ag_name: world.agents[observer.name] for ag_name, observer in observers.items()}

    models_dists = get_model_distributions(observers, models_dists)  # gets initial observers' beliefs
    n_steps = len(trajectory)
    team_trajectory: TeamModelDistTrajectory = []
    total = 0.
    for i in range(n_steps):
        start = timer()
        # append state-action-models dists tuple
        state, team_action, prob = trajectory[i]
        team_trajectory.append(TeamStateActionModelDist(state, team_action, models_dists, prob))

        if i == n_steps - 1:
            break  # avoid last world step since it's not going to be included in the trajectory

        # synchronize all agents' beliefs with state in trajectory
        world.modelGC()
        for agent in team + list(observers.values()):
            for model in agent.models.values():
                update_state(model['beliefs'], state)  # also reset agents' beliefs to match init state

        # get teams' actions as a (stochastic) dynamics tree
        team_action = {ag: makeTree(Distribution(
            {setToConstantMatrix(actionKey(ag), world.value2float(actionKey(ag), action)): prob
             for action, prob in action_dist.items()}))
            for ag, action_dist in team_action.items()}
        world.step(actions=team_action, threshold=threshold)

        models_dists = get_model_distributions(observers, models_dists)

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
                                      models_dists: TeamModelsDistributions,
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
    :param TeamModelsDistributions models_dists: the models distributions with references to the desired mental models.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: list[TeamModelDistTrajectory]
    :return: a list of trajectories containing the state-team action-model distribution tuples.
    """
    args = [(world, team, trajectories[t], observers, models_dists, threshold, seed + t, verbose)
            for t in range(len(trajectories))]
    team_trajectories_with_model_dist: List[TeamModelDistTrajectory] = \
        run_parallel(team_model_inference, args, processes=processes, use_tqdm=True, mp_logging=verbose)
    return team_trajectories_with_model_dist


def create_inference_observers(world: World,
                               init_models_dists: TeamModelsDistributions,
                               belief_threshold: float) -> Dict[str, Agent]:
    """
    Creates model observers that will maintain a probability distribution over agent models (beliefs) as the agents
    act in the environment. One observer will be created per agent and the beliefs will be created and initialized
    according to the provided initial distribution.
    :param World world: the PsychSim world in which to create the observer agents.
    :param TeamModelsDistributions init_models_dists: the initial distribution over other agents' models for each agent.
    :param float belief_threshold: beliefs with probability below this will be pruned.
    :rtype: dict[str, Agent]
    :return: a dictionary containing the observer agent created for each agent.
    """
    # create observer agents
    observers: Dict[str, Agent] = {}
    for ag in init_models_dists.keys():
        observer = world.addAgent(f'{OBSERVER_NAME}_{ag}')
        observers[ag] = observer
        observer.belief_threshold = belief_threshold

    # make observers ignore each other
    for obs1, obs2 in it.combinations(list(observers.keys()), 2):
        observers[obs1].ignore(observers[obs2].name)
        observers[obs2].ignore(observers[obs1].name)

    # mental modeling
    for ag, mental_models in init_models_dists.items():
        observer = observers[ag]
        observer.set_observations()  # observer does not observe other agents' true models

        # set distribution over other agents' models for this observer
        for other_ag, dist in mental_models.items():
            dist.normalize()  # just in case
            world.setMentalModel(observer.name, other_ag, dist, model=None)

        # create agent model just for observer (needs to have special action selection)
        agent: Agent = world.agents[ag]
        model = f'{agent.name}_{OBSERVER_MODEL_NAME}'
        agent.addModel(model,
                       parent=agent.get_true_model(),
                       rationality=MODEL_RATIONALITY,
                       selection=MODEL_SELECTION,
                       horizon=MODEL_HORIZON)
        agent.setAttribute('R', copy.copy(agent.getAttribute('R', model=agent.get_true_model())), model=model)
        world.setMentalModel(observer.name, ag, model, model=None)

        # make all agents and their models ignore this observer
        for _ag in init_models_dists.keys():
            agent = world.agents[_ag]
            for model in agent.models.keys():
                agent.ignore(observer.name, model=model)

    return observers


def get_model_distributions(observers: Dict[str, Agent],
                            prev_models_dists: TeamModelsDistributions) -> TeamModelsDistributions:
    """
    Gets the distribution over models of other agents for all agents given the observers' current beliefs.
    :param dict[str, Agent] observers: the PsychSim observer for each agent/role.
    :param TeamModelsDistributions prev_models_dists: the previous distribution with references to the desired mental models.
    :return: a distribution over other agents' model names for each agent.
    """
    models_dists: TeamModelsDistributions = {}
    for agent, observer in observers.items():
        world = observer.world

        # gets agent observer's beliefs about all other agents' models
        ag_model_dists: Dict[str, Distribution] = {}
        models_dists[agent] = ag_model_dists

        for other, models in prev_models_dists[agent].items():
            # gets observer's current belief over the other agent's models
            dist = Distribution({model: 0 for model in models.keys()})
            ag_model_dists[other] = dist
            obs_dist = world.getFeature(modelKey(other), state=observer.getBelief(model=observer.get_true_model()))
            for model, prob in obs_dist.items():
                model = model.rstrip(string.digits)  # strip digits to get original model name
                dist[model] = prob
            dist.normalize()
    return models_dists


def plot_team_model_inference(trajectories: List[TeamModelDistTrajectory],
                              agent: str,
                              other_ag: str,
                              output_img: str) -> go.Figure:
    """
    Creates plots that show the mean curves of an agent's mental models' likelihoods over time given a set of
    trajectories.
    :param list[TeamModelDistTrajectory] trajectories: a list of trajectories with distributions over another agent's
    models for each timestep.
    :param str agent: the name of the modeler, i.e., the agent modeling the other agent.
    :param str other_ag: the name of the modelee, i.e., the agent being modeled.
    :param str output_img: the path to the file in which to save the plot.
    :return:
    """

    models = [model.replace(f'{other_ag}_', '') for model in trajectories[0][0].models_dists[agent][other_ag].keys()]

    # collect model distribution data from each step of each trajectory
    dfs: List[pd.DataFrame] = []
    for i, trajectory in enumerate(trajectories):
        traj_dist: Dict[str, List[float]] = {model: [] for model in models}
        for _, _, dist, _ in trajectory:
            for model, prob in dist[agent][other_ag].items():
                traj_dist[model.replace(f'{other_ag}_', '')].append(prob)

        # create dataframe and rearrange columns
        df = pd.DataFrame(traj_dist)
        df[TRAJ_NUM_COL] = i
        df = df[[TRAJ_NUM_COL] + models]
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # save plot to file
    return plot_timeseries(df, f'{agent.title()}\'s Model Inference of {other_ag.title()}', output_img,
                           x_label='Timesteps', y_label='Model Likelihood', var_label='Model',
                           average=True, group_by=TRAJ_NUM_COL, y_min=0, y_max=1,
                           legend=dict(x=0.01, y=0.99, yanchor='top'))
