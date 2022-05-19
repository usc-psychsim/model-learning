import copy
import random
import logging
import numpy as np
from timeit import default_timer as timer
from typing import List, Tuple, Dict, Any, Optional, Literal
from psychsim.agent import Agent
from psychsim.world import World
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, turnKey, VectorDistributionSet
from model_learning.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# types
State = VectorDistributionSet
Trajectory = List[Tuple[World, Distribution]]  # list of world-action (distribution) pairs

TOP_LEVEL_STR = 'top_level'


def copy_world(world: World) -> World:
    """
    Creates a copy of the given world. This implementation clones the world's state and all agents so that the dynamic
    world elements are "frozen" in time.
    :param World world: the original world to be copied.
    :rtype: World
    :return: a semi-hard copy of the given world.
    """
    new_world = copy.copy(world)
    new_world.state = copy.deepcopy(world.state)
    new_world.agents = copy.copy(new_world.agents)
    for name, agent in world.agents.items():
        # clones agent with exception of world
        agent.world = None
        # TODO tentative
        new_agent = copy.copy(agent)
        new_agent.models = agent.models.copy()
        new_agent.modelList = agent.modelList.copy()
        new_world.agents[name] = new_agent
        new_agent.world = new_world  # assigns cloned world to cloned agent
        agent.world = world  # puts original world back to old agent
    return new_world


def generate_trajectory(agent: Agent,
                        trajectory_length: int,
                        init_feats: Optional[Dict[str, Any]] = None,
                        model: Optional[str] = None,
                        select: Optional[bool] = None,
                        horizon: Optional[int] = None,
                        selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                        threshold: Optional[float] = None,
                        seed: int = 0,
                        verbose: bool = False) -> Trajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(agent.world)
    random.seed(seed)
    rng = random.Random(seed)

    # generate or select initial state features
    if init_feats is not None:
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)

    # for each step, takes action and registers state-action pairs
    trajectory: Trajectory = []
    total = 0
    if model is not None:
        world.setFeature(modelKey(agent.name), model)
    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        turn = world.getFeature(turnKey(agent.name), unique=True)
        while turn != 0:
            world.step()
            turn = world.getFeature(turnKey(agent.name), unique=True)

        # steps the world, gets the agent's action
        prev_world = copy_world(world)
        world.step(select=select, horizon=horizon, tiebreak=selection, threshold=threshold)
        action = world.getAction(agent.name)
        trajectory.append((prev_world, action))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}s (action: {action if len(action) > 1 else action.first()})')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_trajectories(agent: Agent,
                          n_trajectories: int,
                          trajectory_length: int,
                          init_feats: Optional[Dict[str, Any]] = None,
                          model: Optional[str] = None,
                          select: Optional[bool] = None,
                          horizon: Optional[int] = None,
                          selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                          threshold: Optional[float] = None,
                          processes: int = -1,
                          seed: int = 0,
                          verbose: bool = False,
                          use_tqdm: bool = True) -> List[Trajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    world = agent.world
    for feature in init_feats:
        assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(agent, trajectory_length, init_feats, model, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[Trajectory] = run_parallel(generate_trajectory, args, processes=processes, use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def sample_random_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int,
                                   with_replacement: bool = False,
                                   seed: int = 0) -> List[Trajectory]:
    """
    Randomly samples sub-trajectories from a given trajectory.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :param bool with_replacement: whether to allow repeated sub-trajectories to be sampled.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert with_replacement or len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    # samples randomly
    rng = random.Random(seed)
    idxs = list(range(len(trajectory) - trajectory_length + 1))
    idxs = rng.choices(idxs, k=n_trajectories) if with_replacement else rng.sample(idxs, n_trajectories)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def sample_spread_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int) -> List[Trajectory]:
    """
    Samples sub-trajectories from a given trajectory as spread in time as possible.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    idxs = np.asarray(np.arange(0, len(trajectory) - trajectory_length + 1,
                                (len(trajectory) - trajectory_length) / max(1, n_trajectories - 1)), dtype=int)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def log_trajectories(trajectories: List[Trajectory], features: List[str]):
    """
    Prints the given trajectories to the log at the info level.
    :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories to save, containing
    several sequences of state-action pairs.
    :param list[str] features: the state features to be printed at each step, representing the state of interest.
    """
    if len(trajectories) == 0 or len(trajectories[0]) == 0:
        return

    for i, trajectory in enumerate(trajectories):
        logging.info('-------------------------------------------')
        logging.info(f'Trajectory {i}:')
        for t, sa in enumerate(trajectory):
            world, action = sa
            feat_values = [str(world.getFeature(feat, unique=True)) for feat in features]
            logging.info(f'{t}:\t({", ".join(feat_values)}) -> {action if len(action) > 1 else action.first()}')
