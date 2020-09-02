import copy
import random
import logging
from timeit import default_timer as timer
from typing import Callable
from psychsim.agent import Agent
from psychsim.world import World
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, turnKey, actionKey, VectorDistributionSet
from model_learning.util.multiprocessing import get_pool_and_map

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

TOP_LEVEL_STR = 'top_level'


def copy_world(world):
    """
    Creates a copy of the given world. This implementation clones the world's state and all agents' models so that
    the dynamic world elements are maintained in this timestep copy.
    :param World world: the original world to be copied.
    :rtype: World
    :return: a semi-hard copy of the given world.
    """
    new_world = copy.copy(world)
    new_world.state = copy.deepcopy(world.state)
    for name, agent in world.agents.items():
        new_world.agents[name].modelList = copy.deepcopy(agent.modelList)
        new_world.agents[name].models = copy.deepcopy(agent.models)
    return new_world


def get_agent_action(agent, state):
    """
    Gets the distribution over actions that the agent executed which resulted in the given state.
    :param Agent agent: the agent whose action(s) we want to retrieve.
    :param VectorDistributionSet state: the state resulting from the agent's action(s).
    :rtype: Distribution
    :return: the action(s) that the agent executed.
    """
    return agent.world.getFeature(actionKey(agent.name), state)


def generate_trajectory(agent, trajectory_length, features=None, init_feats=None,
                        model=None, horizon=None, selection=None, threshold=None,
                        seed=0, verbose=None):
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param list[str] features: the list of relevant features (keys) for which to initialize trajectories. If `None`,
    the trajectory starts from the world's current state.
    :param list[list] init_feats: a list of initial feature values from which to randomly initialize the
    trajectory. Each item is a list of feature values in the same order of `features`. If `None`, then features will
    be initialized uniformly at random according to their domain.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool or Callable verbose: whether to show information at each timestep during trajectory generation.
    :rtype: list[tuple[World, Distribution]]
    :return: a trajectory containing a list of state-action pairs.
    :return:
    """
    world = agent.world
    random.seed(seed)
    rng = random.Random(seed)

    # generate or select initial state features
    if features is not None:
        if init_feats is None:
            init_values = [get_random_value(world, feature, rng) for feature in features]
        else:
            init_values = rng.choice(init_feats)

        # set features' initial value
        for feature, init_value in zip(features, init_values):
            world.setFeature(feature, init_value)

    # for each step, takes action and registers state-action pairs
    trajectory = []
    total = 0
    prev_model = world.getFeature(modelKey(agent.name))
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
        world.step(select=True, horizon=horizon, tiebreak=selection, threshold=threshold, debug={TOP_LEVEL_STR: True})
        action = get_agent_action(agent, world.state)
        trajectory.append((prev_world, action))

        step_time = timer() - start
        total += step_time
        if verbose is not None:
            logging.info('Step {} took {:.2f}s (action: {})'.format(
                i, step_time, action if len(action) > 1 else action.first()))
            verbose()

    # puts back agent model
    if model is not None:
        world.setFeature(modelKey(agent.name), prev_model)

    if verbose is not None:
        logging.info('Total time: {:.2f}s'.format(total))

    return trajectory


def generate_trajectories(agent, n_trajectories, trajectory_length, features=None, init_feats=None,
                          model=None, horizon=None, selection=None, threshold=None,
                          processes=None, seed=0, verbose=None):
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param list[str] features: the list of relevant features (keys) for which to initialize trajectories. If `None`,
    each trajectory start from the world's current state.
    :param list[list] init_feats: a list of initial feature states from which to randomly initialize the
    trajectories. Each item is a list of feature values in the same order of `features`. If `None`, then features will
    be initialized uniformly at random according to their domain.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
    :param int seed: the seed used to initialize the random number generator.
    :param bool or Callable verbose: whether to show information at each timestep during trajectory generation.
    :rtype: list[list[tuple[World, Distribution]]]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    world = agent.world
    features = [] if features is None else features
    for feature in features:
        assert feature in world.variables, 'World does not have feature \'{}\'!'.format(feature)

    # generates each trajectory using a different random seed
    start = timer()
    pool, map_func = get_pool_and_map(processes, True)
    trajectories = list(map_func(
        generate_trajectory,
        [[agent, trajectory_length, features, init_feats, model, horizon, selection, threshold, seed + t, verbose]
         for t in range(n_trajectories)]))

    if verbose:
        logging.info('Total time for generating {} trajectories: {:.2f}s'.format(n_trajectories, timer() - start))

    if pool is not None:
        pool.close()

    return trajectories


def sample_sub_trajectories(trajectory, n_trajectories, trajectory_length, with_replacement=False, seed=0):
    """
    Randomly samples sub-trajectories from a given trajectory.
    :param list[tuple[World, Distribution]] trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :param bool with_replacement: whether to allow repeated sub-trajectories to be sampled.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[list[tuple[World, Distribution]]]
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


def log_trajectories(trajectories, features):
    """
    Prints the given trajectories to the log at the info level.
    :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories to save, containing
    several sequences of state-action pairs.
    :param list[str] features: the state features to be printed at each step representing the state.
    :return:
    """
    if len(trajectories) == 0 or len(trajectories[0]) == 0:
        return

    for i, trajectory in enumerate(trajectories):
        logging.info('-------------------------------------------')
        logging.info('Trajectory {}:'.format(i))
        for t, sa in enumerate(trajectory):
            world, action = sa
            feat_values = [str(world.getFeature(feat, unique=True)) for feat in features]
            logging.info('{}:\t({}) -> {}'.format(
                t, ', '.join(feat_values), action if len(action) > 1 else action.first()))
