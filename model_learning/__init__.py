import copy
import random
import multiprocessing as mp
from psychsim.action import ActionSet, Action
from psychsim.agent import Agent
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, KeyedVector, VectorDistributionSet

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def generate_trajectory(agent, trajectory_length, features=None, init_feats=None,
                        model=None, horizon=None, selection=None, seed=0):
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param list[str] features: the list of relevant features (keys) for which to initialize trajectories. If `None`,
    each trajectory start from the world's current state.
    :param list[list] init_feats: a list of initial feature states from which to randomly initialize the
    trajectories. Each item is a list of feature values in the same order of `features`. If `None`, then features will
    be initialized uniformly at random according to their domain.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[list[VectorDistributionSet, ActionSet]]
    :return: a list of trajectories, each containing a list of state-action pairs.
    :return:
    """
    world = agent.world
    rng = random.Random(seed)

    # generate or select initial state features
    if init_feats is None:
        init_values = [get_random_value(world, feature, rng) for feature in features]
    else:
        init_values = rng.choice(init_feats)

    # set features' initial value
    for feature, init_value in zip(features, init_values):
        world.setFeature(feature, init_value)

    # for each step, takes action and registers state-action pairs
    trajectory = []
    for i in range(trajectory_length):
        decision = agent.decide(world.state, horizon, None, model, selection, None)
        action = decision[world.getFeature(modelKey(agent.name), unique=True)]['action']
        if isinstance(action, Distribution):
            action = rng.choices(action.domain(), action.values())[0]

        trajectory.append((copy.deepcopy(world.state), action))
        world.step(action, select=True)

    return trajectory


def generate_trajectories(agent, n_trajectories, trajectory_length, features=None, init_feats=None,
                          model=None, horizon=None, selection=None, processes=-1, seed=0):
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
    :param int processes: number of processes to use. `<=0` indicates all cores available, `1` uses single process.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[list[VectorDistributionSet, ActionSet]]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """

    # initial checks
    world = agent.world
    features = [] if features is None else features
    for feature in features:
        assert feature in world.variables, 'World does not have feature \'{}\'!'.format(feature)

    # generates each trajectory in parallel using a different random seed
    with mp.Pool(processes if processes > 0 else None) as pool:
        trajectories = pool.starmap(
            generate_trajectory,
            [[agent, trajectory_length, features, init_feats, model, horizon, selection, seed + t]
             for t in range(n_trajectories)])

        return trajectories


def get_policy(agent, states, model=None, horizon=None, selection=None):
    """
    Gets an agent's policy (action selection) for the given states.
    :param Agent agent: the agent for which to calculate the policy.
    :param list[KeyedVector or VectorDistributionSet] states: the list of states for which to get the agent's policy.
    :param str model: the name of the agent's model used for decision-making. `None` corresponds to the true model.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :rtype: list[ActionSet or Action or Distribution]
    :return: a list containing an action (or distribution) for each given state.
    """
    world = agent.world
    pi = []
    for s in states:
        decision = agent.decide(s, horizon, None, model, selection)
        pi.append(decision[world.getFeature(modelKey(agent.name), unique=True)]['action'])
    return pi
