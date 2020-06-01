import copy
import random
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, VectorDistributionSet

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def generate_trajectories(world, agent, n_trajectories, trajectory_length,
                          features=None, init_feats=None, model=None, horizon=None, selection=None, seed=0):
    """
    Generates a number of fixed-length agent trajectories/traces/paths (state-action pairs).
    :param World world: the PsychSim world in which generate trajectories.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param list[str] features: the list of relevant features (keys) for which to initialize trajectories. If `None`,
    each trajectory start from the world's current state.
    :param list[list] init_feats: a list of initial feature states from which to randomly initialize the trajectories.
    :param str model: the agent model used to generate the trajectories.
    Each item is a list of feature values in the same order of `features`. If `None`, then features will be initialized
    uniformly at random according to their domain.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[list[VectorDistributionSet, ActionSet]]
    :return:
    """
    # initial checks
    assert agent.name in world.agents, 'Agent \'{}\' does not exist in the world!'.format(agent.name)
    features = [] if features is None else features
    for feature in features:
        assert feature in world.variables, 'World does not have feature \'{}\'!'.format(feature)

    old_state = copy.deepcopy(world.state)
    rng = random.Random(seed)
    trajectories = []
    for t in range(n_trajectories):
        trajectory = []

        # generate or select initial state features
        if init_feats is None:
            init_values = [get_random_value(world, feature, rng) for feature in features]
        else:
            init_values = rng.choice(init_feats)

        # set features' initial value
        for feature, init_value in zip(features, init_values):
            world.setFeature(feature, init_value)

        # for each step, takes action and registers state-action pairs
        for i in range(trajectory_length):
            decision = agent.decide(
                world.state, horizon, None, model, selection, agent.getActions(world.state))
            action = decision[world.getValue(modelKey(agent.name))]['action']
            if isinstance(action, Distribution):
                action = rng.choices(action.domain(), action.values())[0]

            trajectory.append((copy.deepcopy(world.state), action))
            world.step(action)

        trajectories.append(trajectory)

    world.state = old_state
    return trajectories
