from psychsim.action import ActionSet, Action
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, KeyedVector, VectorDistributionSet

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


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
        decision = agent.decide(s, horizon, None, model, selection, agent.getActions(world.state))
        pi.append(decision[world.getFeature(modelKey(agent.name), unique=True)]['action'])
    return pi
