import logging
import numpy as np
from typing import List
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import modelKey

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def track_reward_model_inference(trajectory, models, agent, observer, features=None, verbose=True):
    """
    Updates and tracks the evolution of an observer agent's posterior distribution over the possible reward models of
    an "actor" agent. The inference task
    :param list[tuple[World, Distribution]] trajectory: the trajectory of (world, action) pairs containing the "actor"
    agent's actions. It is assumed that agents `agent` and `observer` are present in each "world" of the trajectory so
    that we can recover the latter's distribution over reward models.
    :param list[str] models: a list of model names of the `agent` agent containing the different reward functions.
    :param Agent agent: the "actor" agent performing actions in the trajectory and containing the reward models.
    :param Agent observer: the observer agent modeling the actor whose beliefs we want to track.
    :param list[str] features: a list of world feature keys to print at each timestep, if `verbose` is `True`.
    :param bool verbose: whether to print info messages to the logger at each timestep.
    :rtype: np.ndarray
    :return: an array of shape (trajectory_length, num_models) containing the probability attributed by the observer
    agent to the reward model of the actor agent, at each timestep along the trajectory.
    """
    # gets the reward models the posterior over which we want to update and track
    rwd_models = [agent.getReward(name) for name in models]

    probs = np.zeros((len(trajectory), len(models)))
    for t, sa in enumerate(trajectory):
        world = sa.world
        action = sa.action
        # world, action = sa

        if verbose and features is not None:
            logging.info('===============================================')
            logging.info('{}.\t({})'.format(t, ','.join([str(world.getFeature(f, unique=True)) for f in features])))

        # get actual agents from the trajectory world
        observer = world.agents[observer.name]
        agent = world.agents[agent.name]

        # get observer's beliefs about agent
        beliefs = observer.getBelief(world.state)
        assert len(beliefs) == 1  # Because we are dealing with a known-identity agent
        belief = next(iter(beliefs.values()))

        # update posterior distribution over models
        model_dist = world.getFeature(modelKey(agent.name), belief)
        for model in model_dist.domain():
            rwd_model = agent.getReward(model)
            probs[t, rwd_models.index(rwd_model)] = model_dist[model]

        if verbose:
            logging.info('Observer models agent as:')
            logging.info(model_dist)
            logging.info(action if len(action) > 1 else action.first())

    return probs


# def add_agent_models(agent: Agent, reward_vector,
#                      reward_weights: List, model_names: List):
#     for model_name in model_names:
#         true_model = agent.get_true_model()
#         model_name = f'{agent.name}_{model_name}'
#         agent.addModel(model_name, parent=true_model)
#         agent_lrv = reward_vector
#         rwd_f_weights = reward_weights
#         if model_name == f'{agent.name}_Opposite':
#             rwd_f_weights = -1. * np.array(rwd_f_weights)
#             rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
#         if model_name == f'{agent.name}_Uniform':
#             rwd_f_weights = [1] * len(rwd_f_weights)
#             rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
#         if model_name == f'{agent.name}_Random':
#             rwd_f_weights = [0] * len(rwd_f_weights)
#         agent_lrv.set_rewards(agent, rwd_f_weights, model=model_name)
#         print(agent.name, model_name, agent_lrv.names, rwd_f_weights)
#     return agent
