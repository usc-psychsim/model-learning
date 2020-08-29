import copy
import numpy as np
from abc import ABC, abstractmethod
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.pwl import VectorDistributionSet

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class ModelLearningAlgorithm(ABC):
    """
    An abstract class for PsychSim model learning algorithms, where a *learner* is given a POMDP definition,
    including the world states, observations and beliefs. Given a set of trajectories produced by some target *expert*
    behavior (demonstrations), the goal of the algorithm is to find a PsychSim model such that the learner's behavior
    resulting from using such model best approximates that of the expert.
    The PsychSim model might include the agent's reward function, its available actions (and legality constraints),
    its planning horizon, etc. Some of these elements might be provided and set fixed by the problem's definition,
    while others are set as free parameters to be optimized by the algorithm.
    """

    def __init__(self, label, base_agent):
        """
        Creates a new algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param Agent base_agent: the base agent containing the model parameters that will not be optimized by the
        algorithm. Contains reference to the PsychSim world containing all definitions.
        """
        self.label = label
        self.world = copy.deepcopy(base_agent.world)
        self.agent = self.world.agents[base_agent.name]

    @abstractmethod
    def learn(self, trajectories):
        """
        Performs model learning by retrieving a PsychSim model approximating an expert's behavior as demonstrated
        through the given trajectories.
        :param list[list[VectorDistributionSet, ActionSet]] trajectories:a list of trajectories, each containing a list
        (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: (dict, dict[str, np.ndarray])
        :return: a tuple (model, stats) containing the learned PsychSim model (a dictionary of parameters), and a
        dictionary with relevant statistics of the algorithm.
        """
        pass
