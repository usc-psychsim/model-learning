import numpy as np
from abc import ABC, abstractmethod
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.pwl import VectorDistributionSet
from model_learning.trajectory import copy_world

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
        self.__base_agent = base_agent
        self.agent = base_agent
        self.world = base_agent.world
        self._reset()

    def _reset(self):
        """
        Resets the algorithm by creating a clean copy of the world and agent
        """
        self.world = copy_world(self.__base_agent.world)
        self.agent = self.world.agents[self.__base_agent.name]

    @abstractmethod
    def learn(self, trajectories):
        """
        Performs model learning by retrieving a PsychSim model approximating an expert's behavior as demonstrated
        through the given trajectories.
        :param list[list[VectorDistributionSet, ActionSet]] trajectories:a list of trajectories, each containing a list
        (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: dict[str, np.ndarray]
        :return: a dictionary with relevant statistics of the algorithm.
        """
        pass

    @abstractmethod
    def save_results(self, stats, output_dir, img_format):
        """
        Saves the several results of a run of the algorithm to the given directory.
        :param dict[str, np.ndarray] stats: a dictionary with relevant statistics of the algorithm's run.
        :param str output_dir: the path to the directory in which to save the results.
        :param str img_format: the format of the images to be saved.
        :return:
        """
        pass
