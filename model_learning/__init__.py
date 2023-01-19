from typing import Union, List, Dict, Literal, NamedTuple

from psychsim.action import ActionSet
from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

PsychSimType = Union[float, int, str, ActionSet]
SelectionType = Literal['distribution', 'random', 'uniform', 'consistent', 'softmax']
State = VectorDistributionSet


# types
class StateActionPair(NamedTuple):
    """
    Represents a state-action pair for a single agent, with an associated likelihood.
    """
    state: State
    action: Distribution
    prob: float = 1.


class StateActionModelDist(NamedTuple):
    """
    Represents a state-action-model distribution tuple for a single agent, with an associated likelihood.
    """
    state: State
    action: Distribution
    models_dist: Distribution  # a distribution over other an agent's models
    prob: float = 1.


Trajectory = List[StateActionPair]  # list of state (world) - action (distribution) pairs
ModelDistTrajectory = List[StateActionModelDist]


class TeamStateActionPair(NamedTuple):
    """
    Represents a state-action pair for a team of agents, with an associated likelihood.
    """
    state: State
    action: Dict[str, Distribution]
    prob: float = 1.


# a distribution over other agents' models for each agent
TeamModelsDistributions = Dict[str, Dict[str, Distribution]]


class TeamStateActionModelDist(NamedTuple):
    """
    Represents a state-action-models distributions tuple for a team of agents, with an associated likelihood.
    """
    state: State
    action: Dict[str, Distribution]
    models_dists: TeamModelsDistributions
    prob: float = 1.


TeamTrajectory = List[TeamStateActionPair]
TeamModelDistTrajectory = List[TeamStateActionModelDist]

StateProbTrajectory = Union[Trajectory, ModelDistTrajectory, TeamTrajectory, TeamModelDistTrajectory]
SingleAgentTrajectory = Union[Trajectory, ModelDistTrajectory]
MultiagentTrajectory = Union[TeamTrajectory, TeamModelDistTrajectory]
