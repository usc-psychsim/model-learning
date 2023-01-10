from typing import Union, List, Dict, Literal, NamedTuple

from psychsim.action import ActionSet
from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


# types
class StateActionPair(NamedTuple):
    """
    Represents a state-action pair for a single agent, with associated a probability.
    """
    world: World
    action: Distribution
    prob: float = 1.


class TeamStateActionPair(NamedTuple):
    """
    Represents a state-action pair for a team of agents, with an associated likelihood.
    """
    world: World
    action: Dict[str, Distribution]
    prob: float = 1.


class TeamStateActionModelDistTuple(NamedTuple):
    """
    Represents a state-action-model distribution tuple for a team of agents, with an associated likelihood.
    """
    state: VectorDistributionSet
    action: Dict[str, Distribution]
    model_dist: Distribution
    prob: float = 1.


PsychSimType = Union[float, int, str, ActionSet]
SelectionType = Literal['distribution', 'random', 'uniform', 'consistent', 'softmax']

State = VectorDistributionSet
Trajectory = List[StateActionPair]  # list of state (world) - action (distribution) pairs
TeamTrajectory = List[TeamStateActionPair]
TeamModelDistTrajectory = List[TeamStateActionModelDistTuple]
