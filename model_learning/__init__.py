from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet
from psychsim.world import World
from typing import Union, List, Tuple

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# types
PsychSimType = Union[float, int, str]
State = VectorDistributionSet
Trajectory = List[Tuple[World, Distribution]]  # list of world-action (distribution) pairs
