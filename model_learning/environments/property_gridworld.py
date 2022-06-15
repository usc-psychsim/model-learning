import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal, Set
from psychsim.world import World
from psychsim.agent import Agent
from psychsim.action import ActionSet
from psychsim.pwl import makeTree, incrementMatrix, noChangeMatrix, thresholdRow, stateKey, VectorDistributionSet, \
    KeyedPlane, KeyedVector, rewardKey, setToConstantMatrix, equalRow, makeFuture, actionKey, state2feature, \
    equalFeatureRow, WORLD
from model_learning.environments.gridworld import GridWorld
from model_learning.util.plot import distinct_colors

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear']
GOAL_FEATURE = 'g'


class Location(NamedTuple):
    x: int
    y: int


class LocationInfo(NamedTuple):
    loci: int
    p: str


# define object property
def get_property_features() -> str:
    p = stateKey(WORLD, PROPERTY_FEATURE)
    return p


def get_goal_features() -> str:
    g = stateKey(WORLD, GOAL_FEATURE)
    return g


class PropertyGridWorld(GridWorld):

    def __init__(self, world: World, width: int, height: int, num_exist: int,
                 name: str = '', seed: int = 0, show_objects: bool = True):
        super().__init__(world, width, height, name)

        self.num_exist = num_exist
        rng = np.random.RandomState(seed)

        # set property for exist locations
        self.exist_locations = rng.choice(np.arange(width * height), num_exist, False).tolist()
        self.property_length = len(PROPERTY_LIST)
        self.n_p_state = self.property_length ** self.num_exist
        self.p_state = {}

        all_loc_all_p: List[List] = []
        for loci, loc in enumerate(self.exist_locations):
            loc_all_p = []
            for pi, p in enumerate(PROPERTY_LIST):
                loc_all_p.append(LocationInfo(loc, p))
            all_loc_all_p.append(loc_all_p)

        for i, comb in enumerate(itertools.product(*all_loc_all_p)):
            self.p_state[i] = {}
            for loc_info in comb:
                self.p_state[i][loc_info.loci] = loc_info.p
        print(self.p_state)

        self.p = self.world.defineState(WORLD, PROPERTY_FEATURE,
                                        int, 0, self.n_p_state - 1, description=f'Each location\'s property')
        self.world.setFeature(self.p, 0)
        self.g = self.world.defineState(WORLD, GOAL_FEATURE, int, 0, len(self.exist_locations),
                                        description=f'GOAL: # of cleared locations')
        self.world.setFeature(self.g, 0)

        # find # of GOAL location for each property state, p_idx: n_loc
        self.property_goal_count: Dict[int, int] = {}
        self.get_property_goal_count()

    def vis_agent_dynamics_in_xy(self):
        for k, v in self.world.dynamics.items():
            if type(k) is ActionSet:
                print('----')
                print(k)
                for kk, vv in v.items():
                    print(kk)
                    print(vv)

    def get_property_goal_count(self):
        for i in range(self.n_p_state):
            self.property_goal_count[i] = 0
        for p_idx in self.get_all_p_idx_has_p(PROPERTY_LIST[-1]):
            n_goal = sum([p == PROPERTY_LIST[-1] for p in self.p_state[p_idx].values()])
            self.property_goal_count[p_idx] = n_goal

    def remove_action(self, agent: Agent, action: str):
        illegal_action = agent.find_action({'action': action})
        agent.setLegal(illegal_action, makeTree(False))
        self.agent_actions[agent.name].remove(illegal_action)

    def get_all_p_idx_has_p(self, p) -> Set:
        all_p_idx: Set = set()
        for loc in self.exist_locations:
            all_p_idx.update(self.get_possible_p_idx(loc, p))
        return all_p_idx

    def get_possible_p_idx(self, loc, p):
        possible_p_idx = []
        if loc not in self.exist_locations:
            return possible_p_idx
        for pi, _loc_p in self.p_state.items():
            if _loc_p[loc] == p:
                possible_p_idx.append(pi)
        return possible_p_idx

    def get_next_p_idx(self, p_idx, new_loc, new_p):
        next_p = self.p_state[p_idx].copy()
        next_p[new_loc] = new_p
        for next_p_idx, _loc_p in self.p_state.items():
            if list(_loc_p.values()) == list(next_p.values()):
                return next_p_idx

    def add_location_property_dynamics(self, agent: Agent, idle: bool = True):
        assert agent.name not in self.agent_actions, f'An agent was already registered with the name \'{agent.name}\''

        self.add_agent_dynamics(agent)
        if not idle:
            self.remove_action(agent, 'nowhere')
        x, y = self.get_location_features(agent)

        # property related action
        action = agent.addAction({'verb': 'handle', 'action': 'search'})
        # search is valid everywhere
        # model dynamics when agent is at a possible location
        tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[0])
            subtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            # model dynamics for all possible property states at the location
            for j, p_idx in enumerate(all_p_idx):
                subtree_dict[j] = setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[1]))
            subtree_dict[None] = noChangeMatrix(self.p)
            tree_dict[i] = subtree_dict
        tree_dict[None] = noChangeMatrix(self.p)
        self.world.setDynamics(self.p, action, makeTree(tree_dict))
        self.agent_actions[agent.name].append(action)

        action = agent.addAction({'verb': 'handle', 'action': 'rescue'})
        # legality assumption: agent has observation on property state
        # rescue is valid only when agent reach a location and the property state is rescue needed
        # future: this action could be role-dependent, legality dynamics could be removed
        legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[1])
            sublegal_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                sublegal_dict[j] = True
            sublegal_dict[None] = False
            legal_dict[i] = sublegal_dict
        legal_dict[None] = False
        agent.setLegal(action, makeTree(legal_dict))

        rescue_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[1])
            subrescue_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                subrescue_dict[j] = setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[2]))
            subrescue_dict[None] = noChangeMatrix(self.p)
            rescue_dict[i] = subrescue_dict
        rescue_dict[None] = noChangeMatrix(self.p)
        self.world.setDynamics(self.p, action, makeTree(rescue_dict))
        self.agent_actions[agent.name].append(action)

        return self.agent_actions[agent.name]

    def add_collaboration_dynamics(self, agents: List[Agent]):
        assert len(agents) == 2, f'Now limited to collaboration with 2 agents'
        for i, agent in enumerate(agents):
            assert agent.name in self.agent_actions, f'An agent was not registered with the name \'{agent.name}\''

        carry_action = {'verb': 'handle', 'action': 'evacuate'}
        action_list = []
        for i, agent in enumerate(agents):
            if carry_action not in self.agent_actions[agent.name]:
                action = agent.addAction(carry_action)
                action_list.append(action)
                self.agent_actions[agent.name].append(action)

        for i, agent in enumerate(agents):
            j = -(i - 1)
            agent1 = agents[i]
            agent2 = agents[j]
            x1, y1 = self.get_location_features(agent1)
            x2, y2 = self.get_location_features(agent2)
            # carry1, carry2 = action_list[i], action_list[j]

            # legality assumption: agent has observation on every agent's location
            legal_dict = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2), True: True, False: False}
            agents[i].setLegal(action_list[i], makeTree(legal_dict))

            tree_dict = {'if': equalRow(actionKey(agents[j].name, True), action_list[j])}
            subtree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.height}), self.exist_locations, 0)}
            for loci, loc in enumerate(self.exist_locations):
                all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[2])
                subsubtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
                for pj, p_idx in enumerate(all_p_idx):
                    subsubtree_dict[pj] = setToConstantMatrix(self.p,
                                                              self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[3]))
                subsubtree_dict[None] = noChangeMatrix(self.p)
                subtree_dict[loci] = subsubtree_dict
            subtree_dict[None] = noChangeMatrix(self.p)
            tree_dict[True] = subtree_dict
            tree_dict[False] = noChangeMatrix(self.p)
            self.world.setDynamics(self.p, action_list[i], makeTree(tree_dict))

            # update GOAL_FEATURE
            tree_dict = {'if': KeyedPlane(KeyedVector({makeFuture(self.p): 1}), list(self.property_goal_count.keys()), 0)}
            for gi, (k, v) in enumerate(self.property_goal_count.items()):
                tree_dict[gi] = setToConstantMatrix(self.g, v)
            tree_dict[None] = noChangeMatrix(self.g)
            self.world.setDynamics(self.g, action_list[i], makeTree(tree_dict))

    def get_all_states_with_properties(self, agent: Agent) -> List[Optional[VectorDistributionSet]]:
        assert agent.world == self.world, 'Agent\'s world different from the environment\'s world!'

        old_state = copy.deepcopy(self.world.state)
        states_wp = [None] * self.width * self.height * len(PROPERTY_LIST)

        # iterate through all agent positions and copy world state
        x, y = self.get_location_features(agent)

        for x_i, y_i in itertools.product(range(self.width), range(self.height)):
            self.world.setFeature(x, x_i)
            self.world.setFeature(y, y_i)
            idx = self.xy_to_idx(x_i, y_i) * len(PROPERTY_LIST)
            for i, property_s in enumerate(PROPERTY_LIST):
                self.world.setFeature(self.p, property_s)
                states_wp[idx + i] = copy.deepcopy(self.world.state)

        # undo world state
        self.world.state = old_state
        return states_wp

    # reward function
    def set_achieve_property_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        reward_dict = {'if': equalRow(actionKey(agent.name), agent.find_action({'action': 'evacuate'}))}
        subreward_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}),
                                           list(self.property_goal_count.keys()), 0)}
        for i, (p_idx, count) in enumerate(self.property_goal_count.items()):
            subreward_dict[i] = setToConstantMatrix(rewardKey(agent.name), count)
        subreward_dict[None] = setToConstantMatrix(rewardKey(agent.name), 0)
        reward_dict[True] = subreward_dict
        reward_dict[False] = setToConstantMatrix(rewardKey(agent.name), 0)
        agent.setReward(makeTree(reward_dict), weight, model)

    # reward function
    # def set_achieve_property_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
    #     reward_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), list(self.property_goal_count.keys()), 0)}
    #     for i, (p_idx, count) in enumerate(self.property_goal_count.items()):
    #         reward_dict[i] = setToConstantMatrix(rewardKey(agent.name), count)
    #     reward_dict[None] = setToConstantMatrix(rewardKey(agent.name), 0)
    #     agent.setReward(makeTree(reward_dict), weight, model)
