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
PROPERTY_LIST = ['unknown', 'medic', 'ready', 'clear']
GOAL_FEATURE = 'g'


class Location(NamedTuple):
    x: int
    y: int


class LocationInfo(NamedTuple):
    loci: int
    p: str


# define object property
class PropertyGridWorld(GridWorld):

    def __init__(self, world: World, width: int, height: int, num_exist: int,
                 name: str = '', seed: int = 0, show_objects: bool = True):
        super().__init__(world, width, height, name)

        self.num_exist = num_exist
        rng = np.random.RandomState(seed)

        # set property for exist locations
        self.exist_locations = rng.choice(np.arange(width * height), num_exist, False).tolist()
        print(self.exist_locations)
        print(PROPERTY_LIST)
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
        print(self.p)
        self.world.setFeature(self.p, 0)

    def vis_agent_dynamics_in_xy(self):
        for k, v in self.world.dynamics.items():
            if type(k) is ActionSet:
                print('----')
                print(k)
                for kk, vv in v.items():
                    print(kk)
                    print(vv)

    def get_agent_goal_feature(self, agent: Agent) -> str:
        g = stateKey(agent.name, GOAL_FEATURE + self.name)
        return g

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

    def add_location_property_dynamics(self, agent: Agent):
        assert agent.name not in self.agent_actions, f'An agent was already registered with the name \'{agent.name}\''

        self.add_agent_dynamics(agent)
        x, y = self.get_location_features(agent)

        g = self.world.defineState(agent.name, GOAL_FEATURE + self.name,
                                   int, 0, self.num_exist, description=f'Each location\'s property')
        self.world.setFeature(g, 0)

        # property related action
        action = agent.addAction({'verb': 'handle', 'action': 'search'})
        tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[0])
            subtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                loc_dict = {'if': equalRow(self.p, p_idx),
                            True: setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[1])),
                            False: noChangeMatrix(self.p)}
                subtree_dict[j] = loc_dict
            subtree_dict[None] = noChangeMatrix(self.p)
            tree_dict[i] = subtree_dict
        tree_dict[None] = noChangeMatrix(self.p)
        self.world.setDynamics(self.p, action, makeTree(tree_dict))
        self.agent_actions[agent.name].append(action)

        # action = agent.addAction({'verb': 'handle', 'action': 'rescue'})
        # legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        # for i, loc in enumerate(self.exist_locations):
        #     legal_dict[i] = True
        # legal_dict[None] = False
        # agent.setLegal(action, makeTree(legal_dict))
        # # problem here TODO
        action = agent.addAction({'verb': 'handle', 'action': 'rescue'})
        tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[1])
            subtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                loc_dict = {'if': equalRow(self.p, p_idx),
                            True: setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[2])),
                            False: noChangeMatrix(self.p)}
                subtree_dict[j] = loc_dict
            subtree_dict[None] = noChangeMatrix(self.p)
            tree_dict[i] = subtree_dict
        tree_dict[None] = noChangeMatrix(self.p)
        self.world.setDynamics(self.p, action, makeTree(tree_dict))
        self.agent_actions[agent.name].append(action)

        return self.agent_actions[agent.name]

    def add_collaboration_dynamics(self, agents: List[Agent]):
        assert len(agents) == 2, f'Now limited to collaboration with 2 agents'
        for i, agent in enumerate(agents):
            assert agent.name in self.agent_actions, f'An agent was not registered with the name \'{agent.name}\''

        carry_action = {'verb': 'handle', 'action': 'carry'}
        action_list = []
        g_list = []
        for i, agent in enumerate(agents):
            if carry_action not in self.agent_actions[agent.name]:
                action = agent.addAction(carry_action)
                action_list.append(action)
                self.agent_actions[agent.name].append(action)
                g = self.get_agent_goal_feature(agent)
                g_list.append(g)
        print(action_list)
        print(g_list)

        for i, agent in enumerate(agents):
            j = -(i - 1)
            agent1 = agents[i]
            agent2 = agents[j]
            x1, y1 = self.get_location_features(agent1)
            x2, y2 = self.get_location_features(agent2)
            # carry1, carry2 = action_list[i], action_list[j]

            legal_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.height}), self.exist_locations, 0)}
            for ii, loc in enumerate(self.exist_locations):
                legal_dict[ii] = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2),
                                  True: True, False: False}
            legal_dict[None] = False
            agents[i].setLegal(action_list[i], makeTree(legal_dict))

            tree_dict = {'if': equalRow(actionKey(agents[j].name, True), action_list[j])}
            subtree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.height}), self.exist_locations, 0)}
            for loci, loc in enumerate(self.exist_locations):
                all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[2])
                subsubtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
                for pj, p_idx in enumerate(all_p_idx):
                    loc_dict = {'if': equalRow(self.p, p_idx),
                                True: setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[3])),
                                False: noChangeMatrix(self.p)}
                    subsubtree_dict[pj] = loc_dict
                subsubtree_dict[None] = noChangeMatrix(self.p)
                subtree_dict[loci] = subsubtree_dict
            subtree_dict[None] = noChangeMatrix(self.p)
            tree_dict[True] = subtree_dict
            tree_dict[False] = noChangeMatrix(self.p)
            self.world.setDynamics(self.p, action_list[i], makeTree(tree_dict))

            tree_dict = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2) &
                               equalRow(actionKey(agents[j].name, True), action_list[j])}
            subtree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.height}), self.exist_locations, 0)}
            for loci, loc in enumerate(self.exist_locations):
                all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[2])
                subsubtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
                for pj, p_idx in enumerate(all_p_idx):
                    loc_dict = {'if': equalRow(self.p, p_idx),
                                True: incrementMatrix(g_list[i], 1),
                                False: noChangeMatrix(g_list[i])}
                    subsubtree_dict[pj] = loc_dict
                subsubtree_dict[None] = noChangeMatrix(g_list[i])
                subtree_dict[loci] = subsubtree_dict
            subtree_dict[None] = noChangeMatrix(g_list[i])
            tree_dict[True] = subtree_dict
            tree_dict[False] = noChangeMatrix(g_list[i])
            self.world.setDynamics(g_list[i], action_list[i], makeTree(tree_dict))

    def get_property_features(self) -> str:
        p = stateKey(WORLD, PROPERTY_FEATURE)
        return p

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

    # def set_achieve_locations_reward(self, agent: Agent, locs: List[Location], weight: float, model: Optional[str] = None):
    #     agent.setReward(makeTree({'if': self.get_location_plane(agent, locs),
    #                               True: setToConstantMatrix(rewardKey(agent.name), 1.),
    #                               False: setToConstantMatrix(rewardKey(agent.name), 0.)}), weight, model)

    # reward function
    def set_achieve_property_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        x, y = self.get_agent_location_features(agent)
        reward_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.height}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            subtree_dict = {'if': equalRow(stateKey(self.name, PROPERTY_FEATURE), len(self.p_state) - 1),
                            True: setToConstantMatrix(rewardKey(agent.name), 1),
                            False: setToConstantMatrix(rewardKey(agent.name), 0)}
            reward_dict[i] = subtree_dict
        reward_dict[None] = setToConstantMatrix(rewardKey(agent.name), 0)
        agent.setReward(makeTree(reward_dict), weight, model)

    def get_property_plane(self, p_idxs: List[int],
                           comp: Literal[KeyedPlane.COMPARISON_MAP] = KeyedPlane.COMPARISON_MAP[0]) -> KeyedPlane:

        assert comp in KeyedPlane.COMPARISON_MAP, \
            f'Invalid comparison provided: {comp}; valid: {KeyedPlane.COMPARISON_MAP}'

        # creates plane that checks if p equals any idx in the set
        return KeyedPlane(KeyedVector({self.p: 1}), p_idxs, KeyedPlane.COMPARISON_MAP.index(comp))

    # def get_location_id_plane(self,
    #                           agent: Agent,
    #                           locs: List,
    #                           comp: Literal[KeyedPlane.COMPARISON_MAP] = KeyedPlane.COMPARISON_MAP[0]) -> KeyedPlane:
    #
    #     assert comp in KeyedPlane.COMPARISON_MAP, \
    #         f'Invalid comparison provided: {comp}; valid: {KeyedPlane.COMPARISON_MAP}'
    #
    #     x_feat, y_feat = self.get_location_features(agent)
    #     loc_idxs = locs
    #
    #     # creates plane that checks if x+(y*width) equals any idx in the set
    #     return KeyedPlane(KeyedVector({x_feat: 1., y_feat: self.width}),
    #                       loc_idxs,
    #                       KeyedPlane.COMPARISON_MAP.index(comp))
