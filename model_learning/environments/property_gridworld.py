import copy
import itertools
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal, Set, Any
from psychsim.world import World
from psychsim.agent import Agent
from psychsim.action import ActionSet
from psychsim.pwl import makeTree, incrementMatrix, noChangeMatrix, thresholdRow, stateKey, VectorDistributionSet, \
    KeyedPlane, KeyedVector, rewardKey, setToConstantMatrix, equalRow, makeFuture, actionKey, state2feature, \
    equalFeatureRow, WORLD
from model_learning.environments.gridworld import GridWorld
from model_learning import Trajectory
from model_learning.trajectory import generate_team_trajectories
from model_learning.util.plot import distinct_colors

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

X_FEATURE = 'x'
Y_FEATURE = 'y'
VISIT_FEATURE = 'v'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
GOAL_FEATURE = 'g'

TITLE_FONT_SIZE = 12
VALUE_CMAP = 'gray'  # 'viridis' # 'inferno'
TRAJECTORY_LINE_WIDTH = 1
LOC_FONT_SIZE = 6
LOC_FONT_COLOR = 'darkgrey'
NOTES_FONT_SIZE = 8
NOTES_FONT_COLOR = 'dimgrey'
POLICY_MARKER_COLOR = 'dimgrey'


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
        self.seed = seed
        rng = np.random.RandomState(self.seed)

        # set property for exist locations
        self.exist_locations = rng.choice(np.arange(width * height), num_exist, False).tolist()
        self.non_exist_locations = list(set(range(width * height)) - set(self.exist_locations))

        self.property_length = len(PROPERTY_LIST)
        self.n_p_state = self.property_length ** (width * height)
        self.p_state = {}

        all_loc_all_p: List[List] = []
        for loc_i in range(width * height):
            loc_all_p = []
            for pi, p in enumerate(PROPERTY_LIST):
                loc_all_p.append(LocationInfo(loc_i, p))
            all_loc_all_p.append(loc_all_p)
        # print(all_loc_all_p)

        for i, comb in enumerate(itertools.product(*all_loc_all_p)):
            self.p_state[i] = {}
            for loc_info in comb:
                self.p_state[i][loc_info.loci] = loc_info.p
        # print(self.p_state)

        self.p = self.world.defineState(WORLD, PROPERTY_FEATURE,
                                        int, 0, self.n_p_state - 1, description=f'Each location\'s property')
        self.world.setFeature(self.p, 0)
        self.g = self.world.defineState(WORLD, GOAL_FEATURE, int, 0, len(self.exist_locations),
                                        description=f'GOAL: # of cleared locations')
        self.world.setFeature(self.g, 0)

        # find # of GOAL location for each property state, p_idx: n_loc
        self.property_goal_count: Dict[int, int] = {}
        self.get_property_goal_count()

        # # set property for exist locations
        # self.exist_locations = rng.choice(np.arange(width * height), num_exist, False).tolist()
        # self.property_length = len(PROPERTY_LIST)
        # self.n_p_state = self.property_length ** self.num_exist
        # self.p_state = {}
        #
        # all_loc_all_p: List[List] = []
        # for loci, loc in enumerate(self.exist_locations):
        #     loc_all_p = []
        #     for pi, p in enumerate(PROPERTY_LIST):
        #         loc_all_p.append(LocationInfo(loc, p))
        #     all_loc_all_p.append(loc_all_p)
        #
        # for i, comb in enumerate(itertools.product(*all_loc_all_p)):
        #     self.p_state[i] = {}
        #     for loc_info in comb:
        #         self.p_state[i][loc_info.loci] = loc_info.p
        # print(self.p_state)
        #
        # self.p = self.world.defineState(WORLD, PROPERTY_FEATURE,
        #                                 int, 0, self.n_p_state - 1, description=f'Each location\'s property')
        # self.world.setFeature(self.p, 0)
        # self.g = self.world.defineState(WORLD, GOAL_FEATURE, int, 0, len(self.exist_locations),
        #                                 description=f'GOAL: # of cleared locations')
        # self.world.setFeature(self.g, 0)
        #
        # # find # of GOAL location for each property state, p_idx: n_loc
        # self.property_goal_count: Dict[int, int] = {}
        # self.get_property_goal_count()

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
        for p_idx in self.get_all_p_idx_has_p(PROPERTY_LIST[3]):
            n_goal = sum([p == PROPERTY_LIST[3] for p in self.p_state[p_idx].values()])
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
        # if loc not in self.exist_locations:
        #     return possible_p_idx
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
        else:
            action = agent.find_action({'action': 'nowhere'})
            legal_dict = {'if': equalRow(self.g, self.num_exist), True: True, False: False}
            agent.setLegal(action, makeTree(legal_dict))
        x, y = self.get_location_features(agent)

        # property related action
        action = agent.addAction({'verb': 'handle', 'action': 'search'})
        # search is valid when location is unknown
        # model dynamics when agent is at a possible location
        legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), list(range(self.width*self.height)), 0)}
        for i, loc in enumerate(list(range(self.width*self.height))):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[0])
            sublegal_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                sublegal_dict[j] = True
            sublegal_dict[None] = False
            legal_dict[i] = sublegal_dict
        legal_dict[None] = False
        agent.setLegal(action, makeTree(legal_dict))

        exist_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.exist_locations, 0)}
        for exist_i, exist_loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(exist_loc, PROPERTY_LIST[0])
            sub_exist_tree = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                sub_exist_tree[j] = setToConstantMatrix(self.p,
                                                        self.get_next_p_idx(p_idx, exist_loc, PROPERTY_LIST[1]))
            sub_exist_tree[None] = noChangeMatrix(self.p)
            exist_tree[exist_i] = sub_exist_tree
        non_exist_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.non_exist_locations, 0)}
        for non_exist_i, non_exist_loc in enumerate(self.non_exist_locations):
            all_p_idx = self.get_possible_p_idx(non_exist_loc, PROPERTY_LIST[0])
            sub_non_exist_tree = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                sub_non_exist_tree[j] = setToConstantMatrix(self.p,
                                                            self.get_next_p_idx(p_idx, non_exist_loc, PROPERTY_LIST[4]))
            sub_non_exist_tree[None] = noChangeMatrix(self.p)
            non_exist_tree[non_exist_i] = sub_non_exist_tree
        exist_tree[None] = non_exist_tree
        self.world.setDynamics(self.p, action, makeTree(exist_tree))
        self.agent_actions[agent.name].append(action)

        # action = agent.addAction({'verb': 'handle', 'action': 'search'})
        # # search is valid everywhere
        # # model dynamics when agent is at a possible location
        # tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.exist_locations, 0)}
        # for i, loc in enumerate(self.exist_locations):
        #     all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[0])
        #     subtree_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
        #     # model dynamics for all possible property states at the location
        #     for j, p_idx in enumerate(all_p_idx):
        #         subtree_dict[j] = setToConstantMatrix(self.p, self.get_next_p_idx(p_idx, loc, PROPERTY_LIST[1]))
        #     subtree_dict[None] = noChangeMatrix(self.p)
        #     tree_dict[i] = subtree_dict
        # tree_dict[None] = noChangeMatrix(self.p)
        # self.world.setDynamics(self.p, action, makeTree(tree_dict))
        # self.agent_actions[agent.name].append(action)

        action = agent.addAction({'verb': 'handle', 'action': 'rescue'})
        # legality assumption: agent has observation on property state
        # rescue is valid only when agent reach a location and the property state is rescue needed
        # future: this action could be role-dependent, legality dynamics could be removed
        legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[1])
            sublegal_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
            for j, p_idx in enumerate(all_p_idx):
                sublegal_dict[j] = True
            sublegal_dict[None] = False
            legal_dict[i] = sublegal_dict
        legal_dict[None] = False
        agent.setLegal(action, makeTree(legal_dict))

        rescue_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.exist_locations, 0)}
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
            legal_dict = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2)}
            sublegal_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), self.exist_locations, 0)}
            for loci, loc in enumerate(self.exist_locations):
                all_p_idx = self.get_possible_p_idx(loc, PROPERTY_LIST[2])
                subsublegal_dict = {'if': KeyedPlane(KeyedVector({self.p: 1}), all_p_idx, 0)}
                for pj, p_idx in enumerate(all_p_idx):
                    subsublegal_dict[pj] = True
                subsublegal_dict[None] = False
                sublegal_dict[loci] = subsublegal_dict
            sublegal_dict[None] = False
            legal_dict[True] = sublegal_dict
            legal_dict[False] = False
            agents[i].setLegal(action_list[i], makeTree(legal_dict))

            tree_dict = {'if': equalRow(actionKey(agents[j].name, True), action_list[j])}
            subtree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), self.exist_locations, 0)}
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
            tree_dict = {
                'if': KeyedPlane(KeyedVector({makeFuture(self.p): 1}), list(self.property_goal_count.keys()), 0)}
            for gi, (k, v) in enumerate(self.property_goal_count.items()):
                tree_dict[gi] = setToConstantMatrix(self.g, v)
            tree_dict[None] = noChangeMatrix(self.g)
            self.world.setDynamics(self.g, action_list[i], makeTree(tree_dict))

    def get_all_states_with_properties(self, agent1: Agent, agent2: Agent) -> List[Optional[VectorDistributionSet]]:
        assert agent1.world == self.world, 'Agent\'s world different from the environment\'s world!'

        old_state = copy.deepcopy(self.world.state)
        states_wp = [None] * self.width * self.height * self.width * self.height * self.n_p_state

        # iterate through all agent positions and copy world state
        x, y = self.get_location_features(agent1)
        x1, y1 = self.get_location_features(agent2)
        for x1_i, y1_i in itertools.product(range(self.width), range(self.height)):
            self.world.setFeature(x1, x1_i)
            self.world.setFeature(y1, y1_i)
            idxx = self.xy_to_idx(x1_i, y1_i) * self.width * self.height * self.n_p_state

            for x_i, y_i in itertools.product(range(self.width), range(self.height)):
                self.world.setFeature(x, x_i)
                self.world.setFeature(y, y_i)

                idx = self.xy_to_idx(x_i, y_i) * self.n_p_state
                for i, p in enumerate(self.p_state):
                    self.world.setFeature(self.p, p)
                    states_wp[idxx + idx + i] = copy.deepcopy(self.world.state)

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

    def generate_team_trajectories(self, team: List[Agent], trajectory_length: int,
                                   n_trajectories: int = 1, init_feats: Optional[Dict[str, Any]] = None,
                                   model: Optional[str] = None, select: bool = True,
                                   selection: Optional[
                                       Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                   horizon: Optional[int] = None, threshold: Optional[float] = None,
                                   processes: Optional[int] = -1, seed: int = 0, verbose: bool = False,
                                   use_tqdm: bool = True) -> List[Dict[str, Trajectory]]:
        assert len(team) > 0, 'No agent in the team'

        x, y = self.get_location_features(team[0])
        p = get_property_features()
        # if not specified, set random values for x, y pos, and property
        if init_feats is None:
            init_feats = {}
        if x not in init_feats:
            init_feats[x] = None
        if y not in init_feats:
            init_feats[y] = None
        if p not in init_feats:
            init_feats[p] = 0

        # generate trajectories starting from random locations in the property gridworld
        return generate_team_trajectories(team, n_trajectories, trajectory_length,
                                          init_feats, model, select, horizon, selection, threshold,
                                          processes, seed, verbose, use_tqdm)

    def play_team_trajectories(self, team_trajectories: List[Dict[str, Trajectory]],
                               team: List[Agent],
                               file_name: str,
                               title: str = 'Team Trajectories'):
        assert len(team_trajectories) > 0 and len(team) > 0
        assert len(team_trajectories[0][team[0].name]) > 0

        for agent in team:
            x, y = self.get_location_features(agent)
            assert x in self.world.variables, f'Agent \'{agent.name}\' does not have x location feature'
            assert y in self.world.variables, f'Agent \'{agent.name}\' does not have y location feature'
        p = get_property_features()
        assert p in self.world.variables, f'World does not have property feature'

        for traj_i, team_traj in enumerate(team_trajectories):
            fig, axes = plt.subplots(len(team))
            fig.set_tight_layout(True)
            # plot base environment
            # plots grid with cell numbers
            grid = np.zeros((self.height, self.width))
            for ag_i in range(len(team)):
                ax = axes[ag_i]
                ax.pcolor(grid, cmap=ListedColormap(['white']), edgecolors='darkgrey')
                for x, y in itertools.product(range(self.width), range(self.height)):
                    ax.annotate('({},{})'.format(x, y), xy=(x + .05, y + .1), fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
                # turn off tick labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', adjustable='box')
                if ag_i == 0:
                    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

            # plots trajectories
            l_traj = len(team_traj[team[0].name])
            t_colors = distinct_colors(len(team))
            team_xs, team_ys, team_as, world_ps = {}, {}, {}, {}
            for loci, loc in enumerate(self.exist_locations):
                world_ps[loc] = [None] * l_traj
            for loci, loc in enumerate(self.non_exist_locations):
                world_ps[loc] = [None] * l_traj
            for ag_i, agent in enumerate(team):
                team_xs[agent.name] = [0] * l_traj
                team_ys[agent.name] = [0] * l_traj
                team_as[agent.name] = [None] * l_traj
            for ag_i, agent in enumerate(team):
                agent_trajectory = team_traj[agent.name]
                x, y = self.get_location_features(agent)
                action = actionKey(agent.name)
                for i, sa in enumerate(agent_trajectory):
                    x_t = sa.world.getFeature(x, unique=True)
                    y_t = sa.world.getFeature(y, unique=True)
                    a = sa.world.getFeature(action, unique=True)
                    a = a['subject'] + '-' + a['action']
                    if ag_i == 0:
                        p_t = sa.world.getFeature(p, unique=True)
                        for loci, loc in enumerate(self.exist_locations):
                            world_ps[loc][i] = self.p_state[p_t][loc]
                        for loci, loc in enumerate(self.non_exist_locations):
                            world_ps[loc][i] = self.p_state[p_t][loc]
                    team_xs[agent.name][i] = x_t + 0.5
                    team_ys[agent.name][i] = y_t + 0.5
                    team_as[agent.name][i] = a

            team_ann_list = [[]] * len(team)
            world_ann_list = []

            def update(ti):
                for ann in world_ann_list:
                    ann.remove()
                world_ann_list[:] = []
                for ag_i, agent in enumerate(team):
                    for ann in team_ann_list[ag_i]:
                        if isinstance(ann, list):
                            ann.pop(0).remove()
                        else:
                            ann.remove()
                    team_ann_list[ag_i][:] = []
                for ag_i, agent in enumerate(team):
                    x_t, y_t, a = team_xs[agent.name][ti], team_ys[agent.name][ti], team_as[agent.name][ti]
                    act = axes[ag_i].annotate(f'{a}', xy=(x_t - .4, y_t + .1),
                                              fontsize=NOTES_FONT_SIZE, c=NOTES_FONT_COLOR)
                    xs = team_xs[agent.name][ti - 1:ti + 1]
                    ys = team_ys[agent.name][ti - 1:ti + 1]
                    traj_line = axes[ag_i].plot(xs, ys, c=t_colors[ag_i], linewidth=TRAJECTORY_LINE_WIDTH)
                    curr_pos = axes[ag_i].annotate('O', xy=(x_t - .05, y_t - .05), c=t_colors[ag_i],
                                                   fontsize=NOTES_FONT_SIZE)
                    ts_ann = axes[ag_i].annotate(f'Time Step: {ti}', xy=(.05, .05),
                                                 fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
                    team_ann_list[ag_i].append(ts_ann)
                    team_ann_list[ag_i].append(act)
                    team_ann_list[ag_i].append(curr_pos)
                    team_ann_list[ag_i].append(traj_line)
                for loci, loc in enumerate(self.exist_locations):
                    x, y = self.idx_to_xy(loc)
                    p = world_ps[loc][ti]
                    for ag_i, agent in enumerate(team):
                        status_ann = axes[ag_i].annotate(f'*V{loci + 1}\n{p}', xy=(x + .47, y + .3),
                                                         fontsize=NOTES_FONT_SIZE, c='k')
                        world_ann_list.append(status_ann)
                for loci, loc in enumerate(self.non_exist_locations):
                    x, y = self.idx_to_xy(loc)
                    p = world_ps[loc][ti]
                    for ag_i, agent in enumerate(team):
                        status_ann = axes[ag_i].annotate(f'\n{p}', xy=(x + .47, y + .3),
                                                         fontsize=NOTES_FONT_SIZE, c='k')
                        world_ann_list.append(status_ann)
                return axes

            anim = FuncAnimation(fig, update, len(team_traj[team[0].name]))
            out_file = os.path.join(file_name, f'traj{traj_i}_{self.width}x{self.height}_v{self.num_exist}.gif')
            anim.save(out_file, dpi=300, fps=1, writer='pillow')
