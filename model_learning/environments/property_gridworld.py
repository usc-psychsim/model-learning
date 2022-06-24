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
from model_learning import Trajectory, TeamTrajectory
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

        self.p_state = []
        for loc_i in range(self.width * self.height):
            p = self.world.defineState(WORLD, PROPERTY_FEATURE + f'{loc_i}',
                                       int, 0, len(PROPERTY_LIST) - 1, description=f'Each location\'s property')
            self.world.setFeature(p, 0)
            self.p_state.append(p)
        # print(self.p_state)

        self.g = self.world.defineState(WORLD, GOAL_FEATURE, int, 0, len(self.exist_locations),
                                        description=f'GOAL: # of cleared locations')
        self.world.setFeature(self.g, 0)

    def vis_agent_dynamics_in_xy(self):
        for k, v in self.world.dynamics.items():
            if type(k) is ActionSet:
                print('----')
                print(k)
                for kk, vv in v.items():
                    print(kk)
                    print(vv)

    def get_property_features(self):
        p_features = []
        for loc_i in range(self.width * self.height):
            p = stateKey(WORLD, PROPERTY_FEATURE + f'{loc_i}')
            p_features.append(p)
        return p_features

    def remove_action(self, agent: Agent, action: str):
        illegal_action = agent.find_action({'action': action})
        agent.setLegal(illegal_action, makeTree(False))
        self.agent_actions[agent.name].remove(illegal_action)


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
        legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), list(range(self.width * self.height)), 0)}
        for i, loc in enumerate(list(range(self.width * self.height))):
            p_loc = self.p_state[loc]
            sub_legal_dict = {'if': equalRow(p_loc, 0), True: True, False: False}
            legal_dict[i] = sub_legal_dict
        legal_dict[None] = False
        agent.setLegal(action, makeTree(legal_dict))

        for loc_i in range(self.width * self.height):
            p_loc = self.p_state[loc_i]
            search_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), loc_i, 0)}
            if loc_i in self.exist_locations:
                search_tree[True] = setToConstantMatrix(p_loc, 1)
            else:
                search_tree[True] = setToConstantMatrix(p_loc, 4)
            search_tree[False] = noChangeMatrix(p_loc)
            self.world.setDynamics(p_loc, action, makeTree(search_tree))
        self.agent_actions[agent.name].append(action)

        action = agent.addAction({'verb': 'handle', 'action': 'rescue'})
        legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.exist_locations, 0)}
        for i, loc in enumerate(self.exist_locations):
            p_loc = self.p_state[loc]
            sub_legal_dict = {'if': equalRow(p_loc, 1), True: True, False: False}
            legal_dict[i] = sub_legal_dict
        legal_dict[None] = False
        agent.setLegal(action, makeTree(legal_dict))

        for loc_i in self.exist_locations:
            p_loc = self.p_state[loc_i]
            search_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), loc_i, 0)}
            search_tree[True] = setToConstantMatrix(p_loc, 2)
            search_tree[False] = noChangeMatrix(p_loc)
            self.world.setDynamics(p_loc, action, makeTree(search_tree))
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

            legal_dict = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2)}
            sub_legal_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), self.exist_locations, 0)}
            for exist_i, exist_loc in enumerate(self.exist_locations):
                p_loc = self.p_state[exist_loc]
                subsub_legal_dict = {'if': equalRow(p_loc, 2), True: True, False: False}
                sub_legal_dict[exist_i] = subsub_legal_dict
            sub_legal_dict[None] = False
            legal_dict[True] = sub_legal_dict
            legal_dict[False] = False
            agents[i].setLegal(action_list[i], makeTree(legal_dict))

            for loc_i in self.exist_locations:
                p_loc = self.p_state[loc_i]
                tree_dict = {'if': equalRow(actionKey(agents[j].name, True), action_list[j])}
                subtree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), loc_i, 0),
                                True: setToConstantMatrix(p_loc, 3),
                                False: noChangeMatrix(p_loc)}
                tree_dict[True] = subtree_dict
                tree_dict[False] = noChangeMatrix(p_loc)
                self.world.setDynamics(p_loc, action_list[i], makeTree(tree_dict))

            # update GOAL_FEATURE
            for loc_i in self.exist_locations:
                p_loc = self.p_state[loc_i]
                tree_dict = {'if': KeyedPlane(KeyedVector({makeFuture(p_loc): 1}), 3, 0),
                             True: incrementMatrix(self.g, 1/len(agents)),
                             False: noChangeMatrix(self.g)}
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

    def generate_team_trajectories(self, team: List[Agent], trajectory_length: int,
                                   n_trajectories: int = 1, init_feats: Optional[Dict[str, Any]] = None,
                                   model: Optional[str] = None, select: bool = True,
                                   selection: Optional[
                                       Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                   horizon: Optional[int] = None, threshold: Optional[float] = None,
                                   processes: Optional[int] = -1, seed: int = 0, verbose: bool = False,
                                   use_tqdm: bool = True) -> List[TeamTrajectory]:
        assert len(team) > 0, 'No agent in the team'

        x, y = self.get_location_features(team[0])
        p_features = self.get_property_features()
        # if not specified, set random values for x, y pos, and property
        if init_feats is None:
            init_feats = {}
        if x not in init_feats:
            init_feats[x] = None
        if y not in init_feats:
            init_feats[y] = None
        for p in p_features:
            if p not in init_feats:
                init_feats[p] = 0

        # generate trajectories starting from random locations in the property gridworld
        return generate_team_trajectories(team, n_trajectories, trajectory_length,
                                          init_feats, model, select, horizon, selection, threshold,
                                          processes, seed, verbose, use_tqdm)

    def play_team_trajectories(self, team_trajectories: List[TeamTrajectory],
                               team: List[Agent],
                               file_name: str,
                               title: str = 'Team Trajectories'):
        assert len(team_trajectories) > 0 and len(team) > 0

        for agent in team:
            x, y = self.get_location_features(agent)
            assert x in self.world.variables, f'Agent \'{agent.name}\' does not have x location feature'
            assert y in self.world.variables, f'Agent \'{agent.name}\' does not have y location feature'
        p_features = self.get_property_features()
        for p in p_features:
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
                    ax.annotate('({},{})'.format(x, y), xy=(x + .05, y + .12), fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
                # turn off tick labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', adjustable='box')
                if ag_i == 0:
                    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

            # plots trajectories
            l_traj = len(team_traj)
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
            for i, tsa in enumerate(team_traj):
                for ag_i, agent in enumerate(team):
                    x, y = self.get_location_features(agent)
                    action = actionKey(agent.name)
                    x_t = tsa.world.getFeature(x, unique=True)
                    y_t = tsa.world.getFeature(y, unique=True)
                    a = tsa.world.getFeature(action, unique=True)
                    a = a['subject'] + '-' + a['action']
                    if ag_i == 0:
                        for loc in range(self.width * self.height):
                            p_t = tsa.world.getFeature(self.p_state[loc], unique=True)
                            world_ps[loc][i] = PROPERTY_LIST[p_t]
                    team_xs[agent.name][i] = x_t + 0.5
                    team_ys[agent.name][i] = y_t + 0.5
                    team_as[agent.name][i] = a
                # for ag_i, agent in enumerate(team)

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
                    ts_ann = axes[ag_i].annotate(f'Time Step: {ti}', xy=(.05, .03),
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

            anim = FuncAnimation(fig, update, len(team_traj))
            out_file = os.path.join(file_name, f'traj{traj_i}_{self.width}x{self.height}_v{self.num_exist}.gif')
            anim.save(out_file, dpi=300, fps=1, writer='pillow')
