import itertools as it
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from typing import Dict, Literal, Any, NamedTuple

from model_learning import TeamTrajectory
from model_learning.environments.gridworld import GridWorld, NOOP_ACTION, RIGHT_ACTION, LEFT_ACTION, UP_ACTION, \
    DOWN_ACTION
from model_learning.features.linear import *
from model_learning.trajectory import generate_team_trajectories, generate_expert_learner_trajectories
from model_learning.util.plot import distinct_colors
from psychsim.action import ActionSet
from psychsim.pwl import incrementMatrix, noChangeMatrix, stateKey, equalRow, makeFuture, equalFeatureRow, WORLD

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'

EVACUATE_ACTION = 'evacuate'
CALL_ACTION = 'call'
TRIAGE_ACTION = 'triage'
SEARCH_ACTION = 'search'

LOC_VIC_STATUS_FEATURE = 'LocVicStatus'
DIST_TO_VIC_FEATURE = 'DistToVic'
DIST_TO_HELP_FEATURE = 'DistToHelp'
VIC_CLEAR_COMB_FEATURE = 'VicClearCombIdx'
HELP_LOC_FEATURE = 'HelpLocIdx'
VICS_CLEARED_FEATURE = 'NumVicsCleared'
NO_VICS_FEATURE = 'NumNoVics'

UNKNOWN_STR = 'unknown'
FOUND_STR = 'found'
READY_STR = 'ready'
CLEAR_STR = 'clear'
EMPTY_STR = 'empty'
VICTIM_STATUS = [UNKNOWN_STR, FOUND_STR, READY_STR, CLEAR_STR, EMPTY_STR]

TITLE_FONT_SIZE = 12
VALUE_CMAP = 'gray'  # 'viridis' # 'inferno'
TRAJECTORY_LINE_WIDTH = 1
LOC_FONT_SIZE = 6
LOC_FONT_COLOR = 'darkgrey'
NOTES_FONT_SIZE = 6
NOTES_FONT_COLOR = 'dimgrey'
POLICY_MARKER_COLOR = 'dimgrey'


class AgentOptions(NamedTuple):
    """
    Options for agent actions and features, to create different search-and-rescue behaviors.
    """
    noop_action: bool
    search_action: bool
    triage_action: bool
    call_action: bool
    num_empty_feature: bool
    dist_to_vic_feature: bool
    dist_to_help_feature: bool


class SearchRescueGridWorld(GridWorld):
    """
    Represents a gridworld environment containing victims that need to be found and evacuated.
    Each location has a property state, used to keep track of each victim status.
    """

    def __init__(self,
                 world: World,
                 width: int,
                 height: int,
                 num_victims: int,
                 name: str = '',
                 vics_cleared_feature: bool = False,
                 seed: int = 0):
        """
        Creates a new gridworld.
        :param World world: the PsychSim world associated with this gridworld.
        :param int width: the number of horizontal cells.
        :param int height: the number of vertical cells.
        :param int num_victims: the number of victims in the world.
        :param str name: the name of this gridworld, used as a prefix for features, actions, etc.
        :param bool vics_cleared_feature: whether to create a feature that counts the number of victims cleared.
        :param int seed: the seed used to initialize the random number generator to create and place the objects.
        """
        super().__init__(world, width, height, name)

        self.num_victims: int = num_victims
        self.vics_cleared_feature: bool = vics_cleared_feature

        self.agent_options: Dict[str, AgentOptions] = {}  # to store the different agent options

        num_locs = self.width * self.height
        all_locs = list(range(num_locs))

        # create victim status feature for each location
        self.vic_status_features: List[str] = []
        for loc_idx in all_locs:
            f = self.world.defineState(WORLD, self.get_loc_vic_status_feature(loc_idx, key=False),
                                       list, VICTIM_STATUS, description=f'Location {loc_idx}\'s victim status')
            self.world.setFeature(f, UNKNOWN_STR)  # initially no one knows where victims are
            self.vic_status_features.append(f)

        # determines locations of victims in the environment
        rng = np.random.RandomState(seed)
        all_locs = np.arange(width * height)
        self.victim_locs = rng.choice(all_locs, num_victims, replace=False).tolist()
        self.empty_locs = list(set(all_locs) - set(self.victim_locs))

        # creates combination index of clear status of victims in each victim location:
        # False: not clear, i.e., a victim was found in the location
        # True: clear, i.e., a victim was not found at a location or was meanwhile rescued
        self.vic_clear_combs: List[Dict[int, bool]] = []
        for comb in it.product([False, True], repeat=len(self.victim_locs)):
            vic_clear_loc_comb = {self.victim_locs[i]: clear for i, clear in enumerate(comb)}
            self.vic_clear_combs.append(vic_clear_loc_comb)

        # precompute distance to the closest vic location (not cleared) for each location for each vic clear combination
        self.clear_counter: List[int] = []
        self.dists_to_vic: List[np.ndarray] = []
        for clear_status in self.vic_clear_combs:
            self.clear_counter.append(np.sum(list(clear_status.values())).item())  # num. cleared locs
            dists_to_vic = np.ones(num_locs)
            vic_locs = [k for k, v in clear_status.items() if not v]  # where there are known, un-evacuated vics
            for loc_idx in all_locs:
                if len(vic_locs) > 0:
                    dists_to_vic[loc_idx] = self._dist_to_closest_loc(loc_idx, vic_locs)
            self.dists_to_vic.append(dists_to_vic)

        # precompute (normalized) distance from all vic locations to other locations (used in the dist_to_help feature)
        self.dists_to_loc: np.ndarray = np.ones((num_victims, num_locs))
        for i, vic_loc_idx in enumerate(self.victim_locs):  # compute only for vic locations
            for loc_idx in all_locs:
                self.dists_to_loc[i, loc_idx] = self._dist_to_closest_loc(loc_idx, [vic_loc_idx])

        # define additional world features:
        # - victims cleared status combination index
        self.clear_comb = self.world.defineState(WORLD, self.get_vic_clear_comb_feature(key=False),
                                                 int, 0, len(self.dists_to_vic) - 1,
                                                 description=f'Index of combination of victims\' cleared status')
        self.world.setFeature(self.clear_comb, len(self.dists_to_vic) - 1)  # start with all victims cleared/none found

        # - help location index
        self.help_loc = self.world.defineState(WORLD, self.get_help_loc_feature(key=False),
                                               int, -1, len(self.victim_locs) - 1,
                                               description=f'Index of location in which help was requested')
        self.world.setFeature(self.help_loc, -1)  # help has not been requested in any location

        if self.vics_cleared_feature:
            # - number of victims that have been cleared
            self.num_clear_vics = self.world.defineState(WORLD, self.get_vics_cleared_feature(key=False),
                                                         int, 0, len(self.victim_locs),
                                                         description='Number of cleared victims')
            self.world.setFeature(self.num_clear_vics, 0)

    def get_loc_vic_status_feature(self, loc_idx: int, key: bool = True) -> str:
        """
        Gets the victim status feature for the given location.
        :param int loc_idx: the location index.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the location's victim status feature.
        """
        f = f'{self.name}_{LOC_VIC_STATUS_FEATURE}_{loc_idx}'
        return stateKey(WORLD, f) if key else f

    def get_vic_clear_comb_feature(self, key: bool = True):
        """
        Gets the victim cleared combination index feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the victim cleared combination index feature.
        """
        f = f'{self.name}_{VIC_CLEAR_COMB_FEATURE}'
        return stateKey(WORLD, f) if key else f

    def get_help_loc_feature(self, key: bool = True):
        """
        Gets the help requested location index feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the help requested location index feature.
        """
        f = f'{self.name}_{HELP_LOC_FEATURE}'
        return stateKey(WORLD, f) if key else f

    def get_vics_cleared_feature(self, key: bool = True):
        """
        Gets the victims cleared count feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the victims cleared count feature.
        """
        f = f'{self.name}_{VICS_CLEARED_FEATURE}'
        return stateKey(WORLD, f) if key else f

    def get_dist_to_vic_feature(self, agent: Agent, key: bool = True):
        """
        Gets the distance to the closest victim (non-clear location) feature for the given agent.
        :param Agent agent: the agent for which to get the feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the distance to the closest victim feature.
        """
        f = f'{self.name}_{DIST_TO_VIC_FEATURE}'
        return stateKey(agent.name, f) if key else f

    def get_dist_to_help_feature(self, agent: Agent, key: bool = True):
        """
        Gets the distance to the location where help was requested feature for the given agent.
        :param Agent agent: the agent for which to get the feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the distance to the help location feature.
        """
        f = f'{self.name}_{DIST_TO_HELP_FEATURE}'
        return stateKey(agent.name, f) if key else f

    def get_empty_feature(self, agent: Agent, key: bool = True):
        """
        Gets the number of no victim locations found feature for the given agent.
        :param Agent agent: the agent for which to get the feature.
        :param bool key: whether to return a PsychSim state feature key (`True`) or just the feature name (`False`).
        :rtype: str
        :return: the number of no victim locations found feature.
        """
        f = f'{self.name}_{NO_VICS_FEATURE}'
        return stateKey(agent.name, f) if key else f

    def add_search_and_rescue_dynamics(self, agent: Agent, options: AgentOptions) -> List[ActionSet]:
        """
        Adds the PsychSim action dynamics for the given agent to search for and triage victims in the environment.
        Also adds the gridworld navigation actions.
        :param Agent agent: the agent to add the action search and rescue dynamics to.
        :param AgentOptions options: the search-and-rescue action and feature options for this agent.
        :rtype: list[ActionSet]
        :return: a list containing the agent's newly created actions.
        """

        assert agent.name not in self.agent_options, f'SAR dynamics have already been set to agent \'{agent.name}\''
        self.agent_options[agent.name] = options

        # movement dynamics
        self.add_agent_dynamics(agent, noop_action=options.noop_action, visit_count_features=False)
        x, y = self.get_location_features(agent)

        n_ci = len(self.dists_to_vic)  # number of victim (status) combinations
        non_clear_vic_comb_idxs = list(range(n_ci - 1))
        n_locs = self.width * self.height  # number of locations in the environment
        all_locs = list(range(n_locs))
        all_vic_loc_idxs = list(range(self.num_victims))

        # noop/wait is allowed is allowed only when all victims are triaged
        if options.noop_action:
            action = agent.find_action({'action': NOOP_ACTION})
            legal_dict = {'if': equalRow(self.clear_comb, len(self.dists_to_vic) - 1), True: True, False: False}
            agent.setLegal(action, makeTree(legal_dict))

        # ==================================================
        # dynamics of distance to the closest victim (non-cleared) location feature
        if options.dist_to_vic_feature:
            d2v = self.world.defineState(agent.name, self.get_dist_to_vic_feature(agent, key=False),
                                         float, 0, 1,
                                         description=f'Distance to the closest victim (non-clear) location')
            self.world.setFeature(d2v, 1)  # initially unknown victims, so maximal distance

            # if all victims cleared, distance is maximal (1)
            ci_dict = {'if': KeyedPlane(KeyedVector({self.clear_comb: 1}), n_ci - 1, 0),
                       True: setToConstantMatrix(d2v, 1)}

            # otherwise set according to vic clear combination and current location indices
            tree_dict = {'if': KeyedPlane(KeyedVector({self.clear_comb: 1}), non_clear_vic_comb_idxs, 0),
                         None: noChangeMatrix(d2v)}
            for vic_comb_idx in non_clear_vic_comb_idxs:
                sub_tree_dict = {
                    'if': KeyedPlane(KeyedVector({makeFuture(x): 1, makeFuture(y): self.width}), all_locs, 0),
                    None: noChangeMatrix(d2v)}
                for loc_idx in all_locs:
                    sub_tree_dict[loc_idx] = setToConstantMatrix(d2v, self.dists_to_vic[vic_comb_idx][loc_idx])
                tree_dict[vic_comb_idx] = sub_tree_dict

            ci_dict[False] = tree_dict

            # set the dynamics to all movement actions
            for move in {RIGHT_ACTION, LEFT_ACTION, UP_ACTION, DOWN_ACTION}:
                action = agent.find_action({'action': move})
                self.world.setDynamics(d2v, action, makeTree(ci_dict))

        # ==================================================
        # dynamics of distance to the help location feature
        if options.dist_to_help_feature:
            d2h = self.world.defineState(agent.name, self.get_dist_to_help_feature(agent, key=False),
                                         float, 0, 1, description=f'distance to help the others')
            self.world.setFeature(d2h, 1)  # initially unknown victims, so maximal distance

            # test cur help/victim location against current agent location
            tree_dict = {'if': KeyedPlane(KeyedVector({makeFuture(self.help_loc): 1}), all_vic_loc_idxs, 0),
                         None: setToConstantMatrix(d2h, 1)}  # no help request location, so maximal distance

            for vic_loc_idx in all_vic_loc_idxs:
                sub_tree_dict = {
                    'if': KeyedPlane(KeyedVector({makeFuture(x): 1, makeFuture(y): self.width}), all_locs, 0),
                    None: noChangeMatrix(d2h)}  # unknown loc, so don't change
                for loc_idx in all_locs:
                    sub_tree_dict[loc_idx] = setToConstantMatrix(d2h, self.dists_to_loc[vic_loc_idx][loc_idx])
                tree_dict[vic_loc_idx] = sub_tree_dict

            # set the dynamics to all movement actions
            for move in {RIGHT_ACTION, LEFT_ACTION, UP_ACTION, DOWN_ACTION}:
                action = agent.find_action({'action': move})
                self.world.setDynamics(d2h, action, makeTree(tree_dict))

        # ==================================================
        # dynamics of victim search action
        if options.search_action:
            action = agent.addAction({'verb': 'handle', 'action': SEARCH_ACTION})
            self.agent_actions[agent.name].append(action)

            # search is valid only when victim status at current location is unknown
            legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), all_locs, 0),
                          None: False}  # invalid location, so illegal
            for loc_idx in all_locs:
                vic_status = self.vic_status_features[loc_idx]
                legal_dict[loc_idx] = {'if': equalRow(vic_status, UNKNOWN_STR), True: True, False: False}
            agent.setLegal(action, makeTree(legal_dict))

            # search action sets victim status to either found or empty/no victim
            for loc_idx in all_locs:
                vic_status = self.vic_status_features[loc_idx]
                next_status = setToConstantMatrix(vic_status, FOUND_STR if loc_idx in self.victim_locs else EMPTY_STR)
                search_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), loc_idx, 0),
                               True: next_status,
                               False: noChangeMatrix(vic_status)}
                self.world.setDynamics(vic_status, action, makeTree(search_tree))

            # ==================================================
            # dynamics of victim clear combination index, updated once a victim is found
            tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.victim_locs, 0),
                         None: noChangeMatrix(self.clear_comb)}
            for i, loc_idx in enumerate(self.victim_locs):
                # get victim clear combination indices in which this location is clear (victim not found)
                vic_comb_idxs = self._get_vic_clear_comb_idxs(loc_idx, clear_status=True)
                subtree_dict = {'if': KeyedPlane(KeyedVector({self.clear_comb: 1}), vic_comb_idxs, 0),
                                None: noChangeMatrix(self.clear_comb)}
                for j, vic_comb_idx in enumerate(vic_comb_idxs):
                    # set next vic comb idx
                    subtree_dict[j] = setToConstantMatrix(
                        self.clear_comb,
                        self._get_next_vic_found_comb_idx(vic_comb_idx, loc_idx, next_clear_status=False))
                tree_dict[loc_idx] = subtree_dict
            self.world.setDynamics(self.clear_comb, action, makeTree(tree_dict))

            # ==================================================
            # dynamics of distance to victim, updated once a victim is found
            # checks for current vic clear combination against current agent location
            if options.dist_to_vic_feature:
                d2v = self.get_dist_to_vic_feature(agent, key=True)

                ci_dict = {'if': KeyedPlane(KeyedVector({makeFuture(self.clear_comb): 1}), n_ci - 1, 0),
                           True: setToConstantMatrix(d2v, 1)}  # if no victims/all clear, distance is maximal

                tree_dict = {
                    'if': KeyedPlane(KeyedVector({makeFuture(self.clear_comb): 1}), non_clear_vic_comb_idxs, 0),
                    None: noChangeMatrix(d2v)}
                for vic_comb_idx in non_clear_vic_comb_idxs:
                    sub_tree_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), all_locs, 0),
                                     None: noChangeMatrix(d2v)}
                    for loc_idx in all_locs:
                        sub_tree_dict[loc_idx] = setToConstantMatrix(d2v, self.dists_to_vic[vic_comb_idx][loc_idx])
                    tree_dict[vic_comb_idx] = sub_tree_dict

                ci_dict[False] = tree_dict
                self.world.setDynamics(d2v, action, makeTree(ci_dict))

            # ==================================================
            # dynamics of number of locations found without victims, updated once a victim is not found after search
            if options.num_empty_feature:
                f = self.world.defineState(agent.name, self.get_empty_feature(agent, key=False),
                                           int, 0, len(self.empty_locs),
                                           description=f'Number of empty locations, i.e., with no victims found')
                self.world.setFeature(f, 0)

                for loc_idx in self.empty_locs:
                    vic_status = self.vic_status_features[loc_idx]
                    tree_dict = {'if': KeyedPlane(KeyedVector({makeFuture(vic_status): 1}), EMPTY_STR, 0),
                                 True: incrementMatrix(f, 1),
                                 False: noChangeMatrix(f)}
                    self.world.setDynamics(f, action, makeTree(tree_dict))

        # ==================================================
        # dynamics of victim triage action
        if options.triage_action:
            action = agent.addAction({'verb': 'handle', 'action': TRIAGE_ACTION})
            self.agent_actions[agent.name].append(action)

            # triage is valid when victim is found
            legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.victim_locs, 0),
                          None: False}
            for i, loc_idx in enumerate(self.victim_locs):
                vic_status = self.vic_status_features[loc_idx]
                legal_dict[i] = {'if': equalRow(vic_status, FOUND_STR), True: True, False: False}
            agent.setLegal(action, makeTree(legal_dict))

            # change victim status to ready once triaged
            for loc_idx in self.victim_locs:
                vic_status = self.vic_status_features[loc_idx]
                search_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), loc_idx, 0),
                               True: setToConstantMatrix(vic_status, READY_STR),
                               False: noChangeMatrix(vic_status)}
                self.world.setDynamics(vic_status, action, makeTree(search_tree))

        # ==================================================
        # dynamics of victim call action
        if options.call_action:
            action = agent.addAction({'verb': 'handle', 'action': CALL_ACTION})
            self.agent_actions[agent.name].append(action)

            # call is valid when victim is ready
            legal_dict = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.victim_locs, 0),
                          None: False}
            for i, loc_idx in enumerate(self.victim_locs):
                vic_status = self.vic_status_features[loc_idx]
                legal_dict[i] = {'if': equalRow(vic_status, READY_STR), True: True, False: False}
            agent.setLegal(action, makeTree(legal_dict))

            # change help location to agent's current vic location index once agent calls for help
            call_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.width}), self.victim_locs, 0),
                         None: setToConstantMatrix(self.help_loc, -1)}  # if not a victim location, set invalid idx
            for vic_loc_idx in all_vic_loc_idxs:
                call_tree[vic_loc_idx] = setToConstantMatrix(self.help_loc, vic_loc_idx)
            self.world.setDynamics(self.help_loc, action, makeTree(call_tree))

        return self.agent_actions[agent.name]

    def add_collaboration_dynamics(self, agents: List[Agent]):
        """
        Adds the PsychSim collaboration action dynamics for the given agents to allow them to evacuate victims from
        the environment.
        :param list[Agent] agents: the list of agents to add the collaboration dynamics to.
        """

        assert len(agents) == 2, f'Collaboration dynamics limited to 2 agents, {len(agents)} given'

        for i, agent in enumerate(agents):
            assert agent.name in self.agent_actions and agent.name in self.agent_options, \
                f'Agent \'{agent.name}\' is not associated with this environment'

        n_ci = len(self.dists_to_vic)  # number of victim (status) combinations
        non_clear_vic_comb_idxs = list(range(n_ci - 1))
        all_vic_comb_idxs = list(range(n_ci))
        n_locs = self.width * self.height  # number of locations in the environment
        all_locs = list(range(n_locs))

        # create evacuate action for both agents
        evac_action = {'verb': 'handle', 'action': EVACUATE_ACTION}
        agent_actions = []
        for i, agent in enumerate(agents):
            if evac_action not in self.agent_actions[agent.name]:
                action = agent.addAction(evac_action)
                agent_actions.append(action)
                self.agent_actions[agent.name].append(action)

        # set dynamics for each agent based on behavior of other agent
        for i, agent in enumerate(agents):
            j = -(i - 1)
            agent1 = agents[i]
            agent2 = agents[j]
            x1, y1 = self.get_location_features(agent1)
            x2, y2 = self.get_location_features(agent2)

            # evacuate is legal only when two agents are at the same location with a "ready" victim
            legal_dict = {'if': equalFeatureRow(x1, x2) & equalFeatureRow(y1, y2),
                          False: False}
            sub_legal_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), self.victim_locs, 0),
                              None: False}
            for k, loc_idx in enumerate(self.victim_locs):
                vic_status = self.vic_status_features[loc_idx]
                sub_legal_dict[k] = {'if': equalRow(vic_status, READY_STR), True: True, False: False}
            legal_dict[True] = sub_legal_dict
            agents[i].setLegal(agent_actions[i], makeTree(legal_dict))

            if i == 0:  # only one agent is needed to update global/common features

                # evacuate changes the victim status from ready to clear, when *both* agents perform evacuate
                for loc_idx in self.victim_locs:
                    vic_status = self.vic_status_features[loc_idx]
                    tree_dict = {'if': (equalRow(actionKey(agents[j].name, True), agent_actions[j]) &
                                        KeyedPlane(KeyedVector({x1: 1, y1: self.width}), loc_idx, 0)),
                                 True: setToConstantMatrix(vic_status, CLEAR_STR),
                                 False: noChangeMatrix(vic_status)}
                    self.world.setDynamics(vic_status, agent_actions[i], makeTree(tree_dict))

                # ==================================================
                # dynamics of victim clear combination index, updated once a victim is evacuated/cleared by both agents
                ci_dict = {'if': equalRow(actionKey(agents[j].name, True), agent_actions[j]),
                           False: noChangeMatrix(self.clear_comb)}
                tree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), self.victim_locs, 0),
                             None: noChangeMatrix(self.clear_comb)}
                for k, loc_idx in enumerate(self.victim_locs):
                    # get victim clear combination indices in which this location is not clear (victim was found)
                    vic_comb_idxs = self._get_vic_clear_comb_idxs(loc_idx, clear_status=False)
                    subtree_dict = {'if': KeyedPlane(KeyedVector({self.clear_comb: 1}), vic_comb_idxs, 0),
                                    None: noChangeMatrix(self.clear_comb)}
                    for v, vic_comb_idx in enumerate(vic_comb_idxs):
                        # set next vic comb idx
                        subtree_dict[v] = setToConstantMatrix(
                            self.clear_comb,
                            self._get_next_vic_found_comb_idx(vic_comb_idx, loc_idx, next_clear_status=True))
                    tree_dict[k] = subtree_dict
                ci_dict[True] = tree_dict
                self.world.setDynamics(self.clear_comb, agent_actions[i], makeTree(ci_dict))

                # ==================================================
                # dynamics of number of cleared victims, updated once a victim is evacuated/cleared by both agents
                if self.vics_cleared_feature:
                    tree_dict = {'if': KeyedPlane(KeyedVector({makeFuture(self.clear_comb): 1}), all_vic_comb_idxs, 0),
                                 None: noChangeMatrix(self.num_clear_vics)}
                    for vic_comb_idx in all_vic_comb_idxs:
                        # num vics clear is pre-computed for each vic clear comb idx
                        tree_dict[vic_comb_idx] = setToConstantMatrix(
                            self.num_clear_vics, self.clear_counter[vic_comb_idx])
                    self.world.setDynamics(self.num_clear_vics, agent_actions[i], makeTree(tree_dict))

            # ==================================================
            # dynamics of distance to closest victim, updated once a victim is evacuated/cleared by both agents
            if self.agent_options[agent.name].dist_to_vic_feature:
                d2v = self.get_dist_to_vic_feature(agents[i], key=True)
                ci_dict = {'if': KeyedPlane(KeyedVector({makeFuture(self.clear_comb): 1}), n_ci, 0),
                           True: setToConstantMatrix(d2v, 1)}  # no victims, so maximal distance
                # checks vic clear combination against current agent location (distance pre-computed)
                tree_dict = {
                    'if': KeyedPlane(KeyedVector({makeFuture(self.clear_comb): 1}), non_clear_vic_comb_idxs, 0),
                    None: noChangeMatrix(d2v)}
                for vic_comb_idx in non_clear_vic_comb_idxs:
                    sub_tree_dict = {'if': KeyedPlane(KeyedVector({x1: 1, y1: self.width}), all_locs, 0),
                                     None: noChangeMatrix(d2v)}
                    for loc_idx in all_locs:
                        sub_tree_dict[loc_idx] = setToConstantMatrix(d2v, self.dists_to_vic[vic_comb_idx][loc_idx])
                    tree_dict[vic_comb_idx] = sub_tree_dict
                ci_dict[False] = tree_dict
                self.world.setDynamics(d2v, agent_actions[i], makeTree(ci_dict))

    def _get_vic_status_features(self) -> List[str]:
        return [self.get_loc_vic_status_feature(loc_idx, key=True) for loc_idx in range(self.width * self.height)]

    def _dist_to_closest_loc(self, loc_idx: int, target_loc_idxs: List[int]):
        # computes closest Manhattan distance between a location and a set of locations
        target_loc_idxs = set(target_loc_idxs)
        if loc_idx in target_loc_idxs:
            return 0
        dist = self.width + self.height - 2
        curr_x, curr_y = self.idx_to_xy(loc_idx)
        for loc in target_loc_idxs:
            loc_x, loc_y = self.idx_to_xy(loc)
            _dist = abs(curr_x - loc_x) + abs(curr_y - loc_y)
            if _dist < dist:
                dist = _dist
        return dist / (self.width + self.height - 2)

    def _get_vic_clear_comb_idxs(self, loc_idx: int, clear_status: bool) -> List[int]:
        # gets all vic clear combination indices for the given location
        idxs = []
        if loc_idx not in self.victim_locs:
            return idxs  # not a victim location, so no indices

        for comb_idx, vic_clear_loc_comb in enumerate(self.vic_clear_combs):
            if vic_clear_loc_comb[loc_idx] == clear_status:
                idxs.append(comb_idx)  # adds index of combination in which the location is (not)clear
        return idxs

    def _get_next_vic_found_comb_idx(self, vic_comb_idx: int, loc_idx: int, next_clear_status: bool) -> int:
        # gets next index of vic clear combination when a victim is found at given location
        next_vic_clear_loc_comb = self.vic_clear_combs[vic_comb_idx].copy()
        next_vic_clear_loc_comb[loc_idx] = next_clear_status
        for comb_idx, vic_clear_loc_comb in enumerate(self.vic_clear_combs):
            if list(vic_clear_loc_comb.values()) == list(next_vic_clear_loc_comb.values()):
                return comb_idx

        raise ValueError(f'Could not find the next combination index for location: {loc_idx} '
                         f'given combination index: {vic_comb_idx} and next clear status: {next_clear_status}')

    def vis_agent_dynamics_in_xy(self):
        for k, v in self.world.dynamics.items():
            if type(k) is ActionSet:
                print('----')
                print(k)
                for kk, vv in v.items():
                    print(kk)
                    print(vv)

    def get_role_reward_vector(self, agent: Agent, roles: Dict[str, float] = None):
        """
        Function to get reward features and weights based on the agent role in the team
        @param agent: the target agent
        @param roles: agent roles in the team
        @return: list of reward features, list of reward weights
        """
        reward_features = []
        rf_weights = []

        if roles is None:
            wait_action = agent.find_action({'action': 'wait'})
            r_wait = ActionLinearRewardFeature('wait', agent, wait_action)
            reward_features.append(r_wait)
            rf_weights.append(1)
            return reward_features, rf_weights

        if 'Goal' in roles:  # scale -1 to 1
            d2c = self.get_d2c_feature(agent)
            r_d2c = NumericLinearRewardFeature(DIST_TO_VIC_FEATURE, d2c)
            reward_features.append(r_d2c)
            rf_weights.append(0.1 * roles['Goal'])

            if self.vics_cleared_feature:
                r_goal = NumericLinearRewardFeature(VICS_CLEARED_FEATURE, stateKey(WORLD, VICS_CLEARED_FEATURE))
                reward_features.append(r_goal)
                rf_weights.append(roles['Goal'])

            search_action = agent.find_action({'action': 'search'})
            r_search = ActionLinearRewardFeature('search', agent, search_action)
            reward_features.append(r_search)
            rf_weights.append(0.1 * roles['Goal'])

            triage_action = agent.find_action({'action': 'triage'})
            r_triage = ActionLinearRewardFeature('triage', agent, triage_action)
            reward_features.append(r_triage)
            rf_weights.append(0.3 * roles['Goal'])

            evacuate_action = agent.find_action({'action': 'evacuate'})
            r_evacuate = ActionLinearRewardFeature('evacuate', agent, evacuate_action)
            reward_features.append(r_evacuate)
            rf_weights.append(roles['Goal'])

            wait_action = agent.find_action({'action': 'wait'})
            r_wait = ActionLinearRewardFeature('wait', agent, wait_action)
            reward_features.append(r_wait)
            rf_weights.append(0.05 * roles['Goal'])

            call_action = agent.find_action({'action': 'call'})
            r_call = ActionLinearRewardFeature('call', agent, call_action)
            reward_features.append(r_call)
            rf_weights.append(0.05 * roles['Goal'])

            for act in {'right', 'left', 'up', 'down', 'wait', 'search', 'triage', 'evacuate'}:
                action = agent.find_action({'action': act})
                self.world.setDynamics(self.help_loc, action, makeTree(setToConstantMatrix(self.help_loc, -1)))

        if 'Navigator' in roles:
            d2h = self.get_dist_to_help_feature(agent, key=True)
            r_d2h = NumericLinearRewardFeature(DIST_TO_HELP_FEATURE, d2h)
            reward_features.append(r_d2h)
            rf_weights.append(roles['Navigator'])

            search_action = agent.find_action({'action': 'search'})
            r_search = ActionLinearRewardFeature('search', agent, search_action)
            reward_features.append(r_search)
            rf_weights.append(roles['Navigator'])

            if self.vics_cleared_feature:
                f = self.get_empty_feature(agent, key=True)
                r_navi = NumericLinearRewardFeature(NO_VICS_FEATURE, f)
                reward_features.append(r_navi)
                rf_weights.append(roles['Navigator'])

            evacuate_action = agent.find_action({'action': 'evacuate'})
            r_evacuate = ActionLinearRewardFeature('evacuate', agent, evacuate_action)
            reward_features.append(r_evacuate)
            rf_weights.append(2 * roles['Navigator'])

        return reward_features, rf_weights

    def add_agent_models(self, agent: Agent, roles: Dict[str, float], model_names: List[str]) -> Agent:
        """
        Function called to add models to agent
        @param Agent agent: the agent that has models
        @param roles: agent roles in the team, used to get reward features
        @param model_names: possible models of agent
        @return: the modified agent
        """
        for model_name in model_names:
            true_model = agent.get_true_model()
            model_name = f'{agent.name}_{model_name}'
            agent.addModel(model_name, parent=true_model)
            rwd_features, rwd_weights = self.get_role_reward_vector(agent, roles)
            agent_lrv = LinearRewardVector(rwd_features)
            rwd_weights = np.array(rwd_weights) / np.linalg.norm(rwd_weights, 1)
            if model_name == f'{agent.name}_Opposite':
                rwd_weights = -1. * np.array(rwd_weights)
                rwd_weights = np.array(rwd_weights) / np.linalg.norm(rwd_weights, 1)
            if model_name == f'{agent.name}_Uniform':
                rwd_weights = [1.] * len(rwd_weights)
                rwd_weights = np.array(rwd_weights) / np.linalg.norm(rwd_weights, 1)
            if model_name == f'{agent.name}_Random':
                rwd_weights = [0.] * len(rwd_weights)

            if agent.name == 'Medic':
                selected_feats = {'dc', 'triage', 'evacuate'}
            elif agent.name == 'Explorer':
                selected_feats = {'search'}
            else:
                selected_feats = {}
            if model_name == f'{agent.name}_Task':
                for rwd_i, rwd_feat in enumerate(agent_lrv.names):
                    if rwd_feat not in selected_feats:
                        rwd_weights[rwd_i] = 0
            if model_name == f'{agent.name}_Social':
                for rwd_i, rwd_feat in enumerate(agent_lrv.names):
                    if rwd_feat in selected_feats:
                        rwd_weights[rwd_i] = 0

            agent_lrv.set_rewards(agent, rwd_weights, model=model_name)
            print(agent.name, model_name, agent_lrv.names, rwd_weights)
        return agent

    def generate_team_trajectories(self, team: List[Agent], trajectory_length: int,
                                   n_trajectories: int = 1, init_feats: Optional[Dict[str, Any]] = None,
                                   model: Optional[str] = None, select: bool = True,
                                   selection: Optional[
                                       Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                   horizon: Optional[int] = None, threshold: Optional[float] = None,
                                   processes: Optional[int] = -1, seed: int = 0, verbose: bool = False,
                                   use_tqdm: bool = True) -> List[TeamTrajectory]:
        """
        Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
        :param List[Agent] team: the team of agents for which to record the actions.
        :param int trajectory_length: the length of the generated trajectories.
        :param int n_trajectories: the number of trajectories to be generated.
        :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
        trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
        values to choose from, a single value, or `None`, in which case a random value will be picked based on the
        feature's domain.
        :param str model: the agent model used to generate the trajectories.
        :param bool select: whether to select from stochastic states after each world step.
        :param str selection: the action selection criterion, to untie equal-valued actions.
        :param int horizon: the agent's planning horizon.
        :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int processes: number of processes to use. Follows `joblib` convention.
        :param int seed: the seed used to initialize the random number generator.
        :param bool verbose: whether to show information at each timestep during trajectory generation.
        :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
        :rtype: list[TeamTrajectory]
        :return: a list of team trajectories, each containing a list of state-team action pairs.
        """
        assert len(team) > 0, 'No agent in the team'

        # if not specified, set random values for x, y pos
        if init_feats is None:
            init_feats = {}
        for agent in team:
            x, y = self.get_location_features(agent)
            if x not in init_feats:
                init_feats[x] = None
            if y not in init_feats:
                init_feats[y] = None
        # generate trajectories starting from random locations in the property gridworld
        return generate_team_trajectories(team, n_trajectories, trajectory_length,
                                          init_feats, model, select, horizon, selection, threshold,
                                          processes, seed, verbose, use_tqdm)

    def generate_expert_learner_trajectories(self, expert_team: List[Agent], learner_team: List[Agent],
                                             trajectory_length: int, n_trajectories: int = 1,
                                             init_feats: Optional[Dict[str, Any]] = None,
                                             model: Optional[str] = None, select: bool = True,
                                             selection: Optional[
                                                 Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                             horizon: Optional[int] = None, threshold: Optional[float] = None,
                                             processes: Optional[int] = -1, seed: int = 0, verbose: bool = False,
                                             use_tqdm: bool = True) -> List[TeamTrajectory]:
        """
        Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
        :param List[Agent] expert_team: the team of agents that step the world.
        :param List[Agent] learner_team: the team of agents for which to record the actions.
        :param int trajectory_length: the length of the generated trajectories.
        :param int n_trajectories: the number of trajectories to be generated.
        :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
        trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
        values to choose from, a single value, or `None`, in which case a random value will be picked based on the
        feature's domain.
        :param str model: the agent model used to generate the trajectories.
        :param bool select: whether to select from stochastic states after each world step.
        :param str selection: the action selection criterion, to untie equal-valued actions.
        :param int horizon: the agent's planning horizon.
        :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int processes: number of processes to use. Follows `joblib` convention.
        :param int seed: the seed used to initialize the random number generator.
        :param bool verbose: whether to show information at each timestep during trajectory generation.
        :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
        :rtype: list[TeamTrajectory]
        :return: a list of trajectories, each containing a list of state-expert-learner-action pairs.
        """
        assert len(expert_team) > 0, 'No agent in the team'
        assert len(learner_team) > 0, 'No agent in the team'

        # if not specified, set random values for x, y pos
        if init_feats is None:
            init_feats = {}
        for agent in expert_team:
            x, y = self.get_location_features(agent)
            if x not in init_feats:
                init_feats[x] = None
            if y not in init_feats:
                init_feats[y] = None

        # generate trajectories starting from random locations in the property gridworld
        return generate_expert_learner_trajectories(expert_team, learner_team, n_trajectories, trajectory_length,
                                                    init_feats, model, select, horizon, selection, threshold,
                                                    processes, seed, verbose, use_tqdm)

    # generate .gif of trajectories
    def play_team_trajectories(self, team_trajectories: List[TeamTrajectory],
                               team: List[Agent],
                               file_name: str,
                               title: str = 'Team Trajectories'):
        assert len(team_trajectories) > 0 and len(team) > 0

        for agent in team:
            x, y = self.get_location_features(agent)
            assert x in self.world.variables, f'Agent \'{agent.name}\' does not have x location feature'
            assert y in self.world.variables, f'Agent \'{agent.name}\' does not have y location feature'
        p_features = self._get_vic_status_features()
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
                for x, y in it.product(range(self.width), range(self.height)):
                    ax.annotate('({},{})'.format(x, y), xy=(x + .05, y + .15), fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
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
            for loci, loc in enumerate(self.victim_locs):
                world_ps[loc] = [None] * l_traj
            for loci, loc in enumerate(self.empty_locs):
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
                            p_t = tsa.world.getFeature(self.vic_status_features[loc], unique=True)
                            world_ps[loc][i] = VICTIM_STATUS[p_t]
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
                for loci, loc in enumerate(self.victim_locs):
                    x, y = self.idx_to_xy(loc)
                    p = world_ps[loc][ti]
                    for ag_i, agent in enumerate(team):
                        status_ann = axes[ag_i].annotate(f'*V{loci + 1}\n{p}', xy=(x + .47, y + .3),
                                                         fontsize=NOTES_FONT_SIZE, c='k')
                        world_ann_list.append(status_ann)
                for loci, loc in enumerate(self.empty_locs):
                    x, y = self.idx_to_xy(loc)
                    p = world_ps[loc][ti]
                    for ag_i, agent in enumerate(team):
                        status_ann = axes[ag_i].annotate(f'\n{p}', xy=(x + .47, y + .3),
                                                         fontsize=NOTES_FONT_SIZE, c='k')
                        world_ann_list.append(status_ann)
                return axes

            anim = FuncAnimation(fig, update, len(team_traj))
            out_file = os.path.join(file_name, f'traj{traj_i}_{self.width}x{self.height}_v{self.num_victims}.gif')
            anim.save(out_file, dpi=300, fps=1, writer='pillow')