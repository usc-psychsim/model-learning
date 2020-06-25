import copy
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.world import World
from psychsim.pwl import makeTree, incrementMatrix, noChangeMatrix, thresholdRow, stateKey, VectorDistributionSet, \
    KeyedPlane, KeyedVector, rewardKey, setToConstantMatrix
from model_learning.util.plot import distinct_colors
from model_learning.trajectory import generate_trajectories

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

X_FEATURE = 'x'
Y_FEATURE = 'y'

ACTION_NO_OP = 0
ACTION_RIGHT_IDX = 1
ACTION_LEFT_IDX = 2
ACTION_UP_IDX = 3
ACTION_DOWN_IDX = 4

# stores shift values for placement and markers for each action
MARKERS_INC = {
    ACTION_RIGHT_IDX: (.7, .5, CARETRIGHTBASE),
    ACTION_UP_IDX: (.5, .7, CARETUPBASE),
    ACTION_LEFT_IDX: (.3, .5, CARETLEFTBASE),
    ACTION_DOWN_IDX: (.5, .3, CARETDOWNBASE),
    ACTION_NO_OP: (.5, .5, '.')  # stand still action
}

ACTION_NAMES = {
    ACTION_RIGHT_IDX: 'right',
    ACTION_UP_IDX: 'up',
    ACTION_LEFT_IDX: 'left',
    ACTION_DOWN_IDX: 'down',
    ACTION_NO_OP: 'no-op',
}

TITLE_FONT_SIZE = 12
VALUE_CMAP = 'gray'  # 'viridis' # 'inferno'
TRAJECTORY_LINE_WIDTH = 1
LOC_FONT_SIZE = 6
LOC_FONT_COLOR = 'darkgrey'
NOTES_FONT_SIZE = 8
NOTES_FONT_COLOR = 'dimgrey'
POLICY_MARKER_COLOR = 'dimgrey'


class GridWorld(object):
    """
    Represents a simple gridworld environment in which agents can move in the 4 cardinal directions or stay in the same
    location.
    """

    def __init__(self, world, width, height, name=''):
        """
        Creates a new gridworld.
        :param World world: the PsychSim world associated with this gridworld.
        :param int width: the number of horizontal cells.
        :param int height: the number of vertical cells.
        :param str name: the name of this gridworld, used as a suffix for features, actions, etc.
        """
        self.world = world
        self.width = width
        self.height = height
        self.name = name

        self.agent_actions = {}

    def add_agent_dynamics(self, agent):
        """
        Adds the PsychSim action dynamics for the given agent to move in this gridworld.
        The 4 cardinal movement actions plus a stay-still/no-op action is added.
        Also registers those actions for later usage.
        :param Agent agent: the agent to which add the action movement dynamics.
        :rtype: list[ActionSet]
        :return: a list containing the agent's newly created actions.
        """
        assert agent.name not in self.agent_actions, \
            'An agent was already registered with the name \'{}\''.format(agent.name)
        self.agent_actions[agent.name] = []

        # creates agent's location feature
        x = self.world.defineState(
            agent.name, X_FEATURE + self.name, int, 0, self.width - 1, '{}\'s horizontal location'.format(agent.name))
        self.world.setFeature(x, 0)
        y = self.world.defineState(
            agent.name, Y_FEATURE + self.name, int, 0, self.height - 1, '{}\'s vertical location'.format(agent.name))
        self.world.setFeature(y, 0)

        # creates dynamics for the agent's movement (cardinal directions + no-op)
        action = agent.addAction({'verb': 'move', 'action': 'nowhere'})
        tree = makeTree(noChangeMatrix(x))
        self.world.setDynamics(x, action, tree)
        self.agent_actions[agent.name].append(action)

        # move right
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'right'})
        tree = makeTree({'if': thresholdRow(x, self.width - 1),  # if loc is right border
                         True: noChangeMatrix(x),  # stay still
                         False: incrementMatrix(x, 1)})  # else move right
        self.world.setDynamics(x, action, tree)
        self.agent_actions[agent.name].append(action)

        # move left
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'left'})
        tree = makeTree({'if': thresholdRow(x, 1),  # if loc is not left border
                         True: incrementMatrix(x, -1),  # move left
                         False: noChangeMatrix(x)})  # else stay still
        self.world.setDynamics(x, action, tree)
        self.agent_actions[agent.name].append(action)

        # move up
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'up'})
        tree = makeTree({'if': thresholdRow(y, self.height - 1),  # if loc is up border
                         True: noChangeMatrix(y),  # stay still
                         False: incrementMatrix(y, 1)})  # else move right
        self.world.setDynamics(y, action, tree)
        self.agent_actions[agent.name].append(action)

        # move down
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'down'})
        tree = makeTree({'if': thresholdRow(y, 1),  # if loc is not bottom border
                         True: incrementMatrix(y, -1),  # move down
                         False: noChangeMatrix(y)})  # else stay still
        self.world.setDynamics(y, action, tree)
        self.agent_actions[agent.name].append(action)

        return self.agent_actions[agent.name]

    def get_location_features(self, agent):
        """
        Gets the agent's (X,Y) features in the gridworld.
        :param Agent agent: the agent for which to get the location features.
        :rtype: (str,str)
        :return: a tuple containing the (X, Y) agent features.
        """
        x = stateKey(agent.name, X_FEATURE + self.name)
        y = stateKey(agent.name, Y_FEATURE + self.name)
        return x, y

    def get_location_plane(self, agent, locs, comp=0):
        """
        Gets a PsychSim plane for the given agent that can be used to compare it's current location against the given
        set of locations. Comparisons are made at the index level, i.e., in the left-right, bottom-up order.
        Also, comparison uses logical OR, i.e., it verifies against *any* of the given locations.
        :param Agent agent: the agent for which to get the comparison plane.
        :param list[(int,int)] locs: a list of target XY coordinate tuples.
        :param int comp: the comparison to be made (0:'==', 1:'>', 2:'<').
        :rtype: KeyedPlane
        :return: the plane corresponding to comparing the agent's location against the given coordinates.
        """
        x_feat = stateKey(agent.name, X_FEATURE + self.name)
        y_feat = stateKey(agent.name, Y_FEATURE + self.name)
        loc_values = {self.xy_to_idx(x, y) for x, y in locs}
        return KeyedPlane(KeyedVector({x_feat: 1., y_feat: self.width}), loc_values, comp)

    def idx_to_xy(self, i):
        """
        Converts the given location index to XY coordinates. Indexes are taken from the left-right, bottom-up order.
        :param int i: the index of the location.
        :rtype: (int, int)
        :return: a tuple containing the XY coordinates corresponding to the given location index.
        """
        return i % self.width, i // self.width

    def xy_to_idx(self, x, y):
        """
        Converts the given XY coordinates to a location index. Indexes are taken from the left-right, bottom-up order.
        :param int x: the location's X coordinate.
        :param int y: the location's Y coordinate.
        :rtype: int
        :return: an integer corresponding to the given coordinates' location index.
        """
        return x + y * self.width

    def generate_trajectories(self, n_trajectories, trajectory_length, agent, init_feats=None,
                              model=None, horizon=None, selection=None, processes=-1, seed=0):
        """
        Generates a number of fixed-length agent trajectories/traces/paths (state-action pairs).
        :param int n_trajectories: the number of trajectories to be generated.
        :param int trajectory_length: the length of the generated trajectories.
        :param Agent agent: the agent for which to record the actions.
        :param list[list] init_feats: a list of initial feature states from which to randomly initialize the
        trajectories. Each item is a list of feature values in the same order of `features`. If `None`, then features
        will be initialized uniformly at random according to their domain.
        :param str model: the agent model used to generate the trajectories.
        :param int horizon: the agent's planning horizon.
        :param str selection: the action selection criterion, to untie equal-valued actions.
        :param int processes: number of processes to use. `<=0` indicates all cores available, `1` uses single process.
        :param int seed: the seed used to initialize the random number generator.
        :rtype: list[list[tuple[World, ActionSet]]]
        :return: the generated agent trajectories.
        """
        # get relevant features for this world (x-y location)
        x, y = self.get_location_features(agent)

        # generate trajectories starting from random locations in the gridworld
        return generate_trajectories(agent, n_trajectories, trajectory_length,
                                     [x, y], init_feats, model, horizon, selection, processes, seed)

    def set_achieve_locations_reward(self, agent, locs, weight, model=None):
        """
        Sets a reward to the agent such that if its current location is equal to one of the given locations it will
        receive the given value. Comparisons are made at the index level, i.e., in the left-right, bottom-up order.
        :param Agent agent: the agent for which to get set the reward.
        :param list[(int,int)] locs: a list of target XY coordinate tuples.
        :param float weight: the weight/value associated with this reward.
        :param str model: the agent's model on which to set the reward.
        :return:
        """
        return agent.setReward(makeTree({'if': self.get_location_plane(agent, locs),
                                         True: setToConstantMatrix(rewardKey(agent.name), 1.),
                                         False: setToConstantMatrix(rewardKey(agent.name), 0.)}), weight, model)

    def get_all_states(self, agent):
        """
        Collects all PsychSim world states that the given agent can be in according to the gridworld's locations.
        Other PsychSim features *are not* changed, i.e., the agent does not perform any actions.
        :param Agent agent: the agent for which to get the states.
        :rtype: list[VectorDistributionSet]
        :return: a list of PsychSim states in the left-right, bottom-up order.
        """
        assert agent.world == self.world, 'Agent\'s world different from the environment\'s world!'

        old_state = copy.deepcopy(self.world.state)
        states = [None] * self.width * self.height

        # iterate through all agent positions and copy world state
        x, y = self.get_location_features(agent)
        for x_i, y_i in itertools.product(range(self.width), range(self.height)):
            self.world.setFeature(x, x_i)
            self.world.setFeature(y, y_i)
            idx = self.xy_to_idx(x_i, y_i)
            states[idx] = copy.deepcopy(self.world.state)

        # undo world state
        self.world.state = old_state
        return states

    def print_trajectories_cmd_line(self, trajectories):
        """
        Prints the given trajectories to the command-line.
        :param list[list[tuple[World, ActionSet]]] trajectories: the set of trajectories to save, containing
        several sequences of state-action pairs.
        :return:
        """
        if len(trajectories) == 0 or len(trajectories[0]) == 0:
            return

        name = trajectories[0][0][1]['subject']
        assert name in self.world.agents, 'Agent \'{}\' does not exist in the world!'.format(name)

        x, y = self.get_location_features(self.world.agents[name])
        assert x in self.world.variables, 'Agent \'{}\' does not have x location feature'.format(name)
        assert y in self.world.variables, 'Agent \'{}\' does not have y location feature'.format(name)

        for i, trajectory in enumerate(trajectories):
            print('-------------------------------------------')
            print('Trajectory {}:'.format(i))
            for t, sa in enumerate(trajectory):
                world, action = sa
                x_t = world.getValue(x)
                y_t = world.getValue(y)
                print('{}:\t({},{}) -> {}'.format(t, x_t, y_t, action))

    def plot(self, file_name, title='Environment', show=False):
        """
        Generates ands saves a grid plot of the environment, including the number of each state.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param bool show: whether to show the plot to the screen.
        :return:
        """
        plt.figure()
        self._plot(None, title, None)
        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved environment \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def plot_func(self, value_func, file_name, title='Environment', cmap=VALUE_CMAP, show=False):
        """
        Generates ands saves a plot of the environment, including a heatmap according to the given value function.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        :return:
        """
        plt.figure()

        self._plot(value_func, title, cmap)

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved environment \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def plot_policy(self, policy, value_func, file_name, title='Policy', cmap=VALUE_CMAP, show=False):
        """
        Generates ands saves a plot of the given policy in the environment.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param np.ndarray policy: the policy to be plotted of shape (n_states, n_actions).
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        :return:
        """
        plt.figure()

        # first plot environment
        self._plot(value_func, title, cmap)

        # then plot max actions for each state
        for x, y in itertools.product(range(self.width), range(self.height)):
            idx = self.xy_to_idx(x, y)

            # plot marker (arrow) for each action, control alpha according to probability
            for a in range(policy.shape[1]):
                x_inc, y_inc, marker = MARKERS_INC[a]
                plt.plot(x + x_inc, y + y_inc, marker=marker, c=POLICY_MARKER_COLOR, mew=0.5, mec='0',
                         alpha=policy[idx, a])

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved policy \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def plot_trajectories(self, trajectories, file_name, title='Trajectories',
                          value_func=None, cmap=VALUE_CMAP, show=False):
        """
        Plots the given set of trajectories over a representation of the environment.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param list[list[tuple[World, ActionSet]]] trajectories: the set of trajectories to save, containing
        several sequences of state-action pairs.
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        :return:
        """
        if len(trajectories) == 0 or len(trajectories[0]) == 0:
            return

        name = trajectories[0][0][1]['subject']
        assert name in self.world.agents, 'Agent \'{}\' does not exist in the world!'.format(name)

        x, y = self.get_location_features(self.world.agents[name])
        assert x in self.world.variables, 'Agent \'{}\' does not have x location feature'.format(name)
        assert y in self.world.variables, 'Agent \'{}\' does not have y location feature'.format(name)

        plt.figure()
        ax = plt.gca()

        # plot base environment
        self._plot(value_func, title, cmap)

        # plots trajectories
        t_colors = distinct_colors(len(trajectories))
        for i, trajectory in enumerate(trajectories):
            xs = []
            ys = []
            for t, sa in enumerate(trajectory):
                world, action = sa
                x_t = world.getValue(x)
                y_t = world.getValue(y)
                xs.append(x_t + .5)
                ys.append(y_t + .5)

                # plots label and final mark
                if t == len(trajectory) - 1:
                    plt.plot(x_t + .5, y_t + .5, 'x', c=t_colors[i], mew=1)
                    ax.annotate(
                        'T{:02d}'.format(i), xy=(x_t + .7, y_t + .7), fontsize=NOTES_FONT_SIZE, c=NOTES_FONT_COLOR)

            plt.plot(xs, ys, c=t_colors[i], linewidth=TRAJECTORY_LINE_WIDTH)

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved trajectories \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def _plot(self, val_func=None, title='Environment', cmap=None):
        ax = plt.gca()

        if val_func is None:
            # plots grid with cell numbers
            grid = np.zeros((self.width, self.height))
            plt.pcolor(grid, cmap=ListedColormap(['white']), edgecolors='darkgrey')
            for x, y in itertools.product(range(self.width), range(self.height)):
                ax.annotate('({},{})'.format(x, y), xy=(x + .05, y + .05), fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
        else:
            # plots given value function as heatmap
            val_func = val_func.reshape((self.width, self.height))
            plt.pcolor(val_func, cmap=cmap)
            plt.colorbar()

        # turn off tick labels
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect('equal', adjustable='box')
        plt.title(title, fontweight='bold', fontsize=TITLE_FONT_SIZE)
