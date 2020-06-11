import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from psychsim.world import World
from psychsim.agent import Agent
from model_learning.environments.gridworld import GridWorld
from model_learning.util.plot import distinct_colors

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class ObjectsGridWorld(GridWorld):
    """
    Represents a gridworld environment containing a set of objects. Each object has a inner and an outer color, which,
    e.g., can be used to manipulate rewards provided to an agent.
    """

    def __init__(self, world, width, height, num_objects, num_colors,
                 name='', seed=0, show_objects=True, single_color=True):
        """
        Creates a new gridworld.
        :param World world: the PsychSim world associated with this gridworld.
        :param int width: the number of horizontal cells.
        :param int height: the number of vertical cells.
        :param str name: the name of this gridworld, used as a suffix for features, actions, etc.
        :param int num_objects: the number of objects to place in the environment.
        :param int num_colors: the number of (inner and outer) colors available for the objects.
        :param int seed: the seed used to initialize the random number generator to create and place the objects.
        :param bool show_objects: whether to show objects when plotting the environment. Can be changed at any time.
        :param bool single_color: whether objects should have a single color (i.e., inner and outer are the same).
        """
        super().__init__(world, width, height, name)
        self.num_objects = num_objects
        self.num_colors = num_colors
        self.show_objects = show_objects

        # initialize objects
        rng = np.random.RandomState(seed)
        locations = rng.choice(np.arange(width * height), num_objects, False)
        self.objects = {}
        for loc in locations:
            x, y = self.idx_to_xy(loc)
            color = rng.randint(num_colors)
            self.objects[(x, y)] = (color, color) if single_color else (color, rng.randint(num_colors))

    def _plot(self, val_func=None, title='Environment', cmap=None):

        super()._plot(val_func, title, cmap)

        if self.show_objects:
            ax = plt.gca()

            # adds colors legend
            obj_colors = distinct_colors(self.num_colors)
            patches = []
            for i, color in enumerate(obj_colors):
                patches.append(mpatches.Patch(color=color, label='Color {}'.format(i + 1)))
            leg = plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(-.35, 0.02, 1, 1), fancybox=False)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(0.8)

            # plots objects as colored circles
            for loc, color in self.objects.items():
                outer_circle = plt.Circle((loc[0] + .5, loc[1] + .5), 0.3, color=obj_colors[color[0]])
                inner_circle = plt.Circle((loc[0] + .5, loc[1] + .5), 0.1, color=obj_colors[color[1]])
                ax.add_artist(outer_circle)
                ax.add_artist(inner_circle)

    def get_loc_feature_matrix(self, outer=True, inner=True):
        """
        Gets a matrix containing boolean features for the presence of object colors in each indexed location in the
        environment, where a feature is `1` if there is an object of the corresponding color in that location, `0`
        otherwise. Location index is given in a left-right, bottom-up order.
        :param bool outer: whether to include features for the presence of objects' outer colors.
        :param bool inner: whether to include features for the presence of objects' inner colors.
        :rtype: np.ndarray
        :return: an array of shape (width*height, num_colors*(outer+inner)) containing the color features for each
        environment location.
        """
        feat_matrix = np.zeros((self.width * self.height, self.num_colors * (outer + inner)))
        for loc, color in self.objects.items():
            idx = self.xy_to_idx(*loc)
            if outer:
                feat_matrix[idx][color[0]] = 1.
            if inner:
                feat_matrix[idx][color[1]] = 1.
        return feat_matrix

    def set_linear_color_reward(self, agent, weights_outer, weights_inner=None):
        """
        Sets a reward to the agent that is a linear combination of the given weights associated with each object color.
        In other words, when the agent is collocated with some object, the received reward will be proportional to the
        value associated with that object.
        :param Agent agent: the agent for which to get set the reward.
        :param np.ndarray or list[float] weights_outer: the reward weights associated with each object's outer color.
        :param np.ndarray or list[float] weights_inner: the reward weights associated with each object's inner color.
        :return:
        """

        # checks
        assert weights_outer is not None and len(weights_outer) == self.num_colors, \
            'Weight vectors should have length {}.'.format(self.num_colors)
        assert weights_inner is None or len(weights_inner) == self.num_colors, \
            'Weight vectors should have length {}.'.format(self.num_colors)

        # gets locations indexed by color
        inner_locs = [[] for _ in range(self.num_colors)]
        outer_locs = [[] for _ in range(self.num_colors)]
        for loc, color in self.objects.items():
            inner_locs[color[0]].append(loc)
            outer_locs[color[1]].append(loc)

        # sets the corresponding weight for each object location
        for color in range(self.num_colors):
            self.set_achieve_locations_reward(agent, outer_locs[color], weights_outer[color])
            if weights_inner is not None:
                self.set_achieve_locations_reward(agent, inner_locs[color], weights_inner[color])
