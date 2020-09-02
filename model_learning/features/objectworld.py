import copy
import numpy as np
from model_learning.features.linear import LinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class ObjectsRewardVector(LinearRewardVector):
    """
    Represents a linear reward vector, i.e., a reward function formed by a linear combination of a set of reward
    features.
    """

    def __init__(self, env, agent, feat_matrix, outer=True, inner=True):
        """
        Creates a new reward vector with the given features.
        :param ObjectsGridWorld env: the objects environment used iby this reward function.
        :param Agent agent: the agent for which to get the location features.
        :param np.ndarray feat_matrix: the feature matrix containing the color features for each environment location.
        :param bool outer: whether to include features for the presence of objects' outer colors.
        :param bool inner: whether to include features for the presence of objects' inner colors.
        """
        super().__init__([])
        assert env.num_colors * outer + env.num_colors * inner == feat_matrix.shape[1], \
            'Invalid feature matrix dimensions for provided inner and outer color arguments.'
        self.env = env
        self.x, self.y = env.get_location_features(agent)
        self.feat_matrix = feat_matrix
        self.outer = outer
        self.inner = inner
        self.names = ['Out Color {}'.format(i + 1) for i in range(env.num_colors if outer else 0)] + \
                     ['In Color {}'.format(i + 1) for i in range(env.num_colors if inner else 0)]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def get_values(self, state):
        """
        Gets an array with the values / counts for each reward feature according to the given state. If the state is
        probabilistic (distribution), then average among all possible states is retrieved, weight by the corresponding
        probabilities.
        :param VectorDistributionSet state: the world state from which to get the feature value.
        :rtype: np.ndarray
        :return: an array containing the values for each feature.
        """
        # get agent's XY features
        state = copy.deepcopy(state)
        state.collapse({self.x, self.y}, False)

        # collects possible locations and associated probabilities
        locs = {}
        for row in state.distributions[state.keyMap[self.x]].domain():
            x_val = row[self.x]
            y_val = row[self.y]
            locs[self.env.xy_to_idx(x_val, y_val)] = state.distributions[state.keyMap[self.x]][row]

        # return weighted average of feature vectors
        return np.multiply(self.feat_matrix[list(locs.keys())],
                           np.array(list(locs.values())).reshape(len(locs), 1)).sum(axis=0)

    def set_rewards(self, agent, weights, model=None):
        """
        Sets a reward to the agent that is a linear combination of the given weights associated with each object color.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param np.ndarray weights: the reward weights associated with each object's outer and inner color.
        :param str model: the name of the agent's model for whom to set the reward.
        :return:
        """
        assert len(weights) == len(self.names), 'Provided weight vector\'s dimension does not match reward features'
        agent.setAttribute('R', {})
        self.env.set_linear_color_reward(agent,
                                         weights[:self.env.num_colors] if self.outer else None,
                                         weights[self.env.num_colors:] if self.inner else None)
