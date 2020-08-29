import operator
import numpy as np
from abc import ABC, abstractmethod
from psychsim.agent import Agent
from psychsim.pwl import VectorDistributionSet, rewardKey, setToConstantMatrix, makeTree, setToFeatureMatrix, \
    KeyedTree, KeyedPlane, KeyedVector, actionKey

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

COMPARISON_OPS = {'==': operator.eq, '>': operator.gt, '<': operator.lt}


class LinearRewardVector(object):
    """
    Represents a linear reward vector, i.e., a reward function formed by a linear combination of a set of reward
    features.
    """

    def __init__(self, rwd_features):
        """
        Creates a new reward vector with the given features.
        :param list[LinearRewardFeature] rwd_features: the list of reward features.
        """
        self.rwd_features = rwd_features
        self.names = [feat.name for feat in self.rwd_features]

    def __len__(self):
        return len(self.rwd_features)

    def __iter__(self):
        return iter(self.rwd_features)

    def get_values(self, state):
        """
        Gets an array with the values / counts for each reward feature according to the given state. If the state is
        probabilistic (distribution), then average among all possible states is retrieved, weight by the corresponding
        probabilities.
        :param VectorDistributionSet state: the world state from which to get the feature value.
        :rtype: np.ndarray
        :return: an array containing the values for each feature.
        """
        return np.array([feat.get_value(state) for feat in self.rwd_features])

    def set_rewards(self, agent, weights, model=None):
        """
        Sets rewards to the agent corresponding to the value of each feature multiplied by the corresponding weight.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param np.ndarray weights: the weights associated with each reward feature.
        :param str model: the name of the agent's model for whom to set the reward.
        :return:
        """
        assert len(weights) == len(self.rwd_features), \
            'Provided weight vector\'s dimension does not match reward features'
        for i, weight in enumerate(weights):
            self.rwd_features[i].set_reward(agent, weight, model)


class LinearRewardFeature(ABC):
    """
    Represents a reward feature that can be used in linearly-parameterized reward functions.
    """

    def __init__(self, name):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        """
        self.name = name

    @abstractmethod
    def get_value(self, state):
        """
        Gets the value / count of the feature according to the given state. If the state is probabilistic
        (distribution), then average among all possible states is retrieved, weight by the corresponding probabilities.
        :param VectorDistributionSet state: the world state from which to get the feature value.
        :rtype: float
        :return: the (mean) value of the feature in the given state.
        """
        pass

    @abstractmethod
    def set_reward(self, agent, weight, model=None):
        """
        Sets a reward to the agent corresponding to the value of the features multiplied by the given weight.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param float weight: the weight associated with the reward feature.
        :param str model: the name of the agent's model for whom to set the reward.
        :return:
        """
        pass


class ValueComparisonLinearRewardFeature(LinearRewardFeature):
    """
    Represents a boolean reward feature that performs a comparison between a feature and a value, and returns `1` if
    the comparison is satisfied, otherwise returns `0`.
    """

    def __init__(self, name, world, key, value, comparison):
        """
        Creates a new reward feature.
        :param World world: the PsychSim world capable of retrieving the feature's value given a state.
        :param str name: the label for this reward feature.
        :param str key: the named key associated with this feature.
        :param str or int or float value: the value to be compared against the feature to determine its truth (boolean) value.
        :param str comparison: the comparison to be performed, one of `{'==', '>', '<'}`.
        """
        super().__init__(name)
        self.world = world
        self.key = key
        self.value = value
        assert comparison in KeyedPlane.COMPARISON_MAP, \
            'Invalid comparison provided: {}; valid: {}'.format(comparison, KeyedPlane.COMPARISON_MAP)
        self.comparison = KeyedPlane.COMPARISON_MAP.index(comparison)
        self.comp_func = COMPARISON_OPS[comparison]

    def get_value(self, state):
        # collects feature value distribution and returns weighted average
        dist = np.array([[float(self.comp_func(self.world.float2value(self.key, kv[self.key]), self.value)), p]
                         for kv, p in state.distributions[state.keyMap[self.key]].items()])
        return dist[:, 0].dot(dist[:, 1])

    def set_reward(self, agent, weight, model=None):
        rwd_key = rewardKey(agent.name)
        tree = {'if': KeyedPlane(KeyedVector({self.key: 1}), self.value, self.comparison),
                True: setToConstantMatrix(rwd_key, 1.),
                False: setToConstantMatrix(rwd_key, 0.)}
        agent.setReward(makeTree(tree), weight, model)


class ActionLinearRewardFeature(ValueComparisonLinearRewardFeature):
    """
    Represents a reward feature that returns `1` if the agent's action is equal to a given action, and `0` otherwise.
    """

    def __init__(self, name, agent, action):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param Agent agent: the agent whose action is going to be tracked by this feature.
        :param str action: the action tracked by this feature.
        """
        super().__init__(name, agent.world, actionKey(agent.name), action, '==')


class NumericLinearRewardFeature(LinearRewardFeature):
    """
    Represents a numeric reward feature, that returns a reward proportional to the feature's value.
    """

    def __init__(self, name, key):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param str key: the named key associated with this feature.
        """
        super().__init__(name)
        self.key = key

    def get_value(self, state):
        # collects feature value distribution and returns weighted average
        dist = np.array([[kv[self.key], p] for kv, p in state.distributions[state.keyMap[self.key]].items()])
        return dist[:, 0].dot(dist[:, 1])

    def set_reward(self, agent, weight, model=None):
        # simply multiply the feature's value by the given weight
        rwd_key = rewardKey(agent.name)
        agent.setReward(KeyedTree(setToFeatureMatrix(rwd_key, self.key)), weight, model)
