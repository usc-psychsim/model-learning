import copy
import logging
import numpy as np
import multiprocessing as mp
from typing import Callable
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.helper_functions import get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet
from model_learning.algorithms import ModelLearningAlgorithm

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# stats names
REWARD_WEIGHTS_STR = 'Weights'
FEATURE_COUNT_DIFF_STR = 'Feature Count Diff.'


def _gen_trajectory(args):
    # unpacks info
    agent, init_state, length = args
    world = agent.world
    world.state = init_state

    # for each step, uses the MaxEnt stochastic policy and registers state-action pairs
    exp_trajectory = []
    for i in range(length):
        decision = agent.decide()
        action = {agent.name: decision['policy']}
        exp_trajectory.append((copy.deepcopy(world.state), action))
        world.step(action)
    return exp_trajectory


class MaxEntRewardLearning(ModelLearningAlgorithm):
    """
    An implementation of the maximal causal entropy (MaxEnt) algorithm for IRL in [1].
    It assumes the expert's reward function is a linear combination (weighted sum) of the state features.
    Optimizes the linear parametrization of the rewards (weights) as follows:
    1. Initialize weights at random
    2. Perform gradient descent iteratively:
        i. Computes MaxEnt stochastic policy given current reward function (backward pass)
        ii. Compute expected state visitation frequencies from policy and trajectories
        iii. Compute loss as difference between empirical (from trajectories) and expected feature counts
        iv. Update weights given loss
    3. Learner's reward is given by the best fit between expert's (via trajectories) and learner's expected svf.
    [1] - Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement
    learning. In AAAI (Vol. 8, pp. 1433-1438).
    """

    def __init__(self, label, base_agent, num_features,
                 feature_func: Callable[[VectorDistributionSet], np.ndarray],
                 reward_func: Callable[[np.ndarray, Agent], None],
                 processes=-1, normalize_weights=True, learning_rate=0.01, max_epochs=200, threshold=1e-2, seed=0):
        """
        Creates a new Max Entropy algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param Agent base_agent: the base agent containing the model parameters that will not be optimized by the
        algorithm. Contains reference to the PsychSim world containing all definitions.
        :param int num_features: the number of features whose associated reward weights are going to be optimized.
        :param feature_func: a function (state) -> feature_values that takes in a PsychSim state and returns the
        feature mapping for that state. The resulting array should be of shape (`num_features`,).
        :param reward_func: a function (array, agent) -> None, that takes in an array of feature weights,
        a PsychSim agent and a model name, and sets its reward function according to the linear reward parametrization.
        :param int processes: number of processes to use. `<=0` indicates all cores available, `1` uses single process.
        :param bool normalize_weights: whether to normalize reward weights at each step of the algorithm.
        :param float learning_rate: the gradient descent learning/update rate.
        :param int max_epochs: the maximum number of gradient descent steps.
        :param float threshold: the termination threshold for the weight vector difference.
        :param int seed: the seed to initialize the random number generator.
        """
        super().__init__(label, base_agent)

        self.num_features = num_features
        self.feature_func = feature_func
        self.set_reward_func = reward_func
        self.processes = processes
        self.normalize_weights = normalize_weights
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.seed = seed

    def _get_mean_feature_counts(self, trajectories):
        """
        Gets the mean path feature counts, i.e., the sum of the feature values for each state along a trajectory,
        averaged across all given trajectories. See [1].
        [1] - Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement
        learning. In AAAI (Vol. 8, pp. 1433-1438).
        :param list[list[VectorDistributionSet, object]] trajectories: a list of trajectories, each containing a list
        (sequence) of state-action pairs.
        :rtype: np.ndarray
        :return: the mean counts for each feature.
        """
        if len(trajectories) == 0:
            return np.zeros(self.num_features)

        # gets all states in the trajectories and get feature counts in parallel
        states = [s for trajectory in trajectories for s, _ in trajectory]
        with mp.Pool(self.processes if self.processes > 0 else None) as pool:
            empirical_fc = np.sum(pool.map(self.feature_func, states), axis=0)
            return (empirical_fc / len(trajectories)) if len(trajectories) > 0 else empirical_fc

    def _generate_max_ent_trajectories(self, trajectories):
        """
        Generates trajectories with the same length as the given expert trajectories but using the max. entropy
        stochastic policy (according to the current reward function) associated with the agent's model.
        :param list[list[VectorDistributionSet, ActionSet]] trajectories:a list of trajectories, each containing a list
        (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: list[list[VectorDistributionSet, Distribution]]
        :return: stochastic trajectories generated using the max. entropy principle, i.e., a distribution over paths.
        """
        # generates each trajectory in parallel
        with mp.Pool(self.processes if self.processes > 0 else None) as pool:
            return pool.map(_gen_trajectory, [(self.agent, t[0][0], len(t)) for t in trajectories])

    def learn(self, trajectories):
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[list[VectorDistributionSet, ActionSet]] trajectories:a list of trajectories, each containing a list
        (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: (dict, dict[str, np.ndarray])
        :return: a tuple (model, stats) containing the learned PsychSim model (a dictionary of parameters), and a
        dictionary with relevant statistics of the algorithm.
        """

        # keep copy of world state
        old_state = copy.deepcopy(self.world.state)

        # keep copy of previous models
        true_model_name = get_true_model_name(self.agent)
        old_true_model = None
        old_models = []
        for name in list(self.agent.models.keys()):
            if name == true_model_name:
                old_true_model = self.agent.models[name].copy()
            else:
                old_models = self.agent.models[name]
                self.agent.deleteModel(name)

        # 0 - parameterizes according to the max. entropy principle
        self.agent.setAttribute('rationality', 1.)
        self.agent.setAttribute('selection', 'distribution')

        # get empirical feature counts (mean feature path) from trajectories
        empirical_fc = self._get_mean_feature_counts(trajectories)

        # 1 - init reward weights at random
        rng = np.random.RandomState(self.seed)
        theta = rng.uniform(-1, 1, self.num_features)
        del self.agent.models[true_model_name]['R']  # we can't know this!
        self.set_reward_func(theta, self.agent)

        # 2 - perform gradient descent to optimize reward weights
        diff = np.float('inf')
        e = 0
        diffs = [1.] if self.normalize_weights else []
        thetas = [theta]
        while diff > self.threshold and e < self.max_epochs:
            with np.printoptions(precision=2, suppress=True):
                logging.info('Step {}: diff={:.3f}, theta={}'.format(e, diff, theta))

            # compute expected svf using a MaxEnt stochastic policy from reward
            expected_trajectories = self._generate_max_ent_trajectories(trajectories)

            # gets expected feature counts (mean feature path) from trajectories
            expected_fc = self._get_mean_feature_counts(expected_trajectories)

            # gradient descent step, update reward weights
            grad = empirical_fc - expected_fc
            new_theta = theta + self.learning_rate * grad
            if self.normalize_weights:
                new_theta /= np.linalg.norm(new_theta, 1)  # min_max_scale(new_theta)

            diff = np.linalg.norm(new_theta - theta)
            diffs.append(diff)
            theta = new_theta
            thetas.append(theta)
            e += 1

            # set updated reward function
            del self.agent.models[true_model_name]['R']
            self.set_reward_func(theta, self.agent)

        # puts back world state
        self.world.state = old_state

        # puts back old models
        learned_model = self.agent.models[true_model_name]
        self.agent.models[true_model_name] = old_true_model
        for model in old_models:
            self.agent.addModel(model['name'], **model)

        # returns model and stats dictionary
        return learned_model, {FEATURE_COUNT_DIFF_STR: np.array([diffs]),
                               REWARD_WEIGHTS_STR: np.array(thetas).T}
