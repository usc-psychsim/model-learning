import logging
import os
import random
import numpy as np
from timeit import default_timer as timer
from psychsim.agent import Agent
from psychsim.pwl import VectorDistributionSet, turnKey
from psychsim.probability import Distribution
from model_learning.util.multiprocessing import get_pool_and_map
from model_learning.util.plot import plot_evolution
from model_learning.trajectory import TOP_LEVEL_STR
from model_learning.algorithms import ModelLearningAlgorithm
from model_learning.features.linear import LinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# stats names
REWARD_WEIGHTS_STR = 'Weights'
FEATURE_COUNT_DIFF_STR = 'Feature Count Diff.'
THETA_STR = 'Optimal Weight Vector'
TIME_STR = 'Time'
LEARN_RATE_STR = 'Learning Rate'


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

    def __init__(self, label, base_agent, reward_vector,
                 processes=None, normalize_weights=True,
                 learning_rate=0.01, max_epochs=200, diff_threshold=1e-2, decrease_rate=False,
                 prune_threshold=1e-2, horizon=2, seed=0):
        """
        Creates a new Max Entropy algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param Agent base_agent: the base agent containing the model parameters that will not be optimized by the
        algorithm. Contains reference to the PsychSim world containing all definitions.
        :param LinearRewardVector reward_vector: the reward vector containing the features whose weights are going to
        be optimized.
        :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
        :param bool normalize_weights: whether to normalize reward weights at each step of the algorithm.
        :param float learning_rate: the gradient descent learning/update rate.
        :param int max_epochs: the maximum number of gradient descent steps.
        :param float diff_threshold: the termination threshold for the weight vector difference.
        :param bool decrease_rate: whether to exponentially decrease the learning rate over time.
        :param float prune_threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int horizon: the planning horizon used to compute feature counts.
        :param int seed: the seed to initialize the random number generator.
        """
        super().__init__(label, base_agent)

        self.reward_vector = reward_vector
        self.num_features = len(reward_vector)
        self.processes = processes
        self.normalize_weights = normalize_weights
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.diff_threshold = diff_threshold
        self.decrease_rate = decrease_rate
        self.prune_threshold = prune_threshold
        self.horizon = horizon
        self.seed = seed

    def _get_empirical_feature_counts(self, trajectories):
        """
        Gets the mean path feature counts, i.e., the sum of the feature values for each state along a trajectory,
        averaged across all given trajectories. See [1].
        :param list[list[tuple[VectorDistributionSet, Distribution]]] trajectories: a list of trajectories, each
        containing a list (sequence) of state-action pairs.
        :rtype: np.ndarray
        :return: the mean counts for each feature.
        """
        if len(trajectories) == 0:
            return np.zeros(self.num_features)

        pool, map_func = get_pool_and_map(self.processes, False)

        # gets all states in the trajectories and get feature counts
        states = [s for trajectory in trajectories for s, _ in trajectory]
        all_fcs = list(map_func(self.reward_vector.get_values, states))
        if pool is not None:
            pool.close()
        empirical_fc = np.sum(all_fcs, axis=0)
        return (empirical_fc / len(trajectories)) if len(trajectories) > 0 else empirical_fc

    def _get_trajectory_feature_counts(self, init_state, length, seed):
        """
        Gets the feature counts, i.e., the sum of the feature values, for each state along a stochastic trajectory,
        i.e., a distribution over paths generated according to the MaxEnt principle, using the agent's current reward
        function.
        :param VectorDistributionSet init_state: the world initial state from where to start collecting the trajectory.
        :param int length: the length of the trajectory/path to be generated.
        :param int seed: the seed used to initialize the random number generator.
        :rtype: np.ndarray
        :return: the counts (sum of values) for each feature in the reward vector.
        """
        self.world.state = init_state
        random.seed(seed)

        # for each step, uses the MaxEnt stochastic policy and registers state-action pairs
        all_fcs = []
        for i in range(length):
            # step the world until it's this agent's turn
            turn = self.world.getFeature(turnKey(self.agent.name), unique=True)
            while turn != 0:
                self.world.step()
                turn = self.world.getFeature(turnKey(self.agent.name), unique=True)

            # gets feature counts for this state
            all_fcs.append(self.reward_vector.get_values(self.world.state))

            # steps the world if needed
            if i < length - 1:
                # todo
                # self.world.step(threshold=self.prune_threshold, debug={TOP_LEVEL_STR: True})
                self.world.step(select=True, debug={TOP_LEVEL_STR: True})

        return np.sum(all_fcs, axis=0)

    def _get_expected_feature_counts(self, trajectories):
        """
        Gets the mean path feature counts, i.e., the sum of the feature values for each state along a trajectory,
        averaged across a set of stochastic trajectories generated according to the MaxEnt principle. See [1].
        :param list[list[tuple[VectorDistributionSet, Distribution]]] trajectories:a list of trajectories, each
        containing a list (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: list[list[tuple[VectorDistributionSet, Distribution]]]
        :return: stochastic trajectories generated using the max. entropy principle, i.e., a distribution over paths.
        """
        pool, map_func = get_pool_and_map(self.processes, True)

        # generates trajectories
        all_fcs = list(map_func(self._get_trajectory_feature_counts,
                                [(t[0][0], len(t), self.seed + i) for i, t in enumerate(trajectories)]))
        if pool is not None:
            pool.close()
        estimated_fc = np.sum(all_fcs, axis=0)
        return (estimated_fc / len(trajectories)) if len(trajectories) > 0 else estimated_fc

    @staticmethod
    def log_progress(e, theta, diff, learning_rate, step_time):
        with np.printoptions(precision=2, suppress=True):
            logging.info('Step {}: diff={:.3f}, θ={}, α={:.2f}, time={:.2f}s'.format(
                e, diff, theta, learning_rate, step_time))

    def learn(self, trajectories, verbose=False):
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[list[tuple[VectorDistributionSet, Distribution]]] trajectories: a list of trajectories, each
        containing a list (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :param bool verbose: whether to show information at each timestep during learning.
        :rtype: dict[str, np.ndarray]
        :return: a dictionary with relevant statistics of the algorithm.
        """
        self._reset()

        # 0 - parameterizes according to the max. entropy principle
        self.agent.setAttribute('rationality', 1.)
        self.agent.setAttribute('selection', 'distribution')
        self.agent.setAttribute('horizon', self.horizon)

        # get empirical feature counts (mean feature path) from trajectories
        empirical_fc = self._get_empirical_feature_counts(trajectories)

        # 1 - initiates reward weights at random
        # rng = np.random.RandomState(self.seed)
        # theta = rng.uniform(-1, 1, self.num_features)
        theta = np.ones(self.num_features) / self.num_features
        self.reward_vector.set_rewards(self.agent, theta)

        # 2 - perform gradient descent to optimize reward weights
        diff = np.float('inf')
        e = 0
        step_time = 0
        learning_rate = self.learning_rate
        diffs = [1.] if self.normalize_weights else []
        thetas = [theta]
        times = []
        rates = []
        while diff > self.diff_threshold and e < self.max_epochs:
            if verbose:
                self.log_progress(e, theta, diff, learning_rate, step_time)

            start = timer()

            # update learning rate
            learning_rate = self.learning_rate
            if self.decrease_rate:
                learning_rate *= np.power(1 - (10 / self.max_epochs), e)

            # gets expected feature counts (mean feature path)
            # by computing the svf using a MaxEnt stochastic policy from reward
            expected_fc = self._get_expected_feature_counts(trajectories)

            # gradient descent step, update reward weights
            grad = empirical_fc - expected_fc
            new_theta = theta + learning_rate * grad
            if self.normalize_weights:
                new_theta /= np.linalg.norm(new_theta, 1)

            step_time = timer() - start

            # registers stats
            diff = np.linalg.norm(new_theta - theta)
            diffs.append(diff)
            theta = new_theta
            thetas.append(theta)
            times.append(step_time)
            rates.append(learning_rate)
            e += 1

            # set updated reward function
            self.reward_vector.set_rewards(self.agent, theta)

        if verbose:
            self.log_progress(e, theta, diff, learning_rate, step_time)
            logging.info('Finished, total time: {:.2f} secs.'.format(sum(times)))

        # returns stats dictionary
        return {
            FEATURE_COUNT_DIFF_STR: np.array([diffs]),
            REWARD_WEIGHTS_STR: np.array(thetas).T,
            THETA_STR: theta,
            TIME_STR: np.array([times]),
            LEARN_RATE_STR: np.array([rates])
        }

    def save_results(self, stats, output_dir, img_format):
        np.savetxt(os.path.join(output_dir, 'learner-theta.csv'), stats[THETA_STR].reshape(1, -1), '%s', ',',
                   header=','.join(self.reward_vector.names), comments='')

        plot_evolution(stats[FEATURE_COUNT_DIFF_STR], ['diff'], 'Feature Count Diff. Evolution', None,
                       os.path.join(output_dir, 'evo-feat-diff.{}'.format(img_format)), 'Epoch', 'Feature Difference')

        plot_evolution(stats[REWARD_WEIGHTS_STR], self.reward_vector.names, 'Reward Parameters Evolution', None,
                       os.path.join(output_dir, 'evo-rwd-weights.{}'.format(img_format)), 'Epoch', 'Weight')

        plot_evolution(stats[TIME_STR], ['time'], 'Step Time Evolution', None,
                       os.path.join(output_dir, 'evo-time.{}'.format(img_format)), 'Epoch', 'Time (secs.)')

        plot_evolution(stats[LEARN_RATE_STR], ['learning rate'], 'Learning Rate Evolution', None,
                       os.path.join(output_dir, 'learning-rate.{}'.format(img_format)), 'Epoch', 'α')
