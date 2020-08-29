import copy
import logging
import random
import numpy as np
from typing import Callable
from timeit import default_timer as timer
from psychsim.agent import Agent
from psychsim.pwl import VectorDistributionSet, turnKey
from psychsim.probability import Distribution
from model_learning.util import get_pool_and_map
from model_learning.trajectory import TOP_LEVEL_STR, get_agent_action
from model_learning.algorithms import ModelLearningAlgorithm

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# stats names
REWARD_WEIGHTS_STR = 'Weights'
FEATURE_COUNT_DIFF_STR = 'Feature Count Diff.'
THETA_STR = 'Optimal Weight Vector'
TIME_STR = 'Time'


def gen_stochastic_trajectory(agent, init_state, length, threshold, seed):
    """
    Generates a stochastic trajectory, i.e., a distribution over paths, using the agent's current reward function.
    :param Agent agent: the agent from whom to collect the trajectory.
    :param VectorDistributionSet init_state: the world initial state from where to start collecting the trajectory.
    :param int length: the length of the trajectory to be generated.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[tuple[VectorDistributionSet, Distribution]]
    :return: the generated stochastic trajectory.
    """
    world = agent.world
    world.state = init_state

    random.seed(seed)
    # todo need this?
    agent.belief_threshold = 1e-2

    # for each step, uses the MaxEnt stochastic policy and registers state-action pairs
    trajectory = []
    for i in range(length):

        # step the world until it's this agent's turn
        turn = world.getFeature(turnKey(agent.name), unique=True)
        while turn != 0:
            world.step()
            turn = world.getFeature(turnKey(agent.name), unique=True)

        # steps the world, gets the agent's action
        prev_state = copy.deepcopy(world.state)
        # todo
        # world.step(threshold=threshold, debug={TOP_LEVEL_STR: True})
        world.step(select=True, threshold=threshold, debug={TOP_LEVEL_STR: True})
        trajectory.append((prev_state, get_agent_action(agent, world.state)))

    return trajectory


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
                 processes=None, normalize_weights=True, learning_rate=0.01, max_epochs=200, diff_threshold=1e-2,
                 prune_threshold=1e-2, seed=0):
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
        :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
        :param bool normalize_weights: whether to normalize reward weights at each step of the algorithm.
        :param float learning_rate: the gradient descent learning/update rate.
        :param int max_epochs: the maximum number of gradient descent steps.
        :param float diff_threshold: the termination threshold for the weight vector difference.
        :param float prune_threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int seed: the seed to initialize the random number generator.
        """
        super().__init__(label, base_agent)

        self.num_features = num_features
        self.feature_func = feature_func
        self.set_reward_func = reward_func
        self.processes = processes
        self.normalize_weights = normalize_weights
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.diff_threshold = diff_threshold
        self.prune_threshold = prune_threshold
        self.seed = seed

    def _get_mean_feature_counts(self, trajectories):
        """
        Gets the mean path feature counts, i.e., the sum of the feature values for each state along a trajectory,
        averaged across all given trajectories. See [1].
        [1] - Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement
        learning. In AAAI (Vol. 8, pp. 1433-1438).
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
        all_fcs = list(map_func(self.feature_func, states))
        empirical_fc = np.sum(all_fcs, axis=0)
        if pool is not None:
            pool.close()
        return (empirical_fc / len(trajectories)) if len(trajectories) > 0 else empirical_fc

    def _generate_max_ent_trajectories(self, trajectories):
        """
        Generates trajectories with the same length as the given expert trajectories but using the max. entropy
        stochastic policy (according to the current reward function) associated with the agent's model.
        :param list[list[tuple[VectorDistributionSet, Distribution]]] trajectories:a list of trajectories, each
        containing a list (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :rtype: list[list[tuple[VectorDistributionSet, Distribution]]]
        :return: stochastic trajectories generated using the max. entropy principle, i.e., a distribution over paths.
        """
        pool, map_func = get_pool_and_map(self.processes, True)

        # generates trajectories
        trajectories = list(map_func(gen_stochastic_trajectory, [
            (self.agent, t[0][0], len(t), self.prune_threshold, self.seed + i) for i, t in enumerate(trajectories)]))
        if pool is not None:
            pool.close()
        return trajectories

    @staticmethod
    def log_progress(e, theta, diff, step_time):
        with np.printoptions(precision=2, suppress=True):
            logging.info('Step {}: diff={:.3f}, theta={}, time={:.2f}s'.format(e, diff, theta, step_time))

    def learn(self, trajectories, verbose=False):
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[list[tuple[VectorDistributionSet, Distribution]]] trajectories: a list of trajectories, each
        containing a list (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :param bool verbose: whether to show information at each timestep during learning.
        :rtype: dict[str, np.ndarray]
        :return: dictionary with relevant statistics of the algorithm.
        """

        # 0 - parameterizes according to the max. entropy principle
        self.agent.setAttribute('rationality', 1.)
        # todo change this
        self.agent.setAttribute('selection', 'distribution')
        # self.agent.setAttribute('selection', 'random')

        # get empirical feature counts (mean feature path) from trajectories
        empirical_fc = self._get_mean_feature_counts(trajectories)

        # 1 - initiates reward weights at random
        rng = np.random.RandomState(self.seed)
        theta = rng.uniform(-1, 1, self.num_features)
        self.set_reward_func(theta, self.agent)

        # 2 - perform gradient descent to optimize reward weights
        diff = np.float('inf')
        e = 0
        step_time = 0
        diffs = [1.] if self.normalize_weights else []
        thetas = [theta]
        times = []
        while diff > self.diff_threshold and e < self.max_epochs:
            if verbose:
                self.log_progress(e, theta, diff, step_time)

            start = timer()

            # compute expected svf using a MaxEnt stochastic policy from reward
            expected_trajectories = self._generate_max_ent_trajectories(trajectories)

            # gets expected feature counts (mean feature path) from trajectories
            expected_fc = self._get_mean_feature_counts(expected_trajectories)

            # gradient descent step, update reward weights
            grad = empirical_fc - expected_fc
            new_theta = theta + self.learning_rate * grad
            if self.normalize_weights:
                new_theta /= np.linalg.norm(new_theta, 1)  # min_max_scale(new_theta)

            step_time = timer() - start

            # registers stats
            diff = np.linalg.norm(new_theta - theta)
            diffs.append(diff)
            theta = new_theta
            thetas.append(theta)
            times.append(step_time)
            e += 1

            # set updated reward function
            self.set_reward_func(theta, self.agent)

        if verbose:
            self.log_progress(e, theta, diff, step_time)

        # returns stats dictionary
        return {
            FEATURE_COUNT_DIFF_STR: np.array([diffs]),
            REWARD_WEIGHTS_STR: np.array(thetas).T,
            THETA_STR: theta,
            TIME_STR: np.array([times])
        }
