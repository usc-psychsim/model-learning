from timeit import default_timer as timer

import logging
import numpy as np
from typing import Optional, List

from model_learning import TeamModelDistTrajectory
from model_learning.algorithms import ModelLearningResult
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, LEARNING_DECAY, FEATURE_COUNT_DIFF_STR, \
    REWARD_WEIGHTS_STR, THETA_STR, TIME_STR, LEARN_RATE_STR
from model_learning.features.counting import empirical_feature_counts, estimate_feature_counts_with_inference
from model_learning.features.linear import LinearRewardVector
from psychsim.agent import Agent

__author__ = 'Pedro Sequeira, Haochen Wu'
__email__ = 'pedrodbs@gmail.com, hcaawu@gmail.com'


class MIRLToM(MaxEntRewardLearning):
    """
    An extension of the maximal causal entropy (MaxEnt) IRL algorithm in [1] to multiagent scenarios.
    It implements the Multiagent IRL with Theory-of-Mind (MIRL-ToM) algorithm defined in [2], corresponding to a
    decentralized MIRL algorithm by applying the concept of ToM where reward model inference is performed over the
    demonstrated trajectories to estimate the evolving distribution over models of the other agents in a team.
    The rest of the process follows the MaxEnt IRL algorithm defined in the base class.
    [1] - Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement
    learning. In AAAI (Vol. 8, pp. 1433-1438).
    [2] - Wu, H., Sequeira, P., Pynadath, D. (2023). Multiagent Inverse Reinforcement Learning via Theory of Mind
    Reasoning. In AAMAS.
    """

    def __init__(self,
                 label: str,
                 agent: Agent,
                 reward_vector: LinearRewardVector,
                 normalize_weights: bool = True,
                 learning_rate: float = 0.01,
                 decrease_rate: bool = False,
                 max_epochs: int = 200,
                 diff_threshold: float = 1e-2,
                 exact: bool = False, num_mc_trajectories=1000,
                 prune_threshold: float = 1e-2,
                 horizon: int = 2,
                 processes: Optional[int] = -1,
                 seed: int = 17):
        """
        Creates a new Max Entropy algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param Agent agent: the agent whose behavior we want to model (the "expert").
        :param LinearRewardVector reward_vector: the reward vector containing the features whose weights are going to
        be optimized.
        :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
        :param bool normalize_weights: whether to normalize reward weights at each step of the algorithm.
        :param float learning_rate: the gradient descent learning/update rate.
        :param int max_epochs: the maximum number of gradient descent steps.
        :param float diff_threshold: the termination threshold for the weight vector difference.
        :param bool decrease_rate: whether to exponentially decrease the learning rate over time.
        :param bool exact: whether the computation of the distribution over paths should be exact (expand all stochastic
        branches) or not. If `False`, Monte Carlo sample trajectories will be generated to estimate the feature counts.
        :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
        :param float prune_threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int horizon: the planning horizon used to compute feature counts.
        :param int seed: the seed to initialize the random number generator.
        """

        super().__init__(label, agent, reward_vector, normalize_weights, learning_rate, decrease_rate, max_epochs,
                         diff_threshold, exact, num_mc_trajectories, prune_threshold, horizon, processes, seed)

    def learn(self,
              trajectories: List[TeamModelDistTrajectory],
              data_id: Optional[str] = None,
              verbose: bool = False) -> ModelLearningResult:
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[TeamModelDistTrajectory] trajectories: a list of team trajectories, each
        containing a list (sequence) of state-team_action-model_dist tuples demonstrated by an "expert" in the task.
        :param str data_id: an (optional) identifier for the data for which model learning was performed.
        :param bool verbose: whether to show information at each timestep during learning.
        :rtype: ModelLearningResult
        :return: the result of the model learning procedure.
        """
        # get empirical feature counts (mean feature path) from trajectories
        feature_func = lambda s: self.reward_vector.get_values(s)
        empirical_fc = empirical_feature_counts(trajectories, feature_func)
        if verbose:
            with np.printoptions(precision=2, suppress=True):
                logging.info(f'Empirical feature counts: {empirical_fc}')

        # gets parameters from given trajectories (considered consistent)
        old_rationality = self.agent.getAttribute('rationality', model=self.agent.get_true_model())
        self.agent.setAttribute('rationality', 1.)  # MaxEnt IRL modeling criterion

        # 1 - initiates reward weights (uniform)
        self.theta = np.ones(self.num_features) / self.num_features

        # 2 - perform gradient descent to optimize reward weights
        diff = np.float('inf')
        e = 0
        step_time = 0
        learning_rate = self.learning_rate
        diffs = [1.] if self.normalize_weights else []
        thetas = [self.theta]
        times = []
        rates = []

        while diff > self.diff_threshold and e < self.max_epochs:
            if verbose:
                self.log_progress(e, self.theta, diff, learning_rate, step_time)

            start = timer()

            # update learning rate
            learning_rate = self.learning_rate
            if self.decrease_rate:
                learning_rate *= np.power(LEARNING_DECAY, e)

            self.reward_vector.set_rewards(self.agent, self.theta)  # set reward function with new weights

            # gets expected feature counts (mean feature path) of using the new reward function
            # by computing the efc using a MaxEnt stochastic policy given the current reward
            # MaxEnt uses a rational agent and we need the distribution over actions if exact
            expected_fc = estimate_feature_counts_with_inference(
                self.agent, trajectories,
                n_trajectories=self.num_mc_trajectories,
                feature_func=feature_func,
                select=True,
                horizon=self.horizon,
                selection='distribution',
                threshold=self.prune_threshold,
                processes=self.processes,
                seed=self.seed,
                verbose=False,
                use_tqdm=True)

            if verbose:
                logging.info(f'Estimated Feature Counts {expected_fc}')

            # gradient descent step, update reward weights
            grad = empirical_fc - expected_fc
            new_theta = self.theta + learning_rate * grad
            if self.normalize_weights:
                new_theta /= np.linalg.norm(new_theta, 1)

            step_time = timer() - start

            # registers stats
            diff = np.linalg.norm(new_theta - self.theta)
            diffs.append(diff)
            self.theta = new_theta
            thetas.append(self.theta)
            times.append(step_time)
            rates.append(learning_rate)
            e += 1

        if verbose:
            self.log_progress(e, self.theta, diff, learning_rate, step_time)
            logging.info(f'Finished, total time: {sum(times):.2f} secs.')

        self.agent.setAttribute('rationality', old_rationality)

        # returns stats dictionary
        return ModelLearningResult(data_id, trajectories, {
            FEATURE_COUNT_DIFF_STR: np.array([diffs]),
            REWARD_WEIGHTS_STR: np.array(thetas).T,
            THETA_STR: self.theta,
            TIME_STR: np.array([times]),
            LEARN_RATE_STR: np.array([rates])
        })
