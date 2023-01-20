import copy
from timeit import default_timer as timer

import logging
import numpy as np
import os
import pandas as pd
from typing import Optional, List

from model_learning import Trajectory
from model_learning.algorithms import ModelLearningAlgorithm, ModelLearningResult
from model_learning.features.counting import empirical_feature_counts, estimate_feature_counts
from model_learning.features.linear import LinearRewardVector
from model_learning.util.plot import plot_timeseries, plot_bar
from psychsim.agent import Agent

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# stats names
REWARD_WEIGHTS_STR = 'Weights'
FEATURE_COUNT_DIFF_STR = 'Feature Count Diff.'
THETA_STR = 'Optimal Weight Vector'
TIME_STR = 'Time'
LEARN_RATE_STR = 'Learning Rate'
LEARNING_DECAY = 0.9


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

    def __init__(self,
                 label: str,
                 agent: Agent,
                 reward_vector: LinearRewardVector,
                 normalize_weights: bool = True,
                 learning_rate: float = 0.01,
                 decrease_rate: bool = False,
                 max_epochs: int = 200,
                 diff_threshold: float = 1e-2,
                 exact: bool = False,
                 num_mc_trajectories=1000,
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
        :param bool exact: whether the computation of the distribution over paths should be exact (expand stochastic
        branches) or not, in which case Monte Carlo sample trajectories will be generated to estimate the feature counts.
        :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
        :param float prune_threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int horizon: the planning horizon used to compute feature counts.
        :param int seed: the seed to initialize the random number generator.
        """
        super().__init__(label, agent)

        self.reward_vector: LinearRewardVector = reward_vector
        self.processes: int = processes
        self.normalize_weights: bool = normalize_weights
        self.learning_rate: float = learning_rate
        self.max_epochs: int = max_epochs
        self.diff_threshold: float = diff_threshold
        self.decrease_rate: bool = decrease_rate
        self.exact: bool = exact
        self.num_mc_trajectories: int = num_mc_trajectories
        self.prune_threshold: float = prune_threshold
        self.horizon: int = horizon
        self.seed: int = seed

        self.num_features: int = len(reward_vector)
        self.theta: np.ndarray = np.ones(self.num_features) / self.num_features

    def log_progress(self, e: int, diff: float, learning_rate: float, step_time: float, efc: np.ndarray):
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'Step {e}: diff={diff:.3f}, θ={self.theta}, α={learning_rate:.2f}, '
                         f'efc: {efc}, time={step_time:.2f}s')

    def learn(self,
              trajectories: List[Trajectory],
              data_id: Optional[str] = None,
              verbose: bool = False) -> ModelLearningResult:
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[Trajectory] trajectories: a list of trajectories, each
        containing a list (sequence) of state-action pairs demonstrated by an "expert" in the task.
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

        # gets initial states for fc estimation from given trajectories (considered consistent)
        initial_states = [t[0].state for t in trajectories]
        traj_len = len(trajectories[0])

        # change and memorize old agent params
        old_rationality = self.agent.getAttribute('rationality', model=self.agent.get_true_model())
        self.agent.setAttribute('rationality', 1.)  # MaxEnt IRL modeling criterion
        old_reward = copy.copy(self.agent.getAttribute('R', model=self.agent.get_true_model()))

        # 1 - initiates reward weights (uniform)
        self.theta = np.ones(self.num_features) / self.num_features

        # 2 - perform gradient descent to optimize reward weights
        diff = np.inf
        e = 0
        step_time = 0
        learning_rate = self.learning_rate
        diffs = [1.] if self.normalize_weights else []
        thetas = [self.theta]
        times = []
        rates = []

        expected_fc = np.full_like(empirical_fc, np.nan)
        while diff > self.diff_threshold and e < self.max_epochs:
            if verbose:
                self.log_progress(e, diff, learning_rate, step_time, expected_fc)

            start = timer()

            # update learning rate
            learning_rate = self.learning_rate
            if self.decrease_rate:
                learning_rate *= np.power(LEARNING_DECAY, e)

            self.reward_vector.set_rewards(self.agent, self.theta)  # set reward function with new weights

            # gets expected feature counts (mean feature path) of using the new reward function
            # by computing the efc using a MaxEnt stochastic policy given the current reward
            # MaxEnt uses a rational agent and we need the distribution over actions if exact
            expected_fc = estimate_feature_counts(
                self.agent, initial_states,
                trajectory_length=traj_len,
                feature_func=feature_func,
                exact=self.exact,
                num_mc_trajectories=self.num_mc_trajectories,
                horizon=self.horizon,
                threshold=self.prune_threshold,
                processes=self.processes,
                seed=self.seed,
                verbose=False, use_tqdm=True)

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
            self.log_progress(e, diff, learning_rate, step_time, expected_fc)
            logging.info(f'Finished, total time: {sum(times):.2f} secs.')

        self.agent.setAttribute('rationality', old_rationality)
        self.agent.setAttribute('R', old_reward)

        # returns stats dictionary
        return ModelLearningResult(self.agent.name, data_id, trajectories, {
            FEATURE_COUNT_DIFF_STR: np.array(diffs),  # shape (timesteps, )
            REWARD_WEIGHTS_STR: np.array(thetas),  # shape (timesteps, n_features)
            THETA_STR: self.theta,  # shape (n_features, )
            TIME_STR: np.array(times),  # shape (timesteps, )
            LEARN_RATE_STR: np.array(rates)  # shape (timesteps, )
        })

    def save_results(self, result: ModelLearningResult, output_dir: str, img_format: str):
        """
        Saves the several results of a run of the algorithm to the given directory.
        :param ModelLearningResult result: the results of the algorithm run.
        :param str output_dir: the path to the directory in which to save the results.
        :param str img_format: the format of the images to be saved.
        :return:
        """
        stats = result.stats

        plot_bar(pd.DataFrame(stats[THETA_STR].reshape(1, -1), columns=self.reward_vector.names),
                 'Optimal Weight Vector',
                 os.path.join(output_dir, f'learner-theta.{img_format}'),
                 x_label='Reward Features', y_label='Weight')

        plot_timeseries(pd.DataFrame(stats[FEATURE_COUNT_DIFF_STR].reshape(-1, 1), columns=['diff']),
                        'Evolution of Reward Parameters Difference',
                        os.path.join(output_dir, f'evo-rwd-weights-diff.{img_format}'),
                        x_label='Epoch', y_label='Abs. Weight Diff.', show_legend=False)

        plot_timeseries(pd.DataFrame(stats[REWARD_WEIGHTS_STR], columns=self.reward_vector.names),
                        'Evolution of Reward Parameters',
                        os.path.join(output_dir, f'evo-rwd-weights.{img_format}'),
                        x_label='Epoch', y_label='Weight', var_label='Reward Feature')

        plot_timeseries(pd.DataFrame(stats[TIME_STR].reshape(-1, 1), columns=['time']),
                        'Evolution of Epoch Time',
                        os.path.join(output_dir, f'evo-time.{img_format}'),
                        x_label='Epoch', y_label='Time (secs.)', show_legend=False)

        plot_timeseries(pd.DataFrame(stats[LEARN_RATE_STR].reshape(-1, 1), columns=['learning rate']),
                        'Evolution of Learning Rate',
                        os.path.join(output_dir, f'learning-rate.{img_format}'),
                        x_label='Epoch', y_label='Learning Rate', show_legend=False)
