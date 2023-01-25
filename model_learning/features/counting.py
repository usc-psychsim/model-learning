import itertools as it
import numpy as np
from typing import List, Callable, Optional, Union, Tuple

from model_learning import Trajectory, TeamModelDistTrajectory, State, TeamModelsDistributions, \
    StateActionProbTrajectory
from model_learning.trajectory import generate_trajectory_distribution, generate_trajectory_distribution_tom
from model_learning.util.mp import run_parallel
from psychsim.agent import Agent
from model_learning.util.math import weighted_nanmean

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def empirical_feature_counts(trajectories: List[StateActionProbTrajectory],
                             feature_func: Callable[[State], np.ndarray],
                             unweighted: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the empirical (mean over paths) feature counts, i.e., the sum of the feature values for each state along
    a trajectory, averaged across all given trajectories and weighted according to the probability of each trajectory.
    :param list[Trajectory] trajectories: a list of trajectories, each containing a sequence of state-action pairs.
    :param Callable feature_func: the function to extract the features out of each state.
    :param bool unweighted: whether to get the raw feature values for each timestep of each trajectory and associated
    probabilities (weights), instead of the weighted feature count average.
    :rtype: np.ndarray
    :return: If `unweighted=False`, returns an array of shape (num_features, ) containing the weighted mean counts for
    each feature over all trajectories. If `unweighted=True`, returns a tuple with an array of shape
    (num_trajectories, timesteps, num_features) containing the raw feature values for each timestep of each trajectory
    and an array of shape (num_traj, timesteps, 1) containing the probabilities associated with each timestep of each
    trajectory.
    """
    # gets feature counts for each timestep of each trajectory
    t_fcs = []
    t_probs = []
    for trajectory in trajectories:
        fcs = []
        probs = []
        for sp in trajectory:
            # gets feature values at this state weighted by its probability, shape: (num_features, )
            fcs.append(feature_func(sp.state))
            probs.append(sp.prob)
        t_probs.append(probs)  # get probs during trajectory, shape: (timesteps, 1)
        t_fcs.append(fcs)  # shape: (timesteps, num_features)

    t_probs = np.array(t_probs)[..., np.newaxis]  # shape: (num_traj, timesteps, 1)
    t_fcs = np.array(t_fcs)  # shape: (num_traj, timesteps, num_features)

    if unweighted:
        return t_fcs, t_probs

    # get weighted average of feature counts/sums
    weighted_fcs = weighted_nanmean(t_fcs, t_probs, axis=0)  # shape: (timesteps, num_features)
    return np.sum(weighted_fcs, axis=0)  # shape: (num_features, )


def estimate_feature_counts(agent: Agent,
                            initial_states: List[State],
                            trajectory_length: int,
                            feature_func: Callable[[State], np.ndarray],
                            exact: bool = False,
                            num_mc_trajectories: int = 100,
                            model: Optional[str] = None,
                            horizon: Optional[int] = None,
                            threshold: Optional[float] = None,
                            unweighted: bool = False,
                            processes: Optional[int] = -1,
                            seed: int = 0,
                            verbose: bool = False,
                            use_tqdm: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimates the expected feature counts by generating trajectories from the given initial states and then computing
    the average feature counts per path.
    :param Agent agent: the agent for which to compute the expected feature counts.
    :param list[State] initial_states: the list of initial states from which trajectories should be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param Callable feature_func: the function to extract the features out of each state.
    :param bool exact: whether the computation of the distribution over paths should be exact (expand stochastic
    branches) or not, in which case Monte Carlo sample trajectories will be generated to estimate the feature counts.
    :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param bool unweighted: whether to get the raw feature values for each timestep of each trajectory and associated
    probabilities (weights), instead of the weighted feature count average.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: np.ndarray
    :return: If `unweighted=False`, returns an array of shape (num_features, ) containing the estimated weighted mean
    counts for each feature over all generated trajectories. If `unweighted=True`, returns a tuple with an array of
    shape (num_trajectories, timesteps, num_features) containing the raw feature values for each timestep of each
    trajectory and an array of shape (num_traj, timesteps, 1) containing the probabilities associated with each
    timestep of each trajectory.
    """

    args = []
    for t, initial_state in enumerate(initial_states):
        args.append((agent, initial_state, trajectory_length, exact, num_mc_trajectories, model, horizon,
                     threshold, processes, seed + t, verbose))
    trajectories = run_parallel(generate_trajectory_distribution, args,
                                processes=processes, use_tqdm=use_tqdm)
    trajectories: List[Trajectory] = list(it.chain(*trajectories))

    # return expected feature counts over all generated trajectories
    return empirical_feature_counts(trajectories, feature_func, unweighted=unweighted)


def estimate_feature_counts_tom(agent: Agent,
                                trajectories: List[TeamModelDistTrajectory],
                                feature_func: Callable[[State], np.ndarray],
                                exact: bool = False,
                                num_mc_trajectories: int = 100,
                                model: Optional[str] = None,
                                horizon: Optional[int] = None,
                                threshold: Optional[float] = None,
                                unweighted: bool = False,
                                processes: int = -1,
                                seed: int = 0,
                                verbose: bool = False,
                                use_tqdm: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimates the expected feature counts by generating trajectories from the initial states in the given trajectories
    and then computing the average feature counts per path.
    This function is applicable to multiagent scenarios where an agent maintains beliefs (probability distributions)
    over the models of other agents, as specified in the given trajectories. These beliefs are used to compute the
    actions of other agents when generating the trajectories.
    :param Agent agent: the agent for which to compute the expected feature counts.
    :param list[TeamModelDistTrajectory] trajectories: a set of demonstrated trajectories specifying both the initial
    states and the beliefs over others' models computed via model inference.
    :param Callable feature_func: the function to extract the features out of each state.
    :param bool exact: whether the computation of the distribution over paths should be exact (expand stochastic
    branches) or not, in which case Monte Carlo sample trajectories will be generated to estimate the feature counts.
    :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param bool unweighted: whether to get the raw feature values for each timestep of each trajectory and associated
    probabilities (weights), instead of the weighted feature count average.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: np.ndarray
    :return: If `unweighted=False`, returns an array of shape (num_features, ) containing the estimated weighted mean
    counts for each feature over all generated trajectories. If `unweighted=True`, returns a tuple with an array of
    shape (num_trajectories, timesteps, num_features) containing the raw feature values for each timestep of each
    trajectory and an array of shape (num_traj, timesteps, 1) containing the probabilities associated with each
    timestep of each trajectory.
    """
    args = []
    for t, trajectory in enumerate(trajectories):
        models_dists: List[TeamModelsDistributions] = [sadp.models_dists for sadp in trajectory]
        args.append((agent, trajectory[0].state, models_dists, exact, num_mc_trajectories, model, horizon,
                     threshold, processes, seed + t, verbose))

    trajectories = run_parallel(generate_trajectory_distribution_tom, args,
                                processes=processes, use_tqdm=use_tqdm)
    trajectories: List[Trajectory] = list(it.chain(*trajectories))
    return empirical_feature_counts(trajectories, feature_func, unweighted=unweighted)
