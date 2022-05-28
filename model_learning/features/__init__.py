import numpy as np
from typing import List, Callable, Optional, Literal
from psychsim.agent import Agent
from model_learning import Trajectory, State
from model_learning.trajectory import generate_trajectory, generate_trajectories, copy_world

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def expected_feature_counts(trajectories: List[Trajectory], feature_func: Callable[[State], np.ndarray]) -> np.ndarray:
    """
    Computes the expected (mean over paths) feature counts, i.e., the sum of the feature values for each state along
    a trajectory, averaged across all given trajectories and weighted according to the probability of each trajectory.
    :param list[Trajectory] trajectories: a list of trajectories, each containing a sequence of state-action pairs.
    :param Callable feature_func: the function to extract the features out of each state.
    :rtype: np.ndarray
    :return: the mean counts for each feature over all trajectories.
    """
    # gets feature counts for each timestep of each trajectory
    t_fcs = []
    t_probs = []
    for trajectory in trajectories:
        fcs = []
        probs = []
        for sap in trajectory:
            # gets feature values at this state weighted by its probability, shape: (num_features, )
            fcs.append(feature_func(sap.world.state) * sap.prob)
            probs.append(sap.prob)
        t_probs.append(np.array(probs).reshape(-1, 1))  # get probs during trajectory, shape: (timesteps, 1)
        t_fcs.append(np.array(fcs))  # shape: (timesteps, num_features)

    t_probs = np.array(t_probs)  # shape: (num_traj, timesteps, 1)
    prob_weights = np.sum(t_probs, axis=0)  # shape: (timesteps, 1)
    t_fcs = np.array(t_fcs)  # shape: (num_traj, timesteps, num_features)
    fcs_weighted = np.sum(t_fcs, axis=0) / prob_weights  # shape: (timesteps, num_features)
    return np.sum(fcs_weighted, axis=0)  # get weighted average of feature counts/sums, shape: (num_features, )


def estimate_feature_counts(agent: Agent,
                            initial_states: List[State],
                            trajectory_length: int,
                            feature_func: Callable[[State], np.ndarray],
                            exact: bool = False,
                            num_mc_trajectories: int = 100,
                            model: Optional[str] = None,
                            horizon: Optional[int] = None,
                            selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                            threshold: Optional[float] = None,
                            processes: Optional[int] = -1,
                            seed: int = 0,
                            verbose: bool = False,
                            use_tqdm: bool = True) -> np.ndarray:
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
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: np.ndarray
    :return: the estimated expected feature counts.
    """
    trajectories = []
    for t in range(len(initial_states)):
        # make copy of world and set initial state
        world = copy_world(agent.world)
        _agent = world.agents[agent.name]
        world.state = initial_states[t]

        if exact:
            # exact computation, generate single stochastic trajectory (select=False) from initial state
            trajectory_dist = generate_trajectory(_agent, trajectory_length, model=model, select=False,
                                                  horizon=horizon, selection='distribution', threshold=threshold,
                                                  seed=seed + t, verbose=verbose)
            trajectories.append(trajectory_dist)
        else:
            # Monte Carlo approximation, generate N single-path trajectories (select=True) from initial state
            trajectories_mc = generate_trajectories(_agent, num_mc_trajectories, trajectory_length,
                                                    model=model, select=True,
                                                    horizon=horizon, selection=selection, threshold=threshold,
                                                    processes=processes, seed=seed + t, verbose=verbose,
                                                    use_tqdm=use_tqdm)
            trajectories.extend(trajectories_mc)

    # return expected feature counts over all generated trajectories
    return expected_feature_counts(trajectories, feature_func)
