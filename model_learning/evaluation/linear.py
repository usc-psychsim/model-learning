import logging
import numpy as np
from psychsim.probability import Distribution
from psychsim.world import World
from model_learning.evaluation.metrics import evaluate_internal
from model_learning.features.linear import LinearRewardVector
from model_learning.planning import get_policy

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def cross_evaluation(trajectories, agent_names, rwd_vectors, rwd_weights,
                     rationality=None, horizon=None, threshold=None, processes=None, invalid_val=-1.):
    """
    Performs cross internal evaluation by testing a set of agents under different linear reward weights.
    :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories containing the experts' behavior
    against which we will compare the behavior resulting from the different reward weight vectors, containing several
    sequences of state-action pairs.
    :param list[str] agent_names: the names of the expert agents in each trajectory used for evaluation.
    :param list[LinearRewardVector] rwd_vectors: the reward functions for each agent.
    :param list[np.ndarray] rwd_weights: the reward weight vectors that we want to compare against the expert's behavior.
    :param float rationality: the rationality of agents when computing their policy to compare against the experts'.
    :param int horizon: the agent's planning horizon.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
    :param float invalid_val: the value set to non-compared pairs, used to initialize the confusion matrix.
    :rtype: dict[str, np.ndarray]
    :return: a dictionary containing, for each internal evaluation metric, a matrix of size (num_trajectories,
    num_rwd_weights) with the evaluation comparing each expert's policy (rows) against each reward weight vector (columns).
    """
    assert len(trajectories) == len(agent_names) == len(rwd_vectors), \
        'Different number of trajectories, agent names or reward vectors provided!'

    eval_matrix = {}
    for i, trajectory in enumerate(trajectories):
        agent = trajectory[-1][0].agents[agent_names[i]]
        agent.setAttribute('rationality', rationality)

        # expert's observed "policy"
        expert_states = [w.state for w, _ in trajectory]
        expert_pi = [a for _, a in trajectory]

        # compute agent's policy under each reward function
        for j, rwd_weight in enumerate(rwd_weights):

            rwd_vectors[i].set_rewards(agent, rwd_weights[j])
            with np.printoptions(precision=2, suppress=True):
                logging.info('Computing policy for agent {} with reward {} for {} states...'.format(
                    agent.name, rwd_weights[j], len(expert_states)))
            agent_pi = get_policy(agent, expert_states, None, horizon, 'distribution', threshold, processes)

            # gets internal performance metrics between policies and stores in matrix
            metrics = evaluate_internal(expert_pi, agent_pi)
            for metric_name, metric in metrics.items():
                if metric_name not in eval_matrix:
                    eval_matrix[metric_name] = np.full((len(trajectories), len(rwd_weights)), invalid_val)
                eval_matrix[metric_name][i, j] = metric

    return eval_matrix
