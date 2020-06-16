__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

# def evd(env, expert_r, learner_r, trajectories, stochastic=False):
#     """
#     Computes the expected value difference (EVD) between the expert and learner's policies.
#     Measures how suboptimal the learned policy is under the true reward. We find the optimal policy under the learned
#     reward, measure its expected sum of discounted rewards under the true (expert) reward function, and subtract this
#     quantity from the expected sum of discounted rewards under the true (expert) policy.
#     :param Environment env: the MDP/environment.
#     :param np.ndarray expert_r: the true (expert) reward function shaped (n_states, 1).
#     :param np.ndarray learner_r: the reward function derived via IRL shaped (n_states, 1).
#     :param np.ndarray trajectories: the expert's demonstrated trajectories, shaped (n_trajectories, traj_length).
#     :param bool stochastic: whether the MDP's policies should be stochastic for the calculation of the EVD.
#     :rtype: (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
#     :return: a tuple containing: the expected value difference (EVD) between the expert's and learner's policy,
#     the expert's computed policy, the expert's computed value function, the learner's computed policy, the learner's
#     computed value function.
#     """
#     # computes expert value function given expert/true reward
#     expert_q = value_iteration(env.transition_probability, expert_r, env.discount)
#     expert_policy = softmax_policy(expert_q, stochastic)
#     expert_v = np.max(expert_q, axis=1)
#
#     # computes learner value function given the learned/optimized reward under the expert's reward
#     learner_q = value_iteration(env.transition_probability, learner_r, env.discount)
#     learner_policy = softmax_policy(learner_q, stochastic)
#     learner_v = np.max(learner_q, axis=1)
#
#     # initial state probabilities derived from demonstrations
#     p_start_state = np.bincount(trajectories[:, 0, 0], minlength=env.n_states) / trajectories.shape[0]
#
#     # computes difference in expected value
#     evd = np.abs(expert_v.dot(p_start_state) - learner_v.dot(p_start_state))
#
#     return evd, expert_policy, expert_v, learner_policy, learner_v
