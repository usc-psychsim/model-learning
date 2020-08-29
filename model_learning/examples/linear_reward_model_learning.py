import itertools
import logging
import os
import copy
import numpy as np
from model_learning.metrics import policy_mismatch_prob, policy_divergence
from model_learning.util.math import min_max_scale
from model_learning.util.plot import plot_evolution
from psychsim.world import World
from model_learning.planning import get_policy, get_action_values
from model_learning.util.io import create_clear_dir
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, FEATURE_COUNT_DIFF_STR, REWARD_WEIGHTS_STR, \
    THETA_STR

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = ''

# env params
ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5
ENV_SEED = 1

# expert params
EXPERT_NAME = 'Expert'
EXPERT_THETA = [0.5, -0.4, 0.1, 0., 0.]
EXPERT_RATIONALITY = 1 / 0.1  # inverse temperature
EXPERT_SELECTION = 'random'
EXPERT_SEED = 1
NUM_TRAJECTORIES = 20
TRAJ_LENGTH = 10  # 15

# learning params
NORM_THETA = True
LEARNING_RATE = 0.1  # 0.01
MAX_EPOCHS = 200
THRESHOLD = 0.3  # 1e-3
LEARNING_SEED = 1

# common params
HORIZON = 3
PRUNE_THRESHOLD = 1e-2
PARALLEL = True

OUTPUT_DIR = 'output/examples/linear-reward-learning'
DEBUG = False
IMG_FORMAT = 'pdf'


def _get_feature_vector(state):
    """
    Gets the feature count vector associated with the given state.
    :param VectorDistributionSet state: the PsychSim state for which to get the feature matrix.
    :rtype: np.ndarray
    :return: the feature count vector.
    """
    global env, expert, feat_matrix

    # get agent's XY features
    x, y = env.get_location_features(expert)
    state = copy.deepcopy(state)
    state.collapse({x, y}, False)

    # collects possible locations and associated probabilities
    locs = {}
    for row in state.distributions[state.keyMap[x]].domain():
        x_val = row[x]
        y_val = row[y]
        locs[env.xy_to_idx(x_val, y_val)] = state.distributions[state.keyMap[x]][row]

    # return weighted average of feature vectors
    return np.multiply(feat_matrix[list(locs.keys())],
                       np.array(list(locs.values())).reshape(len(locs), 1)).sum(axis=0)


def _set_reward_function(theta, agent):
    """
    Sets a reward to the agent that is a linear combination of the given weights associated with each object color.
    :param np.ndarray theta: the reward weights associated with each object's outer and inner color.
    :param Agent agent: the agent to which set the reward.
    :return:
    """
    global env
    inner = theta.shape[0] == 2 * env.num_colors
    agent.setAttribute('R', {})
    env.set_linear_color_reward(agent, theta[:env.num_colors], theta[env.num_colors:] if inner else None)


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR, True)

    # create world and objects environment
    world = World()
    world.parallel = False  # PARALLEL
    env = ObjectsGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_OBJECTS, NUM_COLORS, seed=ENV_SEED)
    env.plot(os.path.join(OUTPUT_DIR, 'env.{}'.format(IMG_FORMAT)))

    # create expert and add world dynamics and reward function
    expert = world.addAgent(EXPERT_NAME)
    expert.setAttribute('selection', EXPERT_SELECTION)
    expert.setAttribute('horizon', HORIZON)
    expert.setAttribute('rationality', EXPERT_RATIONALITY)
    env.add_agent_dynamics(expert)
    env.set_linear_color_reward(expert, EXPERT_THETA)

    world.setOrder([{expert.name}])

    # gets all env states
    states = env.get_all_states(expert)

    # gets policy and value
    logging.info('=================================')
    logging.info('Computing expert policy & value function...')
    expert_pi = get_policy(expert, states, selection='distribution', threshold=PRUNE_THRESHOLD)
    pi = np.array([[dist[a] for a in env.agent_actions[expert.name]] for dist in expert_pi])
    expert_q = np.array(get_action_values(
        expert, list(zip(states, itertools.repeat(env.agent_actions[expert.name], len(states))))))
    expert_v = np.max(expert_q, axis=1)
    env.plot_policy(pi, expert_v, os.path.join(OUTPUT_DIR, 'expert-policy.{}'.format(IMG_FORMAT)), 'Expert Policy')

    # gets rewards
    logging.info('Computing expert rewards...')
    expert_r = min_max_scale(np.array([expert.reward(state) for state in states]))
    env.plot_func(expert_r, os.path.join(OUTPUT_DIR, 'expert-reward.{}'.format(IMG_FORMAT)), 'Expert Rewards')

    # generate trajectories using expert's reward and rationality
    logging.info('=================================')
    logging.info('Generating expert trajectories...')
    trajectories = env.generate_trajectories(NUM_TRAJECTORIES, TRAJ_LENGTH, expert,
                                             threshold=PRUNE_THRESHOLD, seed=EXPERT_SEED)
    env.plot_trajectories(trajectories, expert, os.path.join(OUTPUT_DIR, 'expert-trajectories.{}'.format(IMG_FORMAT)),
                          'Expert Trajectories')

    # create learning algorithm and optimize reward weights
    logging.info('=================================')
    logging.info('Starting optimization...')
    feat_matrix = env.get_loc_feature_matrix(True, False)
    # feat_matrix = env.get_dist_feature_matrix(True, False)
    alg = MaxEntRewardLearning(
        'max-ent', expert, feat_matrix.shape[1], _get_feature_vector, _set_reward_function,
        None if PARALLEL else 1, NORM_THETA, LEARNING_RATE, MAX_EPOCHS, THRESHOLD, PRUNE_THRESHOLD, LEARNING_SEED)
    trajectories = [[(w.state, a) for w, a in t] for t in trajectories]
    stats = alg.learn(trajectories, True)

    # saves results/stats
    np.savetxt(os.path.join(OUTPUT_DIR, 'learner-theta.csv'), stats[THETA_STR].reshape(1, -1), '%s', ',',
               header=','.join(['Color {}'.format(i + 1) for i in range(NUM_COLORS)]), comments='')
    plot_evolution(stats[FEATURE_COUNT_DIFF_STR], ['diff'], 'Feature Count Diff. Evolution', None,
                   os.path.join(OUTPUT_DIR, 'evo-feat-diff.{}'.format(IMG_FORMAT)), 'Epoch', 'Feature Difference')
    plot_evolution(stats[REWARD_WEIGHTS_STR], ['Color {}'.format(i + 1) for i in range(feat_matrix.shape[1])],
                   'Reward Parameters Evolution', None,
                   os.path.join(OUTPUT_DIR, 'evo-rwd-weights.{}'.format(IMG_FORMAT)), 'Epoch', 'Weight')

    # set learner's reward into expert for evaluation (compare to true model)
    theta = stats[THETA_STR]
    _set_reward_function(theta, expert)

    # gets policy and value
    logging.info('=================================')
    logging.info('Computing learner policy & value function...')
    learner_pi = get_policy(expert, states, selection='distribution', threshold=PRUNE_THRESHOLD)
    pi = np.array([[dist[a] for a in env.agent_actions[expert.name]] for dist in learner_pi])
    learner_q = np.array(get_action_values(
        expert, list(zip(states, itertools.repeat(env.agent_actions[expert.name], len(states))))))
    learner_v = np.max(learner_q, axis=1)
    env.plot_policy(pi, learner_v,
                    os.path.join(OUTPUT_DIR, 'learner-policy.{}'.format(IMG_FORMAT)), 'Learner Policy')

    # gets rewards
    logging.info('Computing learner rewards...')
    learner_r = min_max_scale(np.array([expert.reward(state) for state in states]))
    env.plot_func(learner_r, os.path.join(OUTPUT_DIR, 'learner-reward.{}'.format(IMG_FORMAT)), 'Learner Rewards')

    logging.info('=================================')
    logging.info('Computing evaluation metrics...')
    logging.info('Policy mismatch: {:.3f}'.format(policy_mismatch_prob(expert_pi, learner_pi)))
    logging.info('Policy divergence: {:.3f}'.format(policy_divergence(expert_pi, learner_pi)))

    logging.info('\nFinished!')
