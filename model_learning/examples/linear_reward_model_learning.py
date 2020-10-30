import itertools
import logging
import os
import numpy as np
from model_learning.evaluation.metrics import policy_mismatch_prob, policy_divergence
from model_learning.util.math import min_max_scale
from psychsim.world import World
from model_learning.planning import get_policy, get_action_values
from model_learning.util.io import create_clear_dir
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.features.objectworld import ObjectsRewardVector
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, THETA_STR

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = ''

# env params
ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5
OUTER_COLORS = True
INNER_COLORS = False
ENV_SEED = 1

# expert params
EXPERT_NAME = 'Expert'
EXPERT_THETA = [0.5, -0.4, 0.1, 0., 0.]
EXPERT_RATIONALITY = 1 / 0.1  # inverse temperature
EXPERT_SELECTION = 'random'
EXPERT_SEED = 1
NUM_TRAJECTORIES = 8  # 20
TRAJ_LENGTH = 10  # 15

# learning params
NORM_THETA = True
LEARNING_RATE = 1e-2  # 0.01
MAX_EPOCHS = 200
THRESHOLD = 1e-3
LEARNING_SEED = 1

# common params
HORIZON = 3
PRUNE_THRESHOLD = 1e-2
PARALLEL = True

OUTPUT_DIR = 'output/examples/linear-reward-learning'
DEBUG = False
IMG_FORMAT = 'pdf'

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
    pi = np.array([[dist.getProb(a) for a in env.agent_actions[expert.name]] for dist in expert_pi])
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
    feat_matrix = env.get_loc_feature_matrix(OUTER_COLORS, INNER_COLORS)
    rwd_vector = ObjectsRewardVector(env, expert, feat_matrix, OUTER_COLORS, INNER_COLORS)
    alg = MaxEntRewardLearning(
        'max-ent', expert, rwd_vector,
        None if PARALLEL else 1, NORM_THETA, LEARNING_RATE, MAX_EPOCHS, THRESHOLD, True, PRUNE_THRESHOLD, HORIZON,
        LEARNING_SEED)
    trajectories = [[(w.state, a) for w, a in t] for t in trajectories]
    result = alg.learn(trajectories, verbose=True)

    # saves results/stats
    alg.save_results(result, OUTPUT_DIR, IMG_FORMAT)

    # set learner's reward into expert for evaluation (compare to true model)
    rwd_vector.set_rewards(expert, result.stats[THETA_STR])

    # gets policy and value
    logging.info('=================================')
    logging.info('Computing learner policy & value function...')
    learner_pi = get_policy(expert, states, selection='distribution', threshold=PRUNE_THRESHOLD)
    pi = np.array([[dist.getProb(a) for a in env.agent_actions[expert.name]] for dist in learner_pi])
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
