import copy

import logging
import numpy as np
import os

from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld
from model_learning.features import LinearRewardVector
from model_learning.util.io import create_clear_dir
from model_learning.util.logging import change_log_handler
from psychsim.pwl import modelKey, stateKey, WORLD
from psychsim.world import World

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'
CLEARINDICATOR_FEATURE = 'ci'
MARK_FEATURE = 'm'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 3

TEAM_AGENTS = ['Medic', 'Explorer']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params

OUTPUT_DIR = 'output/examples/property_world_trajectories'
NUM_TRAJECTORIES = 5  # 10
TRAJ_LENGTH = 25  # 30
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=3)

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'collect.log'), level=logging.INFO)

    world = World()
    world.setParallel()
    env = SearchRescueGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, track_feature=False, seed=ENV_SEED)
    logging.info('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')

    # team of two agents
    team = []
    for ag_i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[ag_i])
        # define agent dynamics
        env.add_location_property_dynamics(agent, idle=True)
        team.append(agent)
    # collaboration dynamics
    env.add_collaboration_dynamics([agent for agent in team])

    # set agent rewards, and attributes
    # logging.info(env.world.symbolList)
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
        agent_lrv = LinearRewardVector(rwd_features)
        rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
        agent_lrv.set_rewards(agent, rwd_f_weights)
        logging.info(f'{agent.name} Reward Features')
        logging.info(agent_lrv.names, rwd_f_weights)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', RATIONALITY)
        agent.setAttribute('discount', 0.7)

    # logging.info('Action Dynamics')
    # env.vis_agent_dynamics_in_xy()

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    # "compile" reward functions to get trees
    world.dependency.getEvaluation()

    if DEBUG:
        for i in range(TRAJ_LENGTH):
            logging.info('Step', i)
            # logging.info(env.world.state.items())
            prob = env.world.step()
            logging.info(f'probability: {prob}')
            state = copy.deepcopy(env.world.state)
            p_state = []
            for ag_i, agent in enumerate(team):
                if ag_i in [0, 1]:
                    logging.info(f'{agent.name} action: {world.getAction(agent.name)}')
                    x, y = env.get_location_features(agent)
                    f = env.get_navi_features(agent)
                    loc_i = env.xy_to_idx(env.world.getFeature(x, unique=True),
                                          env.world.getFeature(y, unique=True))
                    d2c = env.get_d2c_feature(agent)
                    d2h = env.get_d2h_feature(agent)
                    logging.info(f'{agent.name} state: '
                                 f'x={env.world.getFeature(x, unique=True)}, '
                                 f'y={env.world.getFeature(y, unique=True)}, '
                                 f'loc={loc_i}, '
                                 f'd2c={env.world.getFeature(d2c, unique=True)}, '
                                 f'd2h={world.getFeature(d2h, unique=True)}, '
                                 f'r={agent.reward(env.world.state)}')
                    if env.track_feature:
                        # visits = env.get_visit_feature(agent)
                        logging.info(f'f={env.world.getFeature(env.get_navi_features(agent), unique=True)}'
                                     # f'v={world.getFeature(visits[loc_i], unique=True)}
                                     )

                    p_state = []
                    for loc_i in range(env.width * env.height):
                        p = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE + f'{loc_i}'), unique=True)
                        p_state.append(PROPERTY_LIST[p])
                    ci_state = env.world.getFeature(stateKey(WORLD, CLEARINDICATOR_FEATURE), unique=True)
                    m_state = env.world.getFeature(stateKey(WORLD, MARK_FEATURE), unique=True)
                    logging.info('Locations:', env.exist_locations, 'Properties:',
                                 f'p={p_state}\n', 'Indicator:', f'ci={ci_state}', 'Mark:',
                                 f'm={m_state}')
                    if env.track_feature:
                        logging.info('Clear:', f'g={env.world.getFeature(stateKey(WORLD, GOAL_FEATURE), unique=True)}')

                    decision = agent.decide(state, horizon=HORIZON, selection='distribution')
                    if 'V' in decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)].keys():
                        for k, v in \
                                decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)][
                                    'V'].items():
                            logging.info(k)
                            logging.info('EV', v['__EV__'])
                            logging.info('ER', v['__ER__'])
                    logging.info(decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)][
                                     'action'])

            if sum([p == 'clear' for p in p_state]) == env.num_exist:
                logging.info(p_state)
                logging.info('Reward:', team[1].reward(env.world.state))
                # break

    else:

        # generate trajectories using agent's policy #ACT_SELECTION
        # Team trajectory functions
        if not DEBUG:
            team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                               horizon=HORIZON, selection=ACT_SELECTION,
                                                               processes=PROCESSES,
                                                               threshold=1e-2, seed=ENV_SEED)
            env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)
