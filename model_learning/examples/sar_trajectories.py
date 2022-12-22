import copy

import logging
import numpy as np
import os

from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld, AgentOptions
from model_learning.features import LinearRewardVector
from model_learning.util.io import create_clear_dir
from model_learning.util.logging import change_log_handler
from psychsim.pwl import modelKey
from psychsim.world import World

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'

WORLD_NAME = 'SAR'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 3

# Medic has all actions available and distance to victim feature
# Explorer has search action and distance to help feature
VIC_STATS_FEATURES = False
TEAM_AGENTS = ['Medic', 'Explorer']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]
AGENT_OPTIONS = [AgentOptions(noop_action=True, search_action=True, triage_action=True, call_action=True,
                              num_empty_feature=False, dist_to_vic_feature=True, dist_to_help_feature=False),
                 AgentOptions(noop_action=False, search_action=True, triage_action=False, call_action=False,
                              num_empty_feature=False, dist_to_vic_feature=False, dist_to_help_feature=True)]  # Exp

DISCOUNT = 0.7
HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params
OUTPUT_DIR = 'output/examples/sar_trajectories'
NUM_TRAJECTORIES = 5  # 10
TRAJ_LENGTH = 25  # 30
PROCESSES = -1
DEBUG = False
np.set_printoptions(precision=3)

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'collect.log'), level=logging.INFO)

    world = World()
    world.setParallel()
    env = SearchRescueGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME,
                                vics_cleared_feature=VIC_STATS_FEATURES, seed=ENV_SEED)
    logging.info(f'Initialized World, h:{HORIZON}, x:{env.width}, y:{env.height}, v:{env.num_victims}')

    # team of two agents
    team = []
    for i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[i])
        # define agent dynamics
        env.add_search_and_rescue_dynamics(agent, AGENT_OPTIONS[i])
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
        agent.setAttribute('discount', DISCOUNT)

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
            vic_status_state = []
            for ag_i, agent in enumerate(team):
                if ag_i in [0, 1]:
                    logging.info(f'{agent.name} action: {world.getAction(agent.name)}')
                    x, y = env.get_location_features(agent, key=True)
                    f = env.get_empty_feature(agent, key=True)
                    loc_idx = env.xy_to_idx(env.world.getFeature(x, unique=True),
                                            env.world.getFeature(y, unique=True))
                    d2c = env.get_dist_to_vic_feature(agent, key=True)
                    d2h = env.get_dist_to_help_feature(agent, key=True)
                    logging.info(f'{agent.name} state: '
                                 f'x={env.world.getFeature(x, unique=True)}, '
                                 f'y={env.world.getFeature(y, unique=True)}, '
                                 f'loc={loc_idx}, '
                                 f'd2c={env.world.getFeature(d2c, unique=True)}, '
                                 f'd2h={world.getFeature(d2h, unique=True)}, '
                                 f'r={agent.reward(env.world.state)}')
                    if env.vics_cleared_feature:
                        # visits = env.get_visit_feature(agent)
                        logging.info(f'f={env.world.getFeature(env.get_empty_feature(agent, key=True), unique=True)}'
                                     # f'v={world.getFeature(visits[loc_i], unique=True)}
                                     )

                    vic_status_state = []
                    for loc_idx in range(env.width * env.height):
                        vic_status = env.world.getFeature(env.get_loc_vic_status_feature(loc_idx, key=True),
                                                          unique=True)
                        vic_status_state.append(vic_status)
                    ci_state = env.world.getFeature(env.get_vic_clear_comb_feature(key=True), unique=True)
                    m_state = env.world.getFeature(env.get_help_loc_feature(key=True), unique=True)
                    logging.info('Locations:', env.victim_locs, 'Properties:',
                                 f'p={vic_status_state}\n', 'Indicator:', f'ci={ci_state}', 'Mark:',
                                 f'm={m_state}')
                    if env.vics_cleared_feature:
                        logging.info('Clear:',
                                     f'g={env.world.getFeature(env.get_vics_cleared_feature(key=True), unique=True)}')

                    decision = agent.decide(state, horizon=HORIZON, selection='distribution')
                    if 'V' in decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)].keys():
                        for k, v in decision[agent.world.getFeature(
                                modelKey(agent.name), state=state, unique=True)]['V'].items():
                            logging.info(k)
                            logging.info('EV', v['__EV__'])
                            logging.info('ER', v['__ER__'])
                    logging.info(decision[agent.world.getFeature(
                        modelKey(agent.name), state=state, unique=True)]['action'])

            if sum([p == 'clear' for p in vic_status_state]) == env.num_victims:
                logging.info(vic_status_state)
                logging.info('Reward:', team[1].reward(env.world.state))
                # break

    else:

        # generate trajectories using agent's policy #ACT_SELECTION
        team_trajectories = env.generate_team_trajectories(
            team,
            trajectory_length=TRAJ_LENGTH,
            n_trajectories=NUM_TRAJECTORIES,
            horizon=HORIZON,
            selection=ACT_SELECTION,
            processes=PROCESSES,
            threshold=1e-2,
            seed=ENV_SEED)
        env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)
