import logging
import os
import numpy as np
from psychsim.world import World
from psychsim.pwl import stateKey, WORLD
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, THETA_STR
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.evaluation.metrics import policy_mismatch_prob, policy_divergence
from model_learning.planning import get_policy, get_action_values
from model_learning.util.logging import change_log_handler
from model_learning.util.io import create_clear_dir
from model_learning import StateActionPair


__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Performs Multiagent IRL (reward model learning) in the Property World using MaxEnt IRL.'

# env params
GOAL_FEATURE = 'g'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 2
ENV_SEED = 48
NUM_EXIST = 2

# expert params
TEAM_AGENTS = ['AHA', 'Helper1']
AGENT_ROLES = [{'Goal': 1, 'Navigator': -0.1}, {'Goal': 0.05, 'Navigator': 1}]

EXPERT_RATIONALITY = 1 / 0.1  # inverse temperature
EXPERT_ACT_SELECTION = 'random'
EXPERT_SEED = 17
NUM_TRAJECTORIES = 3
TRAJ_LENGTH = 15

# learning params
NORM_THETA = True
LEARNING_RATE = 1e-2  # 0.01
MAX_EPOCHS = 100
THRESHOLD = 1e-2
DECREASE_RATE = True
EXACT = False
NUM_MC_TRAJECTORIES = 10 #200
LEARNING_SEED = 17

# common params
HORIZON = 2
PRUNE_THRESHOLD = 5e-2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/multiagent-property-world')
PROCESSES = -1
VERBOSE = True

if __name__ == '__main__':
    # create output
    learning_ag_i = 0
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, f'irl_gf_{TEAM_AGENTS[learning_ag_i]}.log'), logging.INFO)

    # create world and objects environment
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')

    # team of two agents
    team = []
    for i in range(len(TEAM_AGENTS)):
        team.append(world.addAgent(AgentRoles(TEAM_AGENTS[i], AGENT_ROLES[i])))

    # # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=True)
    env.add_collaboration_dynamics([agent for agent in team])
    # print('Action Dynamics')
    # env.vis_agent_dynamics_in_xy()

    # set agent rewards, and attributes
    # print(env.world.symbolList)
    team_rwd = []
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, agent_lrv.rwd_weights)
        team_rwd.append(agent_lrv)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', EXPERT_ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', EXPERT_RATIONALITY)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    # gets all env states
    states = env.get_all_states_with_properties(team[0], team[1])

    # generate trajectories using expert's reward and rationality
    logging.info(f'{TEAM_AGENTS[learning_ag_i]} Reward Features')
    logging.info(team_rwd[learning_ag_i].names)
    logging.info(team_rwd[learning_ag_i].rwd_weights)
    logging.info('=================================')
    logging.info('Generating expert trajectories...')
    team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                       horizon=HORIZON, selection=EXPERT_ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)
    # env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)

    # create learning algorithm and optimize reward weights
    # TODO
    # for loop, MaxEnt for each agent 
    logging.info('=================================')
    logging.info('Starting MaxEnt IRL optimization...')

    agent = team[learning_ag_i]
    rwd_vector = team_rwd[learning_ag_i]
    agent_trajs = []
    for team_traj in team_trajectories:
        agent_traj = []
        for team_step in team_traj:
            tsa = team_step
            sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
            agent_traj.append(sa)
        agent_trajs.append(agent_traj)

    alg = MaxEntRewardLearning(
        'max-ent', agent.name, rwd_vector,
        processes=PROCESSES,
        normalize_weights=NORM_THETA,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        diff_threshold=THRESHOLD,
        decrease_rate=DECREASE_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        exact=EXACT,
        num_mc_trajectories=NUM_MC_TRAJECTORIES,
        horizon=HORIZON,
        seed=LEARNING_SEED)
    result = alg.learn(agent_trajs, verbose=True)