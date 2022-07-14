import os
import numpy as np
import random
import copy
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.util.io import create_clear_dir
from model_learning import StateActionPair
from psychsim.world import World
from psychsim.agent import Agent
from model_learning.planning import get_policy, get_action_values
from model_learning.features import expected_feature_counts, estimate_feature_counts
from psychsim.pwl import Distribution
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector
from model_learning.trajectory import generate_trajectory, generate_trajectories, copy_world
from model_learning.evaluation.metrics import policy_divergence, policy_mismatch_prob

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 47
NUM_EXIST = 3

TEAM_AGENTS = ['AHA', 'Helper1']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/test-property-world')
NUM_TRAJECTORIES = 16  # 10
TRAJ_LENGTH = 25  # 15
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=4)
# EVALUATE_BY = 'EPISODES'
EVALUATE_BY = 'FEATURES'
# EVALUATE_BY = 'EMPIRICAL'

if __name__ == '__main__':
    test_ag_i = [0, 1]
    print(test_ag_i)
    create_clear_dir(OUTPUT_DIR, clear=False)
    world = World()
    world.setParallel()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')
    print('Env Seed', ENV_SEED, test_ag_i)


    # team of two agents
    team = []
    for i in range(len(TEAM_AGENTS)):
        team.append(world.addAgent(AgentRoles(TEAM_AGENTS[i], AGENT_ROLES[i])))

    # # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=True)
    env.add_collaboration_dynamics([agent for agent in team])

    # set agent rewards, and attributes
    team_rwd = []
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        agent_lrv.rwd_weights = np.array(agent_lrv.rwd_weights) / np.linalg.norm(agent_lrv.rwd_weights, 1)
        agent_lrv.set_rewards(agent, agent_lrv.rwd_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, agent_lrv.rwd_weights)
        team_rwd.append(agent_lrv)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', RATIONALITY)
        agent.setAttribute('discount', 0.7)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    learner_team = [copy.deepcopy(agent) for agent in team]
    learner_suffix = '_learner'

    learner_team_rwd = []
    for ag_i, agent in enumerate(learner_team):
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        if ag_i == 0:
            # weights = np.array([1, 1, 1, 1, 1])  # feature random reward
            # weights = np.array([0, 0, 0, 0, 0])  # random reward
            # weights = np.array([0.2,    0.181,  0.1814, 0.4105, 0.0271])  # IRL
            weights = np.array([0.2, 0.181, 0.1814, 0.0105, 0.0271])  # IRL
            # weights = np.array([0.129, 0.129, 0.645, 0.032, 0.065])  # gt
        else:
            # weights = np.array([1, 1, 1])  # feature random reward
            # weights = np.array([0, 0, 0])  # random reward
            weights = np.array([0.4567, 0.2724, 0.2709])  # IRL
            # weights = np.array([0.25, 0.25, 0.5])  # gt
        agent_lrv.rwd_weights = np.array(weights) / np.linalg.norm(weights, 1)
        # agent_lrv.rwd_weights = weights
        agent_lrv.set_rewards(agent, agent_lrv.rwd_weights)
        print(f'{agent.name + learner_suffix} Reward Features')
        print(agent_lrv.names, agent_lrv.rwd_weights)
        learner_team_rwd.append(agent_lrv)

    if EVALUATE_BY == 'EMPIRICAL':
        team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH,
                                                           n_trajectories=NUM_TRAJECTORIES,
                                                           horizon=HORIZON, selection=ACT_SELECTION,
                                                           processes=PROCESSES,
                                                           threshold=1e-2, seed=ENV_SEED)
        team_trajs = [] * len(team)
        for ag_i, agent in enumerate(team):
            if ag_i in test_ag_i:
                agent_trajs = []
                for team_traj in team_trajectories:
                    agent_traj = []
                    for team_step in team_traj:
                        tsa = team_step
                        sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                        agent_traj.append(sa)
                    agent_trajs.append(agent_traj)
                team_trajs.append(agent_trajs)

                feature_func = lambda s: team_rwd[ag_i].get_values(s)
                empirical_fc = expected_feature_counts(agent_trajs, feature_func)
                print(empirical_fc)

    if EVALUATE_BY == 'FEATURES':
        team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH,
                                                           n_trajectories=NUM_TRAJECTORIES,
                                                           horizon=HORIZON, selection=ACT_SELECTION,
                                                           processes=PROCESSES,
                                                           threshold=1e-2, seed=ENV_SEED)
        team_trajs = [] * len(team)
        for ag_i, agent in enumerate(team):
            if ag_i in test_ag_i:
                agent_trajs = []
                for team_traj in team_trajectories:
                    agent_traj = []
                    for team_step in team_traj:
                        tsa = team_step
                        sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                        agent_traj.append(sa)
                    agent_trajs.append(agent_traj)
                team_trajs.append(agent_trajs)

                feature_func = lambda s: team_rwd[ag_i].get_values(s)
                empirical_fc = expected_feature_counts(agent_trajs, feature_func)
                print(empirical_fc)

                initial_states = [t[0].world.state for t in agent_trajs]  # initial states for fc estimation
                learner_world = copy_world(learner_team[ag_i].world)
                learner_agent = learner_world.agents[learner_team[ag_i].name]
                learner_feature_func = lambda s: learner_team_rwd[ag_i].get_values(s)
                traj_len = len(agent_trajs[0])

                expected_fc = estimate_feature_counts(learner_agent, initial_states, traj_len, learner_feature_func,
                                                      exact=False, num_mc_trajectories=16,
                                                      horizon=HORIZON, threshold=PRUNE_THRESHOLD,
                                                      processes=PROCESSES, seed=ENV_SEED,
                                                      verbose=False, use_tqdm=True)
                print(expected_fc)
                diff = empirical_fc - expected_fc
                print(agent.name)
                print(f'Feature count different:', diff, f'={np.sum(np.abs(diff))}')

    if EVALUATE_BY == 'EPISODES':
        team_trajectories = env.generate_expert_learner_trajectories(team, learner_team, TRAJ_LENGTH,
                                                                     n_trajectories=NUM_TRAJECTORIES,
                                                                     horizon=HORIZON, selection=ACT_SELECTION,
                                                                     processes=PROCESSES,
                                                                     threshold=1e-2, seed=ENV_SEED)
        # print(team_trajectories)
        team_pi: Dict[str, List[Distribution]] = {}
        expert_and_learner = [agent.name for agent in team] + [agent.name + learner_suffix for agent in learner_team]
        for agent_name in expert_and_learner:
            team_pi[agent_name] = []
        for team_traj in team_trajectories:
            for tsa in team_traj:
                for agent_name, agent_action in tsa.action.items():
                    team_pi[agent_name].append(agent_action)
                    # print(tsa.action)
        # print(team_pi)
        for ag_i, agent in enumerate(team):
            print(agent.name)
            print(f'Policy divergence:'
                  f' {policy_divergence(team_pi[agent.name], team_pi[agent.name + learner_suffix]):.3f}')
        # env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)