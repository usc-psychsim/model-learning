import os
import numpy as np
import random
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld
from psychsim.world import World
from psychsim.agent import Agent
from model_learning.planning import get_policy, get_action_values
from psychsim.pwl import makeTree, equalRow, incrementMatrix, stateKey, WORLD, rewardKey, KeyedPlane, KeyedVector
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector
from model_learning.evaluation.metrics import policy_divergence, policy_mismatch_prob


GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 2
ENV_SEED = 48
NUM_EXIST = 2

TEAM_AGENTS = ['AHA', 'Helper1']
AGENT_ROLES = [{'Goal': 1, 'Navigator': -0.1}, {'Goal': 0.05, 'Navigator': 1}]
HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 5e-2
ACT_SELECTION = 'random'

# common params

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/property-world')
NUM_TRAJECTORIES = 3  # 10
TRAJ_LENGTH = 15  # 15
PROCESSES = -1
DEBUG = 0

if __name__ == '__main__':
    world = World()
    world.setParallel()
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

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    # gets all env states
    states = env.get_all_states_with_properties(team[0], team[1])
    print('Number of possible states: ', len(states))
    random.seed(ENV_SEED)
    states = random.sample(states, 100)
    print('Number of selected states: ', len(states))

    # for s in states[100:200]:
    #     print(s)
    team_expert_pi = []
    for ag_i, agent in enumerate(team):
        expert_pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD, processes=PROCESSES)
        team_expert_pi.append(expert_pi)

    # reset reward to learner's reward
    for ag_i, agent in enumerate(team):
        agent_lrv = team_rwd[ag_i]
        if ag_i == 0:
            # agent_lrv.set_rewards(agent, np.array([0,0,0,0,0,0]))
            weights = np.array([0.43,  0.08,  0.08,  0.05,  0.04, -0.32])
        else:
            # agent_lrv.set_rewards(agent, np.array([0,0,0,0,0,0]))
            weights = np.array([0.37, 0.12, 0.1,  0.03, 0.03, 0.35])
        agent_lrv.rwd_weights = weights
        agent_lrv.set_rewards(agent, agent_lrv.rwd_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, agent_lrv.rwd_weights)

    team_learner_pi = []
    for ag_i, agent in enumerate(team):
        learner_pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD, processes=PROCESSES)
        team_learner_pi.append(learner_pi)
    for ag_i, agent in enumerate(team):
        print(f'Policy divergence: {policy_divergence(team_expert_pi[ag_i], team_learner_pi[ag_i]):.3f}')
        print(f'Policy mismatch: {policy_mismatch_prob(team_expert_pi[ag_i], team_learner_pi[ag_i]):.3f}')

