import os
import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld
from psychsim.world import World
from psychsim.agent import Agent
from model_learning.planning import get_policy, get_action_values
from psychsim.pwl import makeTree, equalRow, incrementMatrix, stateKey, WORLD, rewardKey, KeyedPlane, KeyedVector
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector

GOAL_FEATURE = 'g'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 2

AGENT_NAME = 'AHA'
HORIZON = 2  # 0 for random actions
ACT_SELECTION = 'random'

# common params

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/property-world')
NUM_TRAJECTORIES = 3  # 10
TRAJ_LENGTH = 15  # 15
PROCESSES = -1
DEBUG = 0

if __name__ == '__main__':
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE-2, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')
    p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
    p_state = env.p_state[p_feat]
    # print("Starting Property State:", p_state)
    # print(env.property_goal_count)

    # team of two agents
    team = [world.addAgent(AgentRoles('AHA', ['Goal'])),
            world.addAgent(AgentRoles('Helper1', ['Navigator', 'SubGoal']))]

    # # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=True)
    env.add_collaboration_dynamics([agent for agent in team])
    print('Action Dynamics')
    # env.vis_agent_dynamics_in_xy()


    # set agent rewards, and attributes
    print('Reward Dynamics and Features')
    print(env.world.symbolList)
    for agent in team:
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        print(agent_lrv.names, agent_lrv.rwd_weights)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    states = env.get_all_states_with_properties(team[0], team[1])

    # generate trajectories using agent's policy #ACT_SELECTION
    # Team trajectory functions
    if not DEBUG:
        team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                           horizon=HORIZON, selection=ACT_SELECTION,
                                                           processes=PROCESSES,
                                                           threshold=1e-2, seed=ENV_SEED)
        env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)

    if DEBUG:
        for i in range(20):
            print('Step', i)
            # print(env.world.state.items())
            prob = env.world.step()
            print(f'probability: {prob}')
            for ag_i, agent in enumerate(team):
                print(f'{agent.name} action: {world.getAction(agent.name)}')
                x, y = env.get_location_features(agent)
                visits = env.get_visit_feature(agent)
                loc_i = env.xy_to_idx(world.getFeature(x, unique=True), world.getFeature(y, unique=True))
                print(f'{agent.name} state: '
                      f'x={world.getFeature(x, unique=True)}, '
                      f'y={world.getFeature(y, unique=True)}, '
                      f'v={world.getFeature(visits[loc_i], unique=True)}, '
                      f'r={agent.reward(env.world.state)}')
            p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
            p_state = env.p_state[p_feat]
            g_state = env.world.getFeature(stateKey(WORLD, GOAL_FEATURE), unique=True)
            print('Locations:', env.exist_locations, 'Properties:', f'p={p_state}', 'Clear:', f'g={g_state}')

            if list(p_state.values()) == [PROPERTY_LIST[-1]] * env.num_exist:
                p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
                p_state = env.p_state[p_feat]
                print(p_feat, p_state)
                print(env.property_goal_count)
                print('Reward:', team[1].reward(env.world.state))
                # break
