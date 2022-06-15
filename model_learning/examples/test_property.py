import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld, GridWorld
from psychsim.world import World
from psychsim.agent import Agent
from psychsim.pwl import makeTree, equalRow, incrementMatrix, stateKey, WORLD, rewardKey, KeyedPlane, KeyedVector
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector,\
    PropertyActionComparisonLinearRewardFeature

GOAL_FEATURE = 'g'
# MOVEMENT = ['right', 'left', 'up', 'down']

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 2
ENV_SEED = 45
NUM_EXIST = 2

AGENT_NAME = 'AHA'
HORIZON = 0  # 0 for random actions
ACT_SELECTION = 'random'


if __name__ == '__main__':
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE - 1, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
    p_state = env.p_state[p_feat]
    print("Starting Property State:", p_state)
    print(env.property_goal_count)

    # team of two agents
    team = [world.addAgent(AgentRoles('AHA', ['Goal', 'SubGoal'])),
            world.addAgent(AgentRoles('Helper1', ['Goal', 'Navigator', 'SubGoal']))]

    # # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=False)
    env.add_collaboration_dynamics([agent for agent in team])
    print('Action Dynamics')
    env.vis_agent_dynamics_in_xy()
    # set agent rewards, and attributes
    # print(env.world.state)
    print('Reward Dynamics and Features')
    print(env.world.symbolList)
    for agent in team:
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        print(agent_lrv.names, agent_lrv.rwd_weights)
        agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    for i in range(100):
        print('Step', i)
        # print(env.world.state.items())
        prob = env.world.step()
        print(f'probability: {prob}')
        for ag_i, agent in enumerate(team):
            print(f'{agent.name} action: {world.getAction(agent.name)}')
            x, y = env.get_location_features(agent)
            print(f'{agent.name} state: '
                  f'x={world.getFeature(x, unique=True)}, '
                  f'y={world.getFeature(y, unique=True)}, '
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
            break
