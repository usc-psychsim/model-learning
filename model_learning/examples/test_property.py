import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld
from psychsim.world import World
import random
from psychsim.reward import maximizeFeature
from psychsim.pwl import makeTree, equalRow, incrementMatrix, stateKey, WORLD, REWARD, rewardKey, makeFuture
from psychsim.agent import Agent, isModelKey, isRewardKey

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 45
NUM_EXIST = 2

AGENT_NAME = 'AHA'
HORIZON = 0  # 0 for random actions
ACT_SELECTION = 'random'


if __name__ == '__main__':
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE-2, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
    p_state = env.p_state[p_feat]
    print(p_state)

    # add agents
    agent = world.addAgent('AHA')
    agent2 = world.addAgent('Helper')
    # define agent dynamics
    env.add_location_property_dynamics(agent, idle=False)
    env.add_location_property_dynamics(agent2, idle=False)
    env.add_collaboration_dynamics([agent, agent2])
    # env.vis_agent_dynamics_in_xy()

    # # observability
    # unobservable = set()
    # unobservable.add(env.get_property_features())
    # agent.set_observations(unobservable)
    # print(agent.omega)

    # print(env.p_state)
    # print(f"{agent.name} actions")
    # print(env.agent_actions[agent.name])
    # print(f"{agent2.name} actions")
    # print(env.agent_actions[agent2.name])

    env.set_achieve_property_reward(agent, 5)
    env.set_achieve_property_reward(agent2, 5)
    print('Reward Dynamics')
    agent.printReward(agent.get_true_model())

    print(agent.getReward(agent.get_true_model()))
    print(agent2.getReward(agent2.get_true_model()))
    # print(env.world.getFeature(rewardKey(agent.name), unique=True))
    #

    agent.setAttribute('selection', ACT_SELECTION)
    agent2.setAttribute('selection', ACT_SELECTION)
    agent.setAttribute('horizon', HORIZON)
    agent2.setAttribute('horizon', HORIZON)
    my_turn_order = [{agent.name, agent2.name}]
    env.world.setOrder(my_turn_order)

    x1, y1 = env.get_location_features(agent)
    x2, y2 = env.get_location_features(agent2)
    print(env.world.symbolList)
    for i in range(300):
        print(i)
        # print(env.world.state.items())
        prob = env.world.step()
        print(f'probability: {prob}')
        print(f'agent action: {world.getAction(agent.name)}')
        print(f'agent2 action: {world.getAction(agent2.name)}')
        p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
        p_state = env.p_state[p_feat]
        print(f'agent state: '
              f'x={world.getFeature(x1, unique=True)}, '
              f'y={world.getFeature(y1, unique=True)}, '
              f'r={agent.reward(env.world.state)}')
        print(f'agent2 state: '
              f'x={world.getFeature(x2, unique=True)}, '
              f'y={world.getFeature(y2, unique=True)}, '
              f'r={agent2.reward(env.world.state)}')
        print('Locations:', env.exist_locations, 'Properties:', f'p={p_state}')

        if list(p_state.values()) == [PROPERTY_LIST[-1]] * env.num_exist:
            p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
            p_state = env.p_state[p_feat]
            print(p_feat, p_state)
            print(env.property_goal_count)
            print('Reward:', agent2.reward(env.world.state))
            break
    # env.vis_agent_dynamics_in_xy()
