import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple, Literal
from model_learning.environments.property_gridworld import PropertyGridWorld
from psychsim.world import World
from psychsim.reward import maximizeFeature
from psychsim.pwl import makeTree, equalRow, incrementMatrix, stateKey, WORLD
from psychsim.agent import Agent

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'medic', 'ready', 'clear']
WORLD_NAME = 'PGWorld'
GOAL_FEATURE = 'g'

ENV_SIZE = 2
ENV_SEED = 45
NUM_EXIST = 1

AGENT_NAME = 'AHA'
HORIZON = 1
ACT_SELECTION = 'random'

if __name__ == '__main__':
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE-1, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
    p_state = env.p_state[p_feat]
    print(p_state)

    # add agents
    agent = world.addAgent('AHA')
    agent2 = world.addAgent('Helper')
    # define agent dynamics
    env.add_location_property_dynamics(agent)
    env.add_location_property_dynamics(agent2)
    env.add_collaboration_dynamics([agent, agent2])
    env.vis_agent_dynamics_in_xy()

    # print(env.p_state)
    # print(f"{agent.name} actions")
    # print(env.agent_actions[agent.name])
    # print(f"{agent2.name} actions")
    # print(env.agent_actions[agent2.name])

    agent.setReward(maximizeFeature(env.get_agent_goal_feature(agent), agent), 0.)
    agent2.setReward(maximizeFeature(env.get_agent_goal_feature(agent2), agent2), 0.)
    # env.set_achieve_property_reward(agent, 10)
    print(agent.getReward(agent.get_true_model()))

    agent.setAttribute('selection', ACT_SELECTION)
    agent2.setAttribute('selection', ACT_SELECTION)
    agent.setAttribute('horizon', HORIZON)
    agent2.setAttribute('horizon', HORIZON)
    my_turn_order = [{agent.name, agent2.name}]
    env.world.setOrder(my_turn_order)

    x1, y1 = env.get_location_features(agent)
    x2, y2 = env.get_location_features(agent2)
    print(env.world.symbolList)
    for i in range(500):
        print(i)
        prob = env.world.step()
        print(f'probability: {prob}')
        print(f'agent action: {world.getAction(agent.name)}')
        print(f'agent2 action: {world.getAction(agent2.name)}')
        p_feat = env.world.getFeature(stateKey(WORLD, PROPERTY_FEATURE), unique=True)
        p_state = env.p_state[p_feat]
        print(f'agent state: '
              f'x={world.getFeature(x1, unique=True)}, '
              f'y={world.getFeature(y1, unique=True)}')
        print(f'agent2 state: '
              f'x={world.getFeature(x2, unique=True)}, '
              f'y={world.getFeature(y2, unique=True)}')
        print(env.exist_locations, f'p={p_state}')
        if list(p_state.values()) == [PROPERTY_LIST[-1]]*env.num_exist:
            break
    # env.vis_agent_dynamics_in_xy()
