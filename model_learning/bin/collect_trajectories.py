import os
import numpy as np
from model_learning import get_policy

from psychsim.agent import Agent
from psychsim.helper_functions import get_true_model_name
from psychsim.reward import maximizeFeature
from psychsim.world import World
from psychsim.pwl import makeTree, setToConstantMatrix, rewardKey
from model_learning.environments.grid_world import GridWorld
from model_learning.util.io import create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

ENV_SIZE = 10

AGENT_NAME = 'Agent'
HORIZON = 3
SELECTION = 'distribution'  # stochastic over all actions

NUM_TRAJECTORIES = 20
TRAJ_LENGTH = 15

OUTPUT_DIR = 'output/tests'

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world and agent
    world = World()
    agent = Agent(AGENT_NAME)
    world.addAgent(agent)

    # create grid-world and add world dynamics to agent
    env = GridWorld(world, ENV_SIZE, ENV_SIZE)
    env.add_agent_dynamics(agent)
    env.plot(os.path.join(OUTPUT_DIR, 'env.png'))

    # set reward function
    # agent.setReward(makeTree(setToConstantMatrix(rewardKey(agent.name), 0)))
    x, y = env.get_location_features(agent)
    agent.setReward(maximizeFeature(x, agent.name), 1.)
    agent.setReward(maximizeFeature(y, agent.name), 1.)

    world.setOrder([{agent.name}])

    # generate trajectories using agent's policy
    print('Generating trajectories...')
    trajectories = env.generate_trajectories(NUM_TRAJECTORIES, TRAJ_LENGTH, agent, HORIZON, selection=SELECTION)
    env.print_trajectories_cmd_line(trajectories)
    env.plot_trajectories(trajectories, os.path.join(OUTPUT_DIR, 'trajectories.png'))

    # gets policy and value
    states = env.get_all_states(agent)
    pi = np.array([[dist[a] for a in env.agent_actions[agent.name]]
                   for dist in get_policy(agent, states, selection=SELECTION)])
    q = np.array([[agent.value(s, a, get_true_model_name(agent))['__EV__'] for a in env.agent_actions[agent.name]]
                  for s in states])
    v = np.max(q, axis=1)
    env.plot_policy(pi, v, os.path.join(OUTPUT_DIR, 'policy.png'))

    # gets rewards
    r = np.array([agent.reward(state) for state in states])
    env.plot_func(r, os.path.join(OUTPUT_DIR, 'reward.png'), 'Rewards')

    print('\nFinished!')
