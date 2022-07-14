import os
import logging
import numpy as np
from model_learning.inference import track_reward_model_inference
from model_learning.util.plot import plot_evolution
from model_learning import StateActionPair
from psychsim.probability import Distribution
from psychsim.world import World
from model_learning.environments.property_gridworld import PropertyGridWorld
from psychsim.pwl import modelKey, makeTree, setToConstantMatrix, rewardKey, actionKey, \
    makeTree, equalRow, incrementMatrix, stateKey, WORLD, rewardKey, KeyedPlane, KeyedVector
from psychsim.reward import maximizeFeature
from model_learning.util.io import create_clear_dir
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = ''

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'
CLEARINDICATOR_FEATURE = 'ci'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 2
ENV_SEED = 48
NUM_EXIST = 2

TEAM_AGENTS = ['AHA', 'Helper1']
# AGENT_ROLES = [{'Goal': 1}, {'SubGoal': 0.1, 'Navigator': 0.1}]
# AGENT_ROLES = [{'Goal': 1, 'SubGoal': 0.2}, {'Goal': 0.2, 'SubGoal': 0.2, 'Navigator': 1}]
AGENT_ROLES = [{'Goal': 1}, {'Goal': 0.2, 'Navigator': 0.2}]

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1

# common params

NUM_TRAJECTORIES = 3  # 10
TRAJ_LENGTH = 20  # 30
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=3)

NUM_STEPS = 100
RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = .5

MODEL_SELECTION = 'distribution'  # TODO 'consistent' or 'random' gives an error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/reward-model-multiagent')
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _get_fancy_name(name):
    return name.title().replace('_', ' ')


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world, agent and observer
    world = World()
    world.setParallel()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')

    # team of two agents
    team = []
    for i in range(len(TEAM_AGENTS)):
        team.append(world.addAgent(AgentRoles(TEAM_AGENTS[i], AGENT_ROLES[i])))

    # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=True)
    env.add_collaboration_dynamics([agent for agent in team])

    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        agent_lrv.rwd_weights = np.array(agent_lrv.rwd_weights) / np.linalg.norm(agent_lrv.rwd_weights, 1)
        agent_lrv.set_rewards(agent, agent_lrv.rwd_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, agent_lrv.rwd_weights)

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', RATIONALITY)

    world.setOrder([{agent.name for agent in team}])
    world.dependency.getEvaluation()

    # get the canonical name of the "true" agent model
    for ag_i, agent in enumerate(team):
        true_model = agent.get_true_model()
        agent.addModel(RANDOM_MODEL, parent=true_model, rationality=MODEL_RATIONALITY, selection=MODEL_SELECTION)
        agent.setReward(makeTree(setToConstantMatrix(rewardKey(agent.name), 0)), model=RANDOM_MODEL)

    for ag_i, agent in enumerate(team):
        true_model = agent.get_true_model()
        model_names = [name for name in agent.models.keys() if name != true_model]

        _other_agent = team[ag_i - 1]
        # observer has uniform prior distribution over possible agent models
        world.setMentalModel(_other_agent.name, agent.name,
                             Distribution({name: 1. / (len(agent.models) - 1) for name in model_names}))
        for model in model_names:
            _other_agent.setAttribute('beliefs', True, model=model)
            # agent.ignore(_other_agent.name, model=model)
        _other_agent.set_observations()

    # generates trajectory
    logging.info('Generating trajectory of length {}...'.format(NUM_STEPS))

    team_trajectories = env.generate_team_trajectories(team, NUM_STEPS, n_trajectories=1,
                                                       horizon=HORIZON, selection=ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)

    team_trajs = [] * len(team)
    for ag_i, agent in enumerate(team):
        if ag_i in {0, 1}:
            agent_trajs = []
            for team_traj in team_trajectories:
                agent_traj = []
                for team_step in team_traj:
                    tsa = team_step
                    sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                    agent_traj.append(sa)
                agent_trajs.append(agent_traj)
            team_trajs.append(agent_trajs)

    for ag_i, agent in enumerate(team):
        _other_agent = team[ag_i - 1]
        trajectory = team_trajs[ag_i]
        true_model = agent.get_true_model()
        model_names = [name for name in agent.models.keys() if name != true_model]
        # gets evolution of inference over reward models of the agent
        probs = track_reward_model_inference(trajectory, model_names, _otheragent, agent)

        # create and save inference evolution plot
        plot_evolution(probs.T, [_get_fancy_name(name) for name in model_names],
                       'Evolution of Model Inference', None,
                       os.path.join(OUTPUT_DIR, f'inference{ag_i+1}.png'), 'Time', 'Model Probability', True)
