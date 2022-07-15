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
    makeTree, equalRow, incrementMatrix, stateKey, WORLD, MODEL, ACTION, rewardKey, KeyedPlane, KeyedVector
from psychsim.reward import maximizeFeature
from model_learning.util.io import create_clear_dir
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = ''

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'
CLEARINDICATOR_FEATURE = 'ci'
MARK_FEATURE = 'm'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 3

TEAM_AGENTS = ['Goal', 'Navigator']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]
MODEL_ROLES = ['Self', 'Uniform', 'Random']
OBSERVER_NAME = 'observer'

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params

NUM_TRAJECTORIES = 1  # 10
TRAJ_LENGTH = 5  # 30
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=3)

NUM_STEPS = TRAJ_LENGTH
RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = .5

# MODEL_SELECTION = 'distribution'  # TODO 'consistent' or 'random' gives an error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/reward-model-multiagent')
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _get_fancy_name(name):
    return name.title().replace('_', ' ')


from typing import List, Dict, Tuple, Any
from model_learning import StateActionPair, TeamTrajectory, TeamStateActionPair, Trajectory
from model_learning.trajectory import copy_world
import copy
from psychsim.agent import Agent
from psychsim.world import VectorDistributionSet


def _get_belief(world: World, feature: str, ag: Agent, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(ag.name, unique=True)
    return world.getFeature(feature, state=ag.getBelief(model=model))


def agent_trajectories_in_team(team_trajs: List[TeamTrajectory]) -> Tuple[Dict[str, List[Trajectory]], int, int]:
    assert len(team_trajs) > 0
    assert len(team_trajs[0]) > 0
    assert len(team_trajs[0][0].action.keys()) > 0
    agent_names = list(team_trajs[0][0].action.keys())
    all_agent_trajs = {agent_name: [] for agent_name in agent_names}
    for team_traj in team_trajectories:
        agent_traj = {agent_name: [] for agent_name in agent_names}
        for team_step in team_traj:
            tsa = team_step
            for agent_name in agent_names:
                sa = StateActionPair(tsa.world, tsa.action[agent_name], tsa.prob)
                agent_traj[agent_name].append(sa)
        for agent_name in agent_names:
            all_agent_trajs[agent_name].append(agent_traj[agent_name])
    return all_agent_trajs, len(team_trajs), len(team_trajs[0])

'''
def set_observer_agent_models(env: PropertyGridWorld,
                              team: List[AgentRoles],
                              reward_models: List[str],
                              init_state: VectorDistributionSet):
    _world = copy_world(env.world)
    _world.state = init_state
    # observer = _world.addAgent(OBSERVER_NAME)
    _team = []
    print(_world.agents, team[0])
    for ag_i, agent in enumerate(team):
        agent = _world.agents[team[ag_i].name]
        true_model = agent.get_true_model()
        for model_name in reward_models:
            print(agent.name, model_name)
            agent.addModel(model_name, parent=true_model, rationality=MODEL_RATIONALITY)
            rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
            agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights, model=model_name)
            if model_name == 'Uniform':
                agent_lrv.rwd_weights = [1] * len(agent_lrv.rwd_weights)
                agent_lrv.rwd_weights = np.array(agent_lrv.rwd_weights) / np.linalg.norm(agent_lrv.rwd_weights, 1)
            if model_name == 'Random':
                agent_lrv.rwd_weights = [0] * len(agent_lrv.rwd_weights)

            agent_lrv.set_rewards(agent, agent_lrv.rwd_weights, model=model_name)
            print(agent_lrv.names, agent_lrv.rwd_weights)

        model_names = [name for name in agent.models.keys() if name != true_model]
        dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
        _world.setMentalModel(observer.name, agent.name, dist)

        agent.ignore(observer.name)
        for model in model_names:
            agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
            agent.setAttribute('selection', 'distribution', model=model)  # also set selection to distribution
            agent.setAttribute('beliefs', True, model=model)
            agent.ignore(observer.name, model=model)
        _team.append(agent)
    observer.set_observations()
    return _world, _team, observer
'''


def generate_model_distribution(env: PropertyGridWorld,
                                team: List[AgentRoles],
                                observer: Agent,
                                team_trajectories: List[TeamTrajectory]
                                ) -> List[TeamTrajectory]:
    # add agent models
    for ag_i, agent in enumerate(team):
        true_model = agent.get_true_model()
        for model_name in MODEL_ROLES:
            agent.addModel(model_name, parent=true_model)
            rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
            agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights, model=model_name)
            if model_name == 'Uniform':
                agent_lrv.rwd_weights = [1] * len(agent_lrv.rwd_weights)
                agent_lrv.rwd_weights = np.array(agent_lrv.rwd_weights) / np.linalg.norm(agent_lrv.rwd_weights, 1)
            if model_name == 'Random':
                agent_lrv.rwd_weights = [0] * len(agent_lrv.rwd_weights)

            agent_lrv.set_rewards(agent, agent_lrv.rwd_weights, model=model_name)
            print(agent.name, model_name, agent_lrv.names, agent_lrv.rwd_weights)

        model_names = [name for name in agent.models.keys() if name != true_model]
        # dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
        # env.world.setMentalModel(observer.name, agent.name, dist)

        agent.ignore(observer.name)
        for model in model_names:
            agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
            agent.setAttribute('selection', 'distribution', model=model)  # also set selection to distribution
            agent.setAttribute('beliefs', True, model=model)
            agent.ignore(observer.name, model=model)

    n_traj = len(team_trajectories)
    n_step = len(team_trajectories[0])
    initial_states = [t[0].world.state for t in team_trajectories]
    for traj_i in range(n_traj):
        # reset to initial state and uniform dist of models
        world = copy_world(env.world)
        world.state = initial_states[traj_i]
        for ag_i, agent in enumerate(team):
            agent = world.agents[agent.name]
            true_model = agent.get_true_model()
            model_names = [name for name in agent.models.keys() if name != true_model]
            print(model_names)
            dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
            world.setMentalModel(observer.name, agent.name, dist)

            agent_model = modelKey(agent.name)
            x, y = env.get_location_features(agent)
            init_x = world.getFeature(x, state=initial_states[traj_i], unique=True)
            init_y = world.getFeature(y, state=initial_states[traj_i], unique=True)
            print(f'Initial loc: x:{init_x}, y:{init_y}')
            print(f'Initial belief about Agent {agent.name} model:\n{_get_belief(world, agent_model, observer)}')

        observer.set_observations()

        for step_i in range(n_step):
            team_action = team_trajectories[traj_i][step_i].action
            team_action = {agent.name: team_action[agent.name].first() for agent in team}
            env.world.step(state=world.state, actions=team_action)


            for ag_i, agent in enumerate(team):
                agent_model = modelKey(agent.name)
                print(f'Belief about Agent {agent.name} model:\n{_get_belief(world, agent_model, observer)}')

    return None


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world, agent and observer
    world = World()
    observer = world.addAgent(OBSERVER_NAME)
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
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', RATIONALITY)
        agent.setAttribute('discount', 0.7)

    world.setOrder([{agent.name for agent in team}])
    world.dependency.getEvaluation()

    # for ag_i, agent in enumerate(team):
    #     agent.ignore(observer.name)

    team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                       horizon=HORIZON, selection=ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)
    print(team_trajectories)
    model_dist = generate_model_distribution(env, team, observer, team_trajectories)
