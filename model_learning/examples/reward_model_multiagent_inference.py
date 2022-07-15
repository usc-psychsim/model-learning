import os
import logging
import numpy as np
from typing import List, Dict, Tuple
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey
from psychsim.world import World
from model_learning import StateActionPair, TeamTrajectory, Trajectory
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector
from model_learning.trajectory import copy_world
from model_learning.util.io import create_clear_dir

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
MODEL_ROLES = ['Self', 'Uniform']  # TODO, 'Random']
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

RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = 1  # .5
MODEL_SELECTION = 'distribution'  # 'random'  # 'distribution'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/reward-model-multiagent')
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _get_fancy_name(name):
    return name.title().replace('_', ' ')


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


def generate_model_distribution(world: World,
                                env: PropertyGridWorld,
                                team: List[AgentRoles],
                                team_trajectories: List[TeamTrajectory]) -> List[TeamTrajectory]:
    # perform inference for each trajectory
    n_traj = len(team_trajectories)
    n_step = len(team_trajectories[0])
    initial_states = [t[0].world.state for t in team_trajectories]
    for traj_i in range(n_traj):  # TODO parallelize this

        # reset to initial state and uniform dist of models
        _world = copy_world(world)
        _world.state = initial_states[traj_i]
        observer = _world.addAgent(OBSERVER_NAME)  # create observer
        _team = [_world.agents[agent.name] for agent in team]  # get new world's agents

        # add agent models
        for ag_i, agent in enumerate(_team):
            true_model = agent.get_true_model()
            for model_name in MODEL_ROLES:
                model_name = f'{agent.name}_{model_name}'
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
            dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
            _world.setMentalModel(observer.name, agent.name, dist)

            # ignore observer
            agent.ignore(observer.name)
            for model in model_names:
                agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
                agent.setAttribute('selection', MODEL_SELECTION, model=model)  # also set selection to distribution
                agent.setAttribute('beliefs', True, model=model)
                agent.ignore(observer.name, model=model)

            agent_model = modelKey(agent.name)
            x, y = env.get_location_features(agent)
            init_x = _world.getFeature(x, unique=True)
            init_y = _world.getFeature(y, unique=True)
            print(f'{agent.name} initial loc: x:{init_x}, y:{init_y}')
            print(f'Observer initial belief about {agent.name} model:\n{_get_belief(_world, agent_model, observer)}')

        # observer does not observe agents' true models
        observer.set_observations()

        _world.dependency.getEvaluation()

        for step_i in range(n_step):
            team_action = team_trajectories[traj_i][step_i].action
            team_action = {agent.name: team_action[agent.name].first() for agent in _team}
            _world.step(actions=team_action)

            print('====================')
            print(f'Step {step_i}:')

            for ag_i, agent in enumerate(_team):
                agent_model = modelKey(agent.name)
                print(f'Belief about Agent {agent.name} model:\n{_get_belief(_world, agent_model, observer)}')

    return None


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world
    world = World()
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

    team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                       horizon=HORIZON, selection=ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)
    print(team_trajectories)
    model_dist = generate_model_distribution(world, env, team, team_trajectories)
