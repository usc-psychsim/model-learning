import os
import logging
import numpy as np
import string
import pickle
from typing import List, Dict, Tuple, Optional
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey
from psychsim.world import World
from model_learning import StateActionPair, TeamTrajectory, Trajectory, TeamStateActionModelTuple
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.trajectory import copy_world
from model_learning.util.io import create_clear_dir
from model_learning.features.linear import LinearRewardVector
from model_learning.util.mp import run_parallel
import copy

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

NUM_TRAJECTORIES = 16  # 10
TRAJ_LENGTH = 25  # 30
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=3)

RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = 10  # .5
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
    for team_traj in team_trajs:
        agent_traj = {agent_name: [] for agent_name in agent_names}
        for team_step in team_traj:
            tsa = team_step
            for agent_name in agent_names:
                sa = StateActionPair(tsa.world, tsa.action[agent_name], tsa.prob)
                agent_traj[agent_name].append(sa)
        for agent_name in agent_names:
            all_agent_trajs[agent_name].append(agent_traj[agent_name])
    return all_agent_trajs, len(team_trajs), len(team_trajs[0])


# world, env, team, team_trajectories, n_step, initial_states
def generate_trajectory_model_distribution(world: World,
                                           env: PropertyGridWorld,
                                           team: List[Agent],
                                           team_trajs: List[TeamTrajectory],
                                           traj_i: int) -> TeamTrajectory:
    n_step = len(team_trajs[traj_i])
    init_state = team_trajs[traj_i][0].world.state

    team_trajectory = []
    # reset to initial state and uniform dist of models
    _world = copy_world(world)
    _world.state = init_state
    observer = _world.addAgent(OBSERVER_NAME)  # create observer
    _team = [_world.agents[agent.name] for agent in team]  # get new world's agents

    # add agent models
    for ag_i, agent in enumerate(_team):
        true_model = agent.get_true_model()
        for model_name in MODEL_ROLES:
            model_name = f'{agent.name}_{model_name}'
            agent.addModel(model_name, parent=true_model)
            rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
            agent_lrv = LinearRewardVector(rwd_features)
            rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
            if model_name == f'{agent.name}_Uniform':
                rwd_f_weights = [1] * len(rwd_f_weights)
                rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
            if model_name == f'{agent.name}_Random':
                rwd_f_weights = [0] * len(rwd_f_weights)
            agent_lrv.set_rewards(agent, rwd_f_weights, model=model_name)
            print(agent.name, model_name, agent_lrv.names, rwd_f_weights)

        model_names = [name for name in agent.models.keys() if name != true_model]
        dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
        _world.setMentalModel(observer.name, agent.name, dist)

        # ignore observer
        agent.ignore(observer.name)
        for model in model_names:
            agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
            agent.setAttribute('selection', MODEL_SELECTION, model=model)  # also set selection to distribution
            agent.setAttribute('horizon', HORIZON, model=model)
            agent.setAttribute('discount', 0.7, model=model)
            agent.setAttribute('beliefs', True, model=model)
            agent.ignore(observer.name, model=model)

    # observer does not observe agents' true models
    observer.set_observations()

    _world.dependency.getEvaluation()

    team_models = {f'{agent.name}_{model_name}': 0 for model_name in MODEL_ROLES for agent in _team}
    model_dist = Distribution(team_models)
    for ag_i, agent in enumerate(_team):
        agent_model = modelKey(agent.name)
        x, y = env.get_location_features(agent)
        init_x = _world.getFeature(x, unique=True)
        init_y = _world.getFeature(y, unique=True)
        print(f'{agent.name} initial loc: x:{init_x}, y:{init_y}')
        agent_dist = _get_belief(_world, agent_model, observer)
        for agent_model, model_prob in agent_dist.items():
            agent_model_name = agent_model.rstrip(string.digits)
            model_dist[agent_model_name] += model_prob
    print(f'Belief about Agent models:\n{model_dist}')

    for step_i in range(n_step):
        print('====================')
        print(f'Step {step_i}:')
        team_action = team_trajs[traj_i][step_i].action
        prob = team_trajs[traj_i][step_i].prob
        team_action = {agent.name: team_action[agent.name].first() for agent in _team}
        [print(a) for a in team_action.values()]
        team_trajectory.append(TeamStateActionModelTuple(copy_world(_world), team_action, model_dist, prob))

        if step_i == n_step - 1:
            break
        _world.step(actions=team_action)

        model_dist = Distribution(team_models)
        for ag_i, agent in enumerate(_team):
            agent_model = modelKey(agent.name)
            agent_dist = _get_belief(_world, agent_model, observer)
            for agent_model, model_prob in agent_dist.items():
                agent_model_name = agent_model.rstrip(string.digits)
                model_dist[agent_model_name] += model_prob
        print(f'Belief about Agent models:\n{model_dist}')
    # print(team_trajectory)
    return team_trajectory


def _generate_trajectories_model_distribution(world: World,
                                              env: PropertyGridWorld,
                                              team: List[Agent],
                                              team_trajectories: List[TeamTrajectory],
                                              processes: Optional[int] = -1) -> List[TeamTrajectory]:
    if len(team_trajectories) > 1:
        args = [(world, env, team, team_trajectories, t) for t in range(len(team_trajectories))]
        team_trajectories_with_model_dist: List[TeamTrajectory] = run_parallel(generate_trajectory_model_distribution, args,
                                                                               processes=processes)
        return team_trajectories_with_model_dist
    else:
        # perform inference for each trajectory
        n_traj = len(team_trajectories)
        n_step = len(team_trajectories[0])
        initial_states = [t[0].world.state for t in team_trajectories]

        team_trajectories_with_model_dist = []
        for traj_i in range(n_traj):  # TODO parallelize this
            team_trajectory = []
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
                    rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
                    agent_lrv = LinearRewardVector(rwd_features)
                    rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
                    if model_name == f'{agent.name}_Uniform':
                        rwd_f_weights = [1] * len(rwd_f_weights)
                        rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
                    if model_name == f'{agent.name}_Random':
                        rwd_f_weights = [0] * len(rwd_f_weights)
                    agent_lrv.set_rewards(agent, rwd_f_weights, model=model_name)
                    print(agent.name, model_name, agent_lrv.names, rwd_f_weights)

                model_names = [name for name in agent.models.keys() if name != true_model]
                dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
                _world.setMentalModel(observer.name, agent.name, dist)

                # ignore observer
                agent.ignore(observer.name)
                for model in model_names:
                    agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
                    agent.setAttribute('selection', MODEL_SELECTION, model=model)  # also set selection to distribution
                    agent.setAttribute('horizon', HORIZON, model=model)
                    agent.setAttribute('discount', 0.7, model=model)
                    agent.setAttribute('beliefs', True, model=model)
                    agent.ignore(observer.name, model=model)

            # observer does not observe agents' true models
            observer.set_observations()

            _world.dependency.getEvaluation()

            team_models = {f'{agent.name}_{model_name}': 0 for model_name in MODEL_ROLES for agent in _team}
            model_dist = Distribution(team_models)
            for ag_i, agent in enumerate(_team):
                agent_model = modelKey(agent.name)
                x, y = env.get_location_features(agent)
                init_x = _world.getFeature(x, unique=True)
                init_y = _world.getFeature(y, unique=True)
                print(f'{agent.name} initial loc: x:{init_x}, y:{init_y}')
                agent_dist = _get_belief(_world, agent_model, observer)
                print(f'Belief about Agent {agent.name} model:\n{agent_dist}')
                for agent_model, model_prob in agent_dist.items():
                    agent_model_name = agent_model.rstrip(string.digits)
                    model_dist[agent_model_name] += model_prob
            print(f'Belief about Agent models:\n{model_dist}')

            for step_i in range(n_step):
                print('====================')
                print(f'Step {step_i}:')
                team_action = team_trajectories[traj_i][step_i].action
                prob = team_trajectories[traj_i][step_i].prob
                team_action = {agent.name: team_action[agent.name].first() for agent in _team}
                # print(_world.state)
                [print(a) for a in team_action.values()]
                team_trajectory.append(TeamStateActionModelTuple(copy_world(_world), team_action, model_dist, prob))

                if step_i == n_step - 1:
                    break
                _world.step(actions=team_action)

                model_dist = Distribution(team_models)
                for ag_i, agent in enumerate(_team):
                    agent_model = modelKey(agent.name)
                    agent_dist = _get_belief(_world, agent_model, observer)
                    print(f'Belief about Agent {agent.name} model:\n{agent_dist}')
                    possible_models = list(agent_dist.keys())
                    for pm in possible_models:
                        print(agent.models[pm]['name'])
                        print(agent.models[pm]['beliefs'])
                    for agent_model, model_prob in agent_dist.items():
                        agent_model_name = agent_model.rstrip(string.digits)
                        model_dist[agent_model_name] += model_prob
                print(f'Belief about Agent models:\n{model_dist}')
            print(team_trajectory)
            team_trajectories_with_model_dist.append(team_trajectory)
        return team_trajectories_with_model_dist


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world
    world = World()
    world.setParallel()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')

    # team of two agents
    team = []
    for ag_i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[ag_i])
        # define agent dynamics
        env.add_location_property_dynamics(agent, idle=True)
        team.append(agent)
    # collaboration dynamics
    env.add_collaboration_dynamics([agent for agent in team])

    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
        agent_lrv = LinearRewardVector(rwd_features)
        rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
        agent_lrv.set_rewards(agent, rwd_f_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, rwd_f_weights)
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
    team_trajs_w_model_dist = _generate_trajectories_model_distribution(world, env, team, team_trajectories,
                                                                        processes=PROCESSES)
    f = open(os.path.join(OUTPUT_DIR, f'team_trajs_md_{NUM_TRAJECTORIES}x{TRAJ_LENGTH}.pkl'), 'wb')
    pickle.dump(team_trajs_w_model_dist, f)
    f.close()
