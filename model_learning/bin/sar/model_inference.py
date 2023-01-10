import bz2
import logging
import numpy as np
import os
import pickle
import string
import time
from typing import List, Optional, Union

from model_learning import TeamTrajectory, TeamStateinfoActionModelTuple, TeamInfoModelTrajectory
from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld
from model_learning.features.linear import LinearRewardVector
from model_learning.trajectory import copy_world
from model_learning.util.io import create_clear_dir
from model_learning.util.mp import run_parallel
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey
from psychsim.world import World

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'
__description__ = 'Performs reward model inference in the search-and-rescue domain.' \
                  'First, it loads a set of team trajectories corresponding to observed behavior between two ' \
                  'collaborative agents. ' \
                  'An observer agent then maintains beliefs (a probability distribution over reward models) about ' \
                  'each agent, which are updated via PsychSim inference according to the agents\' actions  .' \
                  'Trajectories with model inference are saved into a .pkl file.'

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

TEAM_AGENTS = ['Medic', 'Explorer']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]
MODEL_ROLES = ['GroundTruth', 'Opposite', 'Random']
# MODEL_ROLES = ['Random', 'Task', 'Social']
# MODEL_ROLES = ['GroundTruth', 'Task']
OBSERVER_NAME = 'observer'

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params

NUM_TRAJECTORIES = 16
TRAJ_LENGTH = 25
DISCOUNT = 0.7
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=3)

RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = 1  # .5
MODEL_SELECTION = 'distribution'  # 'random'  # 'distribution'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/reward-model-multiagent')
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, f'team_trajs_{len(MODEL_ROLES)}models_{NUM_TRAJECTORIES}x{TRAJ_LENGTH}_{MODEL_RATIONALITY}.pkl')
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _get_fancy_name(name):
    return name.title().replace('_', ' ')


def _get_belief(world: World, feature: str, ag: Agent, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(ag.name, unique=True)
    return world.getFeature(feature, state=ag.getBelief(model=model))


def generate_trajectory_model_distribution(world: World,
                                           env: SearchRescueGridWorld,
                                           team: List[Agent],
                                           team_trajs: Union[List[TeamTrajectory],
                                                             List[TeamInfoModelTrajectory]],
                                           traj_i: int) -> TeamInfoModelTrajectory:
    n_step = len(team_trajs[traj_i])
    if hasattr(team_trajs[traj_i][0], 'world'):
        init_state = team_trajs[traj_i][0].world.state
    else:
        init_state = team_trajs[traj_i][0].state

    team_trajectory = []
    # reset to initial state and uniform dist of models
    _world = copy_world(world)
    _world.state = init_state
    observer = _world.addAgent(OBSERVER_NAME)  # create observer
    _team = [_world.agents[agent.name] for agent in team]  # get new world's agents

    # add agent models
    for ag_i, agent in enumerate(_team):
        env.add_agent_models(agent, AGENT_ROLES[ag_i], MODEL_ROLES)
        true_model = agent.get_true_model()
        model_names = [name for name in agent.models.keys() if name != true_model]
        dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
        _world.setMentalModel(observer.name, agent.name, dist)

        # ignore observer
        agent.ignore(observer.name)
        for model in model_names:
            agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
            agent.setAttribute('selection', MODEL_SELECTION, model=model)  # also set selection to distribution
            agent.setAttribute('horizon', HORIZON, model=model)
            agent.setAttribute('discount', DISCOUNT, model=model)
            agent.setAttribute('beliefs', True, model=model)
            agent.ignore(observer.name, model=model)

    # observer does not observe agents' true models
    observer.set_observations()

    _world.dependency.getEvaluation()

    team_models = {f'{agent.name}_{model_name}': 0 for model_name in MODEL_ROLES for agent in _team}
    model_dist = Distribution(team_models)
    print('====================')
    print(f'Step 0:')
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
    print(f'Initial Belief about Agent models:\n{model_dist}')

    for step_i in range(n_step):
        team_action = team_trajs[traj_i][step_i].action
        prob = team_trajs[traj_i][step_i].prob
        team_action = {agent.name: team_action[agent.name].first() for agent in _team}
        team_trajectory.append(TeamStateinfoActionModelTuple(copy_world(_world).state, team_action, model_dist, prob))

        if step_i == n_step - 1:
            break
        _world.step(actions=team_action)

        model_dist = Distribution(team_models)
        print('====================')
        print(f'Step {step_i + 1}:')
        [print(a) for a in team_action.values()]
        for ag_i, agent in enumerate(_team):
            if ag_i == 1:
                dh = _world.getFeature(env.get_dist_to_help_feature(agent, key=True), unique=True)
                print(f'{agent.name} dist2help: {dh}')
            agent_model = modelKey(agent.name)
            agent_dist = _get_belief(_world, agent_model, observer)
            for agent_model, model_prob in agent_dist.items():
                agent_model_name = agent_model.rstrip(string.digits)
                model_dist[agent_model_name] += model_prob
        print(f'Belief about Agent models:\n{model_dist}')
    # print(team_trajectory)
    return team_trajectory


def _generate_trajectories_model_distribution(world: World,
                                              env: SearchRescueGridWorld,
                                              team: List[Agent],
                                              team_trajectories: Union[List[TeamTrajectory],
                                                                       List[TeamInfoModelTrajectory]],
                                              processes: Optional[int] = -1) -> List[TeamInfoModelTrajectory]:
    args = [(world, env, team, team_trajectories, t) for t in range(len(team_trajectories))]
    team_trajectories_with_model_dist: List[TeamInfoModelTrajectory] = \
        run_parallel(generate_trajectory_model_distribution, args,
                     processes=processes)
    return team_trajectories_with_model_dist


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world
    world = World()
    world.setParallel()
    env = SearchRescueGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_victims}')

    # team of two agents
    team = []
    for ag_i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[ag_i])
        # define agent dynamics
        env.add_search_and_rescue_dynamics(agent, noop_action=True)
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
        agent.setAttribute('discount', DISCOUNT)

    world.setOrder([{agent.name for agent in team}])
    world.dependency.getEvaluation()

    print(OUTPUT_FILE)
    team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                       horizon=HORIZON, selection=ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)

    # example of using saved trajectories
    # FOLDER_NAME = f'GT_team_trajs_gtoprd_{NUM_TRAJECTORIES}x{TRAJ_LENGTH}_{MODEL_RATIONALITY}'
    # f = bz2.BZ2File(os.path.join(os.path.join(os.path.dirname(__file__), 'output/examples/property-world'),
    #                              f'{FOLDER_NAME}.pkl'), 'rb')
    # team_trajectories = pickle.load(f)
    # f.close()

    # print(team_trajectories)
    team_trajs_w_model_dist = _generate_trajectories_model_distribution(world, env, team, team_trajectories,
                                                                        processes=PROCESSES)
    start_time = time.time()
    f = bz2.BZ2File(OUTPUT_FILE, 'wb')
    pickle.dump(team_trajs_w_model_dist, f)
    f.close()
    print(f'Loading Time (s): {(time.time() - start_time):.3f}')