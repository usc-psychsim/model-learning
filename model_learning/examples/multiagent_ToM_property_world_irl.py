import logging
import os
import pickle
import bz2
import numpy as np
import copy
import random
import itertools as it
from timeit import default_timer as timer
from typing import List, Dict, Any, Optional, Literal, Tuple, Callable
from psychsim.world import World
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import stateKey, WORLD, modelKey, turnKey
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, ModelLearningAlgorithm, ModelLearningResult
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.features import expected_feature_counts
from model_learning.inference import add_agent_models
from model_learning.util.logging import change_log_handler
from model_learning.util.io import create_clear_dir
from model_learning.util.mp import run_parallel
from model_learning import StateActionPair, TeamTrajectory, Trajectory, TeamStateActionModelTuple, \
    TeamStateinfoActionModelTuple, StateinfoActionModelTuple, State
from model_learning.trajectory import copy_world
from model_learning.features.linear import LinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Performs Multiagent IRL (reward model learning) with ToM in the Property World using MaxEnt IRL.'

# env params
GOAL_FEATURE = 'g'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 3

# expert params
TEAM_AGENTS = ['Goal', 'Navigator']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]
MODEL_ROLES = ['Self', 'Uniform']  # TODO, 'Random']
OBSERVER_NAME = 'observer'

EXPERT_RATIONALITY = 1 / 0.1  # inverse temperature
EXPERT_ACT_SELECTION = 'random'
EXPERT_SEED = 17
NUM_TRAJECTORIES = 16  # 3
TRAJ_LENGTH = 25
DISCOUNT = 0.7

MODEL_RATIONALITY = 10  # .5
MODEL_SELECTION = 'distribution'  # 'random'  # 'distribution'

# learning params
NORM_THETA = True
TEAM_LEARNING_RATE = [5e-2, 1e-1]  # 0.05
MAX_EPOCHS = 100
THRESHOLD = 5e-3
DECREASE_RATE = True
EXACT = False
NUM_MC_TRAJECTORIES = 16  # 10
LEARNING_SEED = 17

# common params
HORIZON = 2
PRUNE_THRESHOLD = 1e-2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/multiagent-ToM-property-world')
PROCESSES = -1
VERBOSE = True
np.set_printoptions(precision=3)


def multi_agent_reward_learning(alg: MaxEntRewardLearning,
                                agent_trajs: List[List[StateActionPair]],
                                verbose: bool) -> ModelLearningResult:
    result = alg.learn(agent_trajs, verbose=verbose)
    return result


def generate_trajectory_with_inference(learner_agent: Agent,
                                       reward_vector: LinearRewardVector,
                                       reward_weights: List,
                                       team_trajs: List[TeamTrajectory],
                                       traj_i: int,
                                       learner_model: Optional[str] = None,
                                       select: Optional[bool] = None,
                                       horizon: Optional[int] = None,
                                       selection: Optional[
                                           Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                       threshold: Optional[float] = None,
                                       seed: int = 0,
                                       verbose: bool = False
                                       ) -> Trajectory:
    random.seed(seed)
    n_step = len(team_trajs[traj_i])
    init_state = team_trajs[traj_i][0].state

    trajectory: Trajectory = []
    # reset to initial state and uniform dist of models
    _world = copy_world(learner_agent.world)
    init_state = copy.deepcopy(init_state)
    del init_state[modelKey('observer')]
    _world.state = init_state

    team_agents = ['Goal', 'Navigator']  # TODO
    _team = [_world.agents[agent_name] for agent_name in team_agents]  # get new world's agents
    learner_agent = _team[team_agents.index(learner_agent.name)]
    if learner_model is not None:
        _world.setFeature(modelKey(learner_agent.name), learner_model)

    model_roles = ['Self', 'Uniform']  # TODO
    # add agent models
    for ag_i, _agent in enumerate(_team):
        if _agent.name != learner_agent.name:
            add_agent_models(_agent, reward_vector, reward_weights, model_roles)
            true_model = _agent.get_true_model()
            model_names = [name for name in _agent.models.keys() if name != true_model]
            dist = Distribution({model: 1. / (len(_agent.models) - 1) for model in model_names})
            _world.setMentalModel(learner_agent.name, _agent.name, dist)

            for model in model_names:
                _agent.setAttribute('rationality', 10, model=model)
                _agent.setAttribute('selection', selection, model=model)  # also set selection to distribution
                _agent.setAttribute('horizon', horizon, model=model)
                _agent.setAttribute('discount', 0.7, model=model)
                # _agent.setAttribute('beliefs', False, model=model)

    # learner_agent.set_observations()

    _world.dependency.getEvaluation()

    total = 0
    prob = 1
    for step_i in range(5):
        # print('====================')
        # print(f'Step {step_i}:')
        start = timer()

        # step the world until it's this agent's turn
        turn = _world.getFeature(turnKey(learner_agent.name), unique=True)
        while turn != 0:
            _world.step()
            turn = _world.getFeature(turnKey(learner_agent.name), unique=True)

        prev_world = copy_world(_world)
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= _world.state.select()

        # steps the world (do not select), gets the agent's action
        _world.step(select=False, horizon=horizon, tiebreak=selection,
                    threshold=threshold)  # TODO remove updateBeliefs
        action = _world.getAction(_team[0].name)

        for ag_i, _agent in enumerate(_team):
            if _agent.name != learner_agent.name:
                true_model = _agent.get_true_model()
                model_names = [name for name in _agent.models.keys() if name != true_model]

                dist = Distribution({model: team_trajs[traj_i][step_i].model_dist[model] for model in model_names})
                learner_agent.setBelief(modelKey(_agent.name), dist)
                # model = _world.getModel(learner_agent.name, unique=True)
                # print(_world.getFeature(modelKey(_agent.name), state=learner_agent.getBelief(model=model)))

        # print(_world.state)
        # print(team_trajs[traj_i][step_i].action)
        # print(action)
        # print(_world.getAction(_team[1].name))
        trajectory.append(StateinfoActionModelTuple(prev_world.state,
                                                    action,
                                                    team_trajs[traj_i][step_i].model_dist,
                                                    prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {step_i} took {step_time:.2f}')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_trajectories_with_inference(learner_agent: Agent,
                                         reward_vector: LinearRewardVector,
                                         reward_weights: List,
                                         team_trajs: List[TeamTrajectory],
                                         traj_i: int,
                                         n_trajectories: int,
                                         learner_model: Optional[str] = None,
                                         select: Optional[bool] = None,
                                         horizon: Optional[int] = None,
                                         selection: Optional[
                                             Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                         threshold: Optional[float] = None,
                                         processes: int = -1,
                                         seed: int = 0,
                                         verbose: bool = False,
                                         use_tqdm: bool = True
                                         ) -> List[Trajectory]:
    start = timer()
    args = [(learner_agent, reward_vector, reward_weights, team_trajs, traj_i, learner_model, select,
             horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[Trajectory] = run_parallel(generate_trajectory_with_inference, args, processes=processes,
                                                  use_tqdm=False)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories: '
                     f'{timer() - start:.3f}s')

    return trajectories


def estimate_feature_counts_with_inference(learner_agent: Agent,
                                           reward_vector: LinearRewardVector,
                                           reward_weights: List,
                                           team_trajs: List[Trajectory],
                                           n_trajectories: int,
                                           feature_func: Callable[[State], np.ndarray],
                                           learner_model: Optional[str] = None,
                                           select: Optional[bool] = None,
                                           horizon: Optional[int] = None,
                                           selection: Optional[
                                               Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                           threshold: Optional[float] = None,
                                           processes: int = -1,
                                           seed: int = 0,
                                           verbose: bool = False,
                                           use_tqdm: bool = True) -> np.ndarray:
    args = []
    for traj_i, team_traj in enumerate(team_trajs):
        args.append((learner_agent, reward_vector, reward_weights, team_trajs, traj_i, n_trajectories, learner_model,
                     select, horizon, selection, threshold, processes, seed + traj_i, verbose))
    trajectories = run_parallel(generate_trajectories_with_inference, args, processes=processes, use_tqdm=use_tqdm)
    trajectories = list(it.chain(*trajectories))
    # print(trajectories)
    return expected_feature_counts(trajectories, feature_func)


if __name__ == '__main__':
    learner_ag_i = 0
    print(learner_ag_i)
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, f'mairl_tom_learner{learner_ag_i + 1}.log'), logging.INFO)

    # create world and property environment
    world = World()
    world.setParallel()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')
    print('Output:', OUTPUT_DIR)

    # team of two agents
    team = []
    for ag_i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[ag_i])
        # define agent dynamics
        env.add_location_property_dynamics(agent, idle=True)
        team.append(agent)
    # collaboration dynamics
    env.add_collaboration_dynamics([agent for agent in team])

    team_rwd = []
    team_rwd_w = []
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
        agent_lrv = LinearRewardVector(rwd_features)
        rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
        agent_lrv.set_rewards(agent, rwd_f_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, rwd_f_weights)
        team_rwd.append(agent_lrv)
        team_rwd_w.append(rwd_f_weights)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', EXPERT_ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', EXPERT_RATIONALITY)
        agent.setAttribute('discount', DISCOUNT)
        # agent.setAttribute('beliefs', True)

    world.setOrder([{agent.name for agent in team}])
    world.dependency.getEvaluation()

    # generate trajectories using expert's reward and rationality
    logging.info('=================================')
    logging.info('Retrieving expert trajectories with model distributions...')
    traj_dir = os.path.join(os.path.dirname(__file__), 'output/examples/reward-model-multiagent')
    FOLDER_NAME = f'team_trajs_md_{NUM_TRAJECTORIES}x{TRAJ_LENGTH}_{MODEL_RATIONALITY}'

    f = bz2.BZ2File(
        os.path.join(traj_dir, f'team_trajs_md_{NUM_TRAJECTORIES}x{TRAJ_LENGTH}_{MODEL_RATIONALITY}_base.pkl'), 'rb')
    team_trajectories = pickle.load(f)
    # print(team_trajectories)

    # generate_trajectory_with_inference(team[0], env, team_trajectories, 0)
    rwd_vector = team_rwd[1-learner_ag_i]
    rwd_weights = team_rwd_w[1-learner_ag_i]
    # trajectories_mc = generate_trajectories_with_inference(team[learner_ag_i], rwd_vector, rwd_weights,
    #                                                        team_trajectories, 0,
    #                                                        NUM_MC_TRAJECTORIES, None,
    #                                                        True, HORIZON, 'distribution', PRUNE_THRESHOLD, PROCESSES,
    #                                                        seed=ENV_SEED, verbose=True, use_tqdm=True)

    #
    feature_func = lambda s: team_rwd[learner_ag_i].get_values(s)
    estimate_feature_counts_with_inference(team[learner_ag_i], rwd_vector, rwd_weights, team_trajectories,
                                           NUM_MC_TRAJECTORIES, feature_func, None,
                                           True, HORIZON, 'distribution', PRUNE_THRESHOLD, PROCESSES,
                                           seed=LEARNING_SEED, verbose=False, use_tqdm=True)
    #
    bb
    LEARNING_RATE = TEAM_LEARNING_RATE[learner_ag_i]
    learner_agent = team[learner_ag_i]
    rwd_vector = team_rwd[1-learner_ag_i]
    rwd_weights = team_rwd_w[1-learner_ag_i]
    alg = MaxEntRewardLearning(
        'max-ent', learner_agent.name, rwd_vector,
        processes=PROCESSES,
        normalize_weights=NORM_THETA,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        diff_threshold=THRESHOLD,
        decrease_rate=DECREASE_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        exact=EXACT,
        num_mc_trajectories=NUM_MC_TRAJECTORIES,
        horizon=HORIZON,
        seed=LEARNING_SEED)
    result = alg.learn_with_inference(learner_agent, rwd_weights, team_trajectories, verbose=True)
    #
    # vb
    # # for loop, MaxEnt for each agent
    # logging.info('=================================')
    # logging.info('Starting MaxEnt IRL optimization...')
    #
    # team_trajs = []
    # team_algs = []
    # for ag_i, agent in enumerate(team):
    #     if ag_i == learner_ag_i:
    #         rwd_vector = team_rwd[ag_i]
    #         agent_trajs = []
    #         for team_traj in team_trajectories:
    #             agent_traj = []
    #             for team_step in team_traj:
    #                 tsa = team_step
    #                 sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
    #                 agent_traj.append(sa)
    #             agent_trajs.append(agent_traj)
    #         team_trajs.append(agent_trajs)
    #
    #         LEARNING_RATE = TEAM_LEARNING_RATE[learner_ag_i]
    #         alg = MaxEntRewardLearning(
    #             'max-ent', agent.name, rwd_vector,
    #             processes=PROCESSES,
    #             normalize_weights=NORM_THETA,
    #             learning_rate=LEARNING_RATE,
    #             max_epochs=MAX_EPOCHS,
    #             diff_threshold=THRESHOLD,
    #             decrease_rate=DECREASE_RATE,
    #             prune_threshold=PRUNE_THRESHOLD,
    #             exact=EXACT,
    #             num_mc_trajectories=NUM_MC_TRAJECTORIES,
    #             horizon=HORIZON,
    #             seed=LEARNING_SEED)
    #         result = alg.learn(agent_trajs, verbose=True)
    #
