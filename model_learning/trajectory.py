import copy
from timeit import default_timer as timer

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional

from model_learning import Trajectory, StateActionPair, TeamTrajectory, TeamStateActionPair, SelectionType, State, \
    TeamModelsDistributions, StateActionProbTrajectory
from model_learning.util.mp import run_parallel
from psychsim.agent import Agent
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, turnKey, actionKey, setToConstantMatrix, makeTree, VectorDistributionSet, \
    isSpecialKey
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

TOP_LEVEL_STR = 'top_level'


def copy_world(world: World) -> World:
    """
    Creates a copy of the given world. This implementation clones the world's state and all agents so that the dynamic
    world elements are "frozen" in time.
    :param World world: the original world to be copied.
    :rtype: World
    :return: a semi-hard copy of the given world.
    """
    new_world = copy.copy(world)
    new_world.state = copy.deepcopy(world.state)
    new_world.symbols = world.symbols.copy()
    new_world.symbolList = world.symbolList.copy()
    new_world.agents = copy.copy(world.agents)
    for name, agent in world.agents.items():
        # clones agent with exception of world
        agent.world = None
        # TODO tentative
        new_agent = copy.copy(agent)
        new_agent.models = copy.deepcopy(agent.models)
        new_agent.modelList = agent.modelList.copy()
        new_world.agents[name] = new_agent
        new_agent.world = new_world  # assigns cloned world to cloned agent
        agent.world = world  # puts original world back to old agent
    return new_world


def update_state(state: VectorDistributionSet, new_state: VectorDistributionSet):
    """
    Updates a state based on the value sof another state (overrides values).
    :param VectorDistributionSet state: the state to be updated.
    :param VectorDistributionSet new_state: the state from which to get the values used to update the first state.
    """
    # TODO the update() method of PsychSim state appears not to be working
    for key in new_state.keys():
        if key in state and not isSpecialKey(key):
            val = new_state.certain[key] if new_state.keyMap[key] is None else new_state[key]
            state.join(key, val)


def generate_trajectory(agent: Agent,
                        trajectory_length: int,
                        init_feats: Optional[Dict[str, Any]] = None,
                        model: Optional[str] = None,
                        select: Optional[bool] = None,
                        horizon: Optional[int] = None,
                        selection: Optional[SelectionType] = None,
                        threshold: Optional[float] = None,
                        seed: int = 0,
                        verbose: bool = False) -> Trajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(agent.world)

    # generate or select initial state features
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)

    random.seed(seed)

    if model is not None:
        world.setFeature(modelKey(agent.name), model)

    # for each step, takes action and registers state-action pairs
    trajectory: Trajectory = []
    total = 0
    prob = 1.
    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        turn = world.getFeature(turnKey(agent.name), unique=True)
        while turn != 0:
            world.step()
            turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_state = copy.deepcopy(world.state)  # keep (possibly stochastic) state and prob before selection
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        # steps the world (do not select), gets the agent's action
        world.step(select=False, horizon=horizon, tiebreak=selection, threshold=threshold)
        action = world.getAction(agent.name)
        trajectory.append(StateActionPair(prev_state, action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}s (action: {action if len(action) > 1 else action.first()})')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_trajectories(agent: Agent,
                          n_trajectories: int,
                          trajectory_length: int,
                          init_feats: Optional[Dict[str, Any]] = None,
                          model: Optional[str] = None,
                          select: Optional[bool] = None,
                          horizon: Optional[int] = None,
                          selection: Optional[SelectionType] = None,
                          threshold: Optional[float] = None,
                          processes: int = -1,
                          seed: int = 0,
                          verbose: bool = False,
                          use_tqdm: bool = True) -> List[Trajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    world = agent.world
    if init_feats is not None:
        for feature in init_feats:
            assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(agent, trajectory_length, init_feats, model, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[Trajectory] = run_parallel(generate_trajectory, args, processes=processes, use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def generate_trajectory_distribution(agent: Agent,
                                     initial_state: Optional[State],
                                     trajectory_length: int,
                                     exact: bool,
                                     num_mc_trajectories: int,
                                     model: Optional[str],
                                     horizon: Optional[int],
                                     threshold: Optional[float],
                                     processes: Optional[int],
                                     seed: int,
                                     verbose: bool) -> List[Trajectory]:
    """
    Generates a distribution over trajectories (paths) by sampling an agent's stochastic policy in the environment
    starting from the given state. It can perform exact computation (via PsychSim planning without selection), or
    sample Monte-Carlo trajectories to approximate the distribution.
    :param Agent agent: the PsychSim agent whose stochastic policy we want to compute the distribution.
    :param State initial_state: the optional initial state from which the agent should depart.
    :param int trajectory_length: the length of the trajectory which distribution we want to compute.
    :param bool exact: whether to perform an exact computation, resulting in a single trajectory being generated
    by setting `select=False` while stepping the Psychsim agent.
    :param int num_mc_trajectories: the number of (deterministic) Monte-Carlo trajectories to sample, if `exact=False`.
    :param str model: the agent's model to use.
    :param int horizon: the agent's planning horizon while producing the trajectories.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list containing the generated agent trajectories.
    """

    # make copy of world and set initial state
    world = copy_world(agent.world)
    _agent = world.agents[agent.name]
    if initial_state is not None:
        update_state(world.state, initial_state)

    if exact:
        # exact computation, generate a single stochastic trajectory (select=False) from initial state
        trajectory_dist = generate_trajectory(_agent, trajectory_length, model=model, select=False,
                                              horizon=horizon, selection='distribution', threshold=threshold,
                                              seed=seed, verbose=verbose)
        return [trajectory_dist]

    # Monte Carlo approximation, generate N deterministic (single-path) trajectories (select=True) from initial state
    trajectories_mc = generate_trajectories(_agent, num_mc_trajectories, trajectory_length,
                                            model=model, select=True,
                                            horizon=horizon, selection='distribution', threshold=threshold,
                                            processes=processes, seed=seed, verbose=verbose,
                                            use_tqdm=False)
    return trajectories_mc


def generate_trajectory_distribution_tom(agent: Agent,
                                         initial_state: Optional[State],
                                         models_dists: List[TeamModelsDistributions],
                                         exact: bool,
                                         num_mc_trajectories: int,
                                         model: Optional[str],
                                         horizon: Optional[int],
                                         threshold: Optional[float],
                                         processes: Optional[int],
                                         seed: int,
                                         verbose: bool) -> List[Trajectory]:
    """
    Generates a distribution over trajectories (paths) by sampling an agent's stochastic policy in the environment
    starting from the given state. It can perform exact computation (via PsychSim planning without selection), or
    sample Monte-Carlo trajectories to approximate the distribution.
    This function is to be applied to multiagent scenarios, where an agent models other agents. A distribution over
    models of others is used during trajectory generation to sample their actions.
    :param Agent agent: the PsychSim agent whose stochastic policy we want to compute the distribution.
    :param State initial_state: the optional initial state from which the agent should depart.
    :param list[TeamModelsDistributions] models_dists: the sequence of distributions over other agents' models. The
    length of this sequence is used to compute the trajectory distribution.
    :param bool exact: whether to perform an exact computation, resulting in a single trajectory being generated
    by setting `select=False` while stepping the Psychsim agent.
    :param int num_mc_trajectories: the number of (deterministic) Monte-Carlo trajectories to sample, if `exact=False`.
    :param str model: the agent's model to use.
    :param int horizon: the agent's planning horizon while producing the trajectories.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list containing the generated agent trajectories.
    """

    # make copy of world and set initial state
    world = copy_world(agent.world)
    agent = world.agents[agent.name]
    if initial_state is not None:
        update_state(world.state, initial_state)

    if exact:
        # exact computation, generate a single stochastic trajectory (select=False) from initial state
        trajectory_dist = generate_trajectory_tom(agent, models_dists, model=model, select=False,
                                                  horizon=horizon, selection='distribution', threshold=threshold,
                                                  seed=seed, verbose=verbose)
        return [trajectory_dist]

    # Monte Carlo approximation, generate N deterministic (single-path) trajectories (select=True) from initial state
    trajectories_mc = generate_trajectories_tom(agent, num_mc_trajectories, models_dists,
                                                model=model, select=True,
                                                horizon=horizon, selection='distribution', threshold=threshold,
                                                processes=processes, seed=seed, verbose=verbose,
                                                use_tqdm=False)
    return trajectories_mc


def generate_trajectory_tom(agent: Agent,
                            models_dists: List[TeamModelsDistributions],
                            init_feats: Optional[Dict[str, Any]] = None,
                            model: Optional[str] = None,
                            select: Optional[bool] = None,
                            horizon: Optional[int] = None,
                            selection: Optional[SelectionType] = None,
                            threshold: Optional[float] = None,
                            seed: int = 0,
                            verbose: bool = False) -> Trajectory:
    """
    Generates a number of fixed-length agent trajectories in multiagent settings where the agent simulates the
    decisions of other agents using Theory-of-Mind. A distribution over others' models is used at each step to
    compute their decisions.
    :param Agent agent: the agent for which to record the actions
    :param list[TeamModelsDistributions] models_dists: the sequence of distributions over other agents' models. The
    length of this sequence is used to compute the trajectory distribution.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :rtype: TeamModelDistTrajectory]
    :return: the trajectory containing a list of state-action-model_distribution tuples.
    """
    # copy world and get agents
    world = copy_world(agent.world)
    agent: Agent = world.agents[agent.name]
    other_agents: List[Agent] = [world.agents[other] for other in models_dists[0][agent.name].keys()]

    # generate or select initial state features
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)

    random.seed(seed)

    if model is not None:
        world.setFeature(modelKey(agent.name), model)

    # for each step, agent takes action while
    n_steps = len(models_dists)
    trajectory: Trajectory = []
    total = 0
    prob = 1.
    for i in range(n_steps):
        start = timer()

        # compute other agents' actions based on provided distribution over their models
        other_actions = {}
        for other in other_agents:
            # get agent's mental model distribution over other agents at this step
            dist = models_dists[i][agent.name][other.name]
            included_features = set(agent.getBelief(model=agent.get_true_model()).keys())
            agent.resetBelief(include=included_features)  # override actual features in belief states

            if select:
                # select only one model
                model, model_prob = dist.sample()
                dist = Distribution({model: 1.})
                prob *= model_prob  # weight this path

            # compute action for each of other's models, weight by prob
            other_models = other.models.copy()
            agent_models = agent.models.copy()
            action_dist = {}
            for model, model_prob in dist.items():
                decision = other.decide(selection='random', model=model,
                                        )
                                        # horizon=1)  # TODO
                action = world.value2float(actionKey(other.name), decision['action'])
                action_dist[setToConstantMatrix(actionKey(other.name), action)] = model_prob
            other.models = other_models  # we don't need all the new models
            agent.models = agent_models
            other_actions[other.name] = makeTree(Distribution(action_dist))

        # step the world until it's this agent's turn
        turn = world.getFeature(turnKey(agent.name), unique=True)
        while turn != 0:
            world.step()
            turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_state = copy.deepcopy(world.state)
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        # step the world to compute agent's action conditioned on other agents' actions
        world.step(actions=other_actions, select=False, horizon=horizon, tiebreak=selection,
                   threshold=threshold, updateBeliefs=False)

        action = world.getAction(agent.name)
        trajectory.append(StateActionPair(prev_state, action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}s')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_trajectories_tom(agent: Agent,
                              n_trajectories: int,
                              models_dists: List[TeamModelsDistributions],
                              init_feats: Optional[Dict[str, Any]] = None,
                              model: Optional[str] = None,
                              select: Optional[bool] = None,
                              horizon: Optional[int] = None,
                              selection: Optional[SelectionType] = None,
                              threshold: Optional[float] = None,
                              processes: int = -1,
                              seed: int = 0,
                              verbose: bool = False,
                              use_tqdm: bool = True) -> List[Trajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    This function is to be applied to multiagent scenarios, where an agent models other agents. A distribution over
    models of others is used during trajectory generation to sample their actions.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param list[TeamModelsDistributions] models_dists: the sequence of distributions over other agents' models. The
    length of this sequence is used to compute the trajectory distribution.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    world = agent.world
    if init_feats is not None:
        for feature in init_feats:
            assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(agent, models_dists, init_feats, model, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[Trajectory] = run_parallel(generate_trajectory_tom, args, processes=processes, use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {len(models_dists)}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def generate_team_trajectory(team: List[Agent],
                             trajectory_length: int,
                             init_feats: Optional[Dict[str, Any]] = None,
                             models: Optional[Dict[str, str]] = None,
                             select: Optional[bool] = None,
                             horizon: Optional[int] = None,
                             selection: Optional[SelectionType] = None,
                             threshold: Optional[float] = None,
                             seed: int = 0,
                             verbose: bool = False) -> TeamTrajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param List[Agent] team: the team of agents for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param dict[str,str] models: the agents' models used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(team[0].world)
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)

    random.seed(seed)

    # for each step, agents take an action and we register the state-team action pairs
    team_trajectory: TeamTrajectory = []
    total = 0
    prob = 1.
    if models is not None:
        for agent in team:
            if agent.name in models:
                world.setFeature(modelKey(agent.name), models[agent.name])

    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        for agent in team:
            turn = world.getFeature(turnKey(agent.name), unique=True)
            while turn != 0:
                world.step()
                turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_state = copy.deepcopy(world.state)  # keep (possibly stochastic) state and prob before selection
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        # steps the world (do not select), gets the agent's action
        world.step(select=False, horizon=horizon, tiebreak=selection, threshold=threshold)
        team_action: Dict[str, Distribution] = {}
        for agent in team:
            action = world.getAction(agent.name)
            team_action[agent.name] = action
        team_trajectory.append(TeamStateActionPair(prev_state, team_action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'[Seed {seed}] Step {i} took {step_time:.2f}s')

    if verbose:
        logging.info(f'[Seed {seed}] Total time: {total:.2f}s')

    return team_trajectory


def generate_team_trajectories(team: List[Agent],
                               n_trajectories: int,
                               trajectory_length: int,
                               init_feats: Optional[Dict[str, Any]] = None,
                               models: Optional[Dict[str, str]] = None,
                               select: Optional[bool] = None,
                               horizon: Optional[int] = None,
                               selection: Optional[SelectionType] = None,
                               threshold: Optional[float] = None,
                               processes: int = -1,
                               seed: int = 0,
                               verbose: bool = False,
                               use_tqdm: bool = True) -> List[TeamTrajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agents in the world.
    :param List[Agent] team: the team of agents for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param dict[str,str] models: the agents' models used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    for ag_i, agent in enumerate(team):
        assert agent.world == team[ag_i - 1].world
    world = team[0].world
    for feature in init_feats:
        assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(team, trajectory_length, init_feats, models, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[TeamTrajectory] = run_parallel(
        generate_team_trajectory, args, processes=processes, use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def sample_random_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int,
                                   with_replacement: bool = False,
                                   seed: int = 0) -> List[Trajectory]:
    """
    Randomly samples sub-trajectories from a given trajectory.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :param bool with_replacement: whether to allow repeated sub-trajectories to be sampled.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert with_replacement or len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    # samples randomly
    rng = random.Random(seed)
    idxs = list(range(len(trajectory) - trajectory_length + 1))
    idxs = rng.choices(idxs, k=n_trajectories) if with_replacement else rng.sample(idxs, n_trajectories)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def sample_spread_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int) -> List[Trajectory]:
    """
    Samples sub-trajectories from a given trajectory as spread in time as possible.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    idxs = np.asarray(np.arange(0, len(trajectory) - trajectory_length + 1,
                                (len(trajectory) - trajectory_length) / max(1, n_trajectories - 1)), dtype=int)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def log_trajectory(trajectory: StateActionProbTrajectory,
                   world: World,
                   features: List[str],
                   log_actions: bool = True,
                   log_prob: bool = True):
    """
    Prints the given trajectories to the log at the info level.
    :param StateActionProbTrajectory trajectory: the trajectory to be logged, containing sequences of state-action-prob tuples.
    :param World world: the PsychSim world from which to get the feature values.
    :param list[str] features: the state features to be printed at each step, representing the state of interest.
    :param bool log_actions: whether to log the actions taken by the agent(s) in the trajectory.
    :param bool log_prob: whether to log the probability associated with reaching each state in the trajectory.
    """
    for t, sap in enumerate(trajectory):
        feat_values = {feat: str(world.getFeature(feat, state=sap.state)) for feat in features}
        logging.info('----------------------------------------')
        logging.info(f'Timestep: {t}')
        logging.info(f'State: {feat_values}')
        if log_actions:
            logging.info(f'Actions: {sap.action}')
        if log_prob:
            logging.info(f'P={sap.prob:.2f}')


def get_trajectory_action_counts(trajectory: StateActionProbTrajectory, world: World) -> Dict[str, Dict[str, int]]:
    """
    Gets action counts for the given trajectory.
    :param StateActionProbTrajectory trajectory: the trajectory from which to get the action counts.
    :param World world: the PsychSim world from which to get all the agents' actions.
    :rtype: dict[str, dict[str, int]]
    :return: a dictionary containing, for each agent, a dictionary with the counts for each action.
    """
    # get agent names and their actions (assume consistent trajectory)
    action_stats: Dict[str, Dict[str, int]] = {}
    is_single_agent = isinstance(trajectory[0].action, Distribution)
    if is_single_agent:
        agent = dict(next(iter(trajectory[0].action)))['subject']
        assert agent in world.agents, f'Agent {agent} referred to in the trajectory was not found in the given world'
        action_stats[agent] = {}
    else:
        # assume multiagent trajectory
        for agent in trajectory[0].action.keys():
            action_stats[agent] = {}
    for agent in action_stats.keys():
        agent = world.agents[agent]
        action_stats[agent.name] = {action: 0 for action in agent.actions}

    # collects counts (weighted if stochastic actions) for each step in the trajectory (for each agent if multiagent)
    for sap in trajectory:
        if is_single_agent:
            agent = dict(next(iter(sap.action)))['subject']
            agents_actions = {agent: sap.action}
        else:
            agents_actions = sap.action
        for agent, actions in agents_actions.items():
            for action, prob in actions.items():
                action_stats[agent][action] += prob

    return action_stats
