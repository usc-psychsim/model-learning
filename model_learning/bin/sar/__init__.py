import argparse
import logging
import numpy as np
import os
import pandas as pd
import tqdm
from typing import List, Tuple, Dict, Optional, get_args

from model_learning import SelectionType, MultiagentTrajectory, State
from model_learning.bin.sar.config import AgentProfiles, TeamConfig
from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld
from model_learning.features.counting import estimate_feature_counts_tom, estimate_feature_counts, mean_feature_counts
from model_learning.features.linear import LinearRewardFunction, add_linear_reward_model
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.inference import MODEL_SELECTION, MODEL_RATIONALITY, MODEL_HORIZON, create_inference_observers, \
    get_model_distributions
from model_learning.util.cmd_line import str2bool, str2log_level
from model_learning.util.plot import plot_bar
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'

# default env params
WORLD_NAME = 'SAR'
ENV_SIZE = 3
ENV_SEED = 48
NUM_VICTIMS = 3
VICS_CLEARED_FEATURE = False

# default agent params
DISCOUNT = 0.7
HORIZON = 2  # 0 for random actions
ACT_SELECTION = 'softmax'  # 'random'
RATIONALITY = 1 / 0.1

# default common params
SEED = 48
BELIEF_THRESHOLD = 0  # 1e-5
PRUNE_THRESHOLD = 1e-2
PROCESSES = -1
CLEAR = False  # True  # False
PROFILES_FILE_PATH = os.path.abspath(os.path.dirname(__file__) + '/../../data/sar/profiles.json')

# default trajectory generation params
NUM_TRAJECTORIES = 16  # 5  # 10
TRAJ_LENGTH = 25  # 30


def add_common_arguments(parser: argparse.ArgumentParser):
    """
    Adds the command-line arguments that are common to apply MIRL-ToM in the search-and-rescue domain.
    :param argparse.ArgumentParser parser: the argument parse to add the arguments to.
    """
    parser.add_argument('--team-config', '-cfg', type=str, required=True, help='Path to team configuration Json file.')
    parser.add_argument('--profiles', '-pf', type=str, default=PROFILES_FILE_PATH,
                        help='Path to agent profiles Json file.')
    parser.add_argument('--output', '-o', type=str, required=True, help='Directory in which to save results.')

    parser.add_argument('--size', '-sz', type=int, default=ENV_SIZE, help='Size of gridworld/environment.')
    parser.add_argument('--victims', '-nv', type=int, default=NUM_VICTIMS,
                        help='Number of victims in the environment.')
    parser.add_argument('--vics-cleared-feature', '-vcf', type=str2bool, default=VICS_CLEARED_FEATURE,
                        help='Whether to create a feature that counts the number of victims cleared.')
    parser.add_argument('--discount', '-d', type=float, default=DISCOUNT, help='Agents\' planning discount factor.')

    parser.add_argument('--prune', '-pt', type=float, default=PRUNE_THRESHOLD,
                        help='Likelihood threshold for pruning outcomes during planning. `None` means no pruning.')
    parser.add_argument('--img-format', '-f', type=str, default='pdf', help='The format of image files.')

    parser.add_argument('--seed', '-s', type=int, default=SEED, help='Random seed to initialize world and agents.')
    parser.add_argument('--processes', '-p', type=int, default=PROCESSES,
                        help='The number of parallel processes to use. `-1` or `None` will use all available CPUs.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=logging.INFO, help='Logging verbosity level.')


def add_agent_arguments(parser: argparse.ArgumentParser):
    """
    Adds the command-line arguments that allow parameterizing the PsychSim agents.
    :param argparse.ArgumentParser parser: the argument parse to add the arguments to.
    """
    parser.add_argument('--selection', '-as', type=str, choices=get_args(SelectionType), default=ACT_SELECTION,
                        help='Agents\' action selection criterion, to untie equal-valued actions.')
    parser.add_argument('--horizon', '-hz', type=int, default=HORIZON, help='Agents\' planning horizon.')
    parser.add_argument('--rationality', '-r', type=float, default=RATIONALITY,
                        help='Agents\' rationality when selecting actions under a probabilistic criterion.')


def add_trajectory_arguments(parser: argparse.ArgumentParser):
    """
    Adds the command-line arguments that allow parameterizing the PsychSim agents.
    :param argparse.ArgumentParser parser: the argument parse to add the arguments to.
    """
    parser.add_argument('--trajectories', '-t', type=int, default=NUM_TRAJECTORIES,
                        help='Number of trajectories to generate.')
    parser.add_argument('--length', '-l', type=int, default=TRAJ_LENGTH, help='Length of the trajectories to generate.')


def create_sar_world(args: argparse.Namespace) -> \
        Tuple[SearchRescueGridWorld, List[Agent], TeamConfig, AgentProfiles]:
    """
    Creates and initializes a search-and-rescue world from the given command-line arguments. A team of agents is also
    created and initialized, including setting their reward functions.
    :param argparse.Namespace args: the parsed arguments containing the parameters to initialize the world and agents.
    :return: a tuple containing the created search-and-rescue world, the team of PsychSim agents, and the team config
    and agent profiles loaded from file.
    """
    # check config files
    assert os.path.isfile(args.team_config), \
        f'Could not find Json file with team\'s configuration in {args.team_config}'
    assert os.path.isfile(args.profiles), \
        f'Could not find Json file with agent profiles in {args.profiles}'

    logging.info('========================================')
    logging.info('Loading agent configurations...')
    profiles = AgentProfiles.load(args.profiles)
    logging.info(f'Loaded a total of {sum([len(p) for r, p in profiles.items()])} profiles for {len(profiles)} roles')

    team_config = TeamConfig.load(args.team_config)
    team_config.check_profiles(profiles)

    logging.info('========================================')
    logging.info('Creating world and agents...')
    world = World()
    env = SearchRescueGridWorld(world, args.size, args.size, args.victims, WORLD_NAME,
                                vics_cleared_feature=args.vics_cleared_feature,
                                seed=ENV_SEED)  # environment uses fixed seed for now
    logging.info(f'Initialized World, w:{env.width}, h:{env.height}, v:{env.num_victims}')

    # team of two agents
    team: List[Agent] = []
    for role, ag_conf in team_config.items():
        # create and define agent dynamics
        agent = world.add_agent(role)
        profile = profiles[role][ag_conf.profile]
        env.add_search_and_rescue_dynamics(agent, profile)
        team.append(agent)

    world.setOrder([{agent.name for agent in team}])

    # collaboration dynamics
    env.add_collaboration_dynamics(team)

    # set agent rewards and attributes
    logging.info('Setting agent attributes...')
    args = vars(args)
    for agent in team:
        agent_lrv = SearchRescueRewardVector(env, agent)
        role = agent.name
        profile = profiles[role][team_config[role].profile]
        rwd_f_weights = profile.to_array()
        agent_lrv.set_rewards(agent, rwd_f_weights)

        logging.info(f'{agent.name} Reward Features')
        logging.info(agent_lrv.names)
        logging.info(rwd_f_weights)

        agent.setAttribute('selection', args.get('selection', ACT_SELECTION))
        agent.setAttribute('horizon', args.get('horizon', HORIZON))
        agent.setAttribute('rationality', args.get('rationality', RATIONALITY))
        agent.setAttribute('discount', args.get('discount', DISCOUNT))
        agent.belief_threshold = BELIEF_THRESHOLD

    return env, team, team_config, profiles


def create_agent_models(env: SearchRescueGridWorld,
                        team_config: TeamConfig,
                        profiles: AgentProfiles,
                        rationality: Optional[float] = MODEL_RATIONALITY,
                        horizon: Optional[int] = MODEL_HORIZON,
                        selection: Optional[SelectionType] = MODEL_SELECTION):
    """
    Creates reward-based agent models according to the given team configuration and profiles.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    :param float rationality: the agents' rationality when selecting actions, as modelled by the observers.
    :param int horizon: the agents' planning horizon as modelled by the observers.
    :param str selection: the agents' action selection criterion, as modelled by the observers.
    """
    logging.info('Creating agent models...')
    for role in tqdm.tqdm(team_config.get_all_modeled_roles()):
        agent = env.world.agents[role]
        agent_lrv = SearchRescueRewardVector(env, agent)

        # create new agent reward models
        models = team_config.get_all_model_profiles(role, profiles)
        for model, profile in models.items():
            rwd_function = LinearRewardFunction(agent_lrv, profile.to_array())
            add_linear_reward_model(agent, rwd_function, model=model, parent=None,
                                    rationality=rationality,
                                    selection=selection,
                                    horizon=horizon)
            logging.info(f'Set model {model} to agent {agent.name}')


def create_mental_models(env: SearchRescueGridWorld, team_config: TeamConfig, profiles: AgentProfiles):
    """
    Create the agent models and mental models (beliefs) based on the given team configuration and agent profiles.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    """
    logging.info('========================================')

    world = env.world
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        agent = world.agents[role]
        if ag_conf.mental_models is not None:
            agent.create_belief_state()
            agent.set_observations()  # agent does not observe other agents' true models

    # creates agent models
    create_agent_models(env, team_config, profiles)

    logging.info('Creating agent mental models of teammates...')
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        agent = world.agents[role]
        if ag_conf.mental_models is not None:

            # set distribution over other agents' models
            for other, models_probs in ag_conf.mental_models.items():
                dist = Distribution({f'{other}_{model}': prob for model, prob in models_probs.items()})
                dist.normalize()
                world.setMentalModel(agent.name, other, dist, model=None)
                logging.info(f'Set agent {role}\'s mental models of agent {other}:\n{dist}')

            # zero_level = agent.get_true_model()
            # zero_level = agent.zero_level(static=False, sample=True)
            # zero_level = f'{agent.name}_zero'
            # agent.addModel(zero_level, parent=agent.get_true_model())
            # agent.setAttribute('R', copy.copy(agent.getAttribute('R', model=agent.get_true_model())), model=zero_level)

            # also, set mental model of other agents' models to a zero-level model
            # (exact same behavior as true agent) to avoid infinite recursion
            for other, models_probs in ag_conf.mental_models.items():
                for model in models_probs.keys():
                    model = f'{other}_{model}'

                    zero_level = f'{agent.name}_zero_{other}_{model}'
                    agent.addModel(zero_level, parent=agent.get_true_model())
                    # agent.setAttribute('level', 0, model=zero_level)
                    # agent.setAttribute('beliefs', True, model=zero_level)
                    world.setMentalModel(agent.name, other, model, model=zero_level)

                    # if world.agents[other_ag].getAttribute('beliefs', model= f'{other_ag}_{model}') is True:
                    #     world.agents[other_ag].create_belief_state(model=f'{other_ag}_{model}')

                    # world.agents[other].create_belief_state(model=model)
                    world.setMentalModel(other, agent.name, zero_level, model=model)


def create_observers(env: SearchRescueGridWorld,
                     team_config: TeamConfig,
                     profiles: AgentProfiles) -> Dict[str, Agent]:
    """
    Creates model observers that will maintain a probability distribution over agent models (beliefs) as the agents
    act in the environment. One observer will be created per agent and the beliefs will be created and initialized
    according to the provided configuration.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    :rtype: dict[str, Agent]
    :return: a dictionary containing the observer agent created for each agent role.
    """

    logging.info('========================================')

    # creates agents' models
    create_agent_models(env, team_config, profiles)

    logging.info('Creating observers and setting mental models...')
    init_models_dists = team_config.get_models_distributions()  # creates initial distributions over models
    observers = create_inference_observers(env.world, init_models_dists, belief_threshold=BELIEF_THRESHOLD)

    models_dists = get_model_distributions(observers, init_models_dists)
    for agent, observer in observers.items():
        for other, models in models_dists[agent].items():
            logging.info(f'Set mental models of agent {other} to {observer.name}:\n{models}')

    return observers


def setup_modeling_agent(agent: str,
                         env: SearchRescueGridWorld,
                         team_config: TeamConfig,
                         profiles: AgentProfiles,
                         rationality: float = MODEL_RATIONALITY,
                         horizon: int = MODEL_HORIZON,
                         selection: SelectionType = MODEL_SELECTION):
    """
    Creates the (mental) models of other agents for a given agent according to the given configuration.
    This is useful for decentralized MIRL and feature count estimation since we are only assessing an agent's behavior
    against known models of other agents, i.e., the agent itself is not modeled by others.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    :param str agent: the role for which to create the mental models.
    :param float rationality: the other agent models' rationality when selecting actions, as modelled by the agent.
    :param int horizon: the other agent models' planning horizon as modelled by the agent.
    :param str selection: the other agent models' action selection criterion, as modelled by the agent.
    """
    # filters team_config to contain only the agent's config with mental models
    team_config = TeamConfig({agent: team_config[agent]})

    # creates models of the other agents
    create_agent_models(env, team_config, profiles,
                        rationality=rationality,
                        horizon=horizon,
                        selection=selection)

    # create new models of learner agent to avoid cycles
    agent: Agent = env.world.agents[agent]
    world: World = agent.world
    mental_models = team_config[agent.name].mental_models
    if mental_models is not None:
        for other, models in mental_models.items():
            for model in models.keys():
                model = f'{other}_{model}'
                learner_model = f'{model}_zero_{agent.name}'
                agent.addModel(learner_model, parent=agent.get_true_model())
                world.setMentalModel(agent.name, other, model, model=learner_model)
                world.setMentalModel(other, agent.name, learner_model, model=model)


def plot_feature_counts(features: np.ndarray,
                        probs: np.ndarray,
                        title: str,
                        labels: List[str],
                        output_dir: str,
                        img_format: str):
    """
    Plots a bar chart with the mean counts over trajectories for each feature.
    :param np.ndarray features: the "raw" feature values for a set of trajectories, corresponding to an array of
    shape (num_traj, timesteps, num_features).
    :param np.ndarray probs: the probabilities associated with each timestep of each trajectory, corresponding to
    an array of shape (num_traj, timesteps).
    :param str title: the plot title.
    :param list[str] labels: the names of the features.
    :param str output_dir: the path to the directory in which to save the plot.
    :param str img_format: the format of the image in which to plot the chart.
    """
    # transforms data to get fcs and weights by trajectory
    features = np.sum(features, axis=1)  # shape (num_traj, num_features)
    probs = probs[:, -1].reshape(-1, 1)  # shape (num_traj, 1)
    df = pd.DataFrame(np.concatenate([features, probs], axis=1), columns=labels + ['Weights'])
    plot_bar(df, title, os.path.join(output_dir, f'{title.lower().replace(" ", "-")}.{img_format}'),
             weights='Weights', x_label='Reward Features', y_label='Mean Count')


def get_estimated_feature_counts(trajectories: List[MultiagentTrajectory],
                                 env: SearchRescueGridWorld,
                                 agent: Agent,
                                 team_config: TeamConfig,
                                 profiles: AgentProfiles,
                                 output_dir: str,
                                 args: argparse.Namespace) -> np.ndarray:
    """
    Gets estimated feature counts via Monte Carlo trajectory sampling (possibly using model distribution sequence)
    :param list[MultiagentTrajectory] trajectories: the set of trajectories used to estimate feature counts.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param Agent agent: the PsychSim agent used to estimate the feature counts.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    :param str output_dir: the directory in which to save the feature count bar chart.
    :param argparse.Namespace args: the command-line arguments.
    :rtype: np.ndarray
    :return: an array of shape (num_features, ) containing the mean feature counts.
    """
    # creates mental models for agent and set same agent params to models since they will only be used to simulate the
    # other agents' actions, on which the learner agent's actions will then be conditioned
    logging.info(f'Setting up agent  {agent.name} for FC estimation...')
    setup_modeling_agent(agent.name, env, team_config, profiles,
                         rationality=args.rationality,
                         horizon=args.horizon,
                         selection=args.selection)

    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    logging.info(f'Estimating feature counts for agent {agent.name} '
                 f'using {args.trajectories} Monte-Carlo trajectories...')
    agent_lrv = SearchRescueRewardVector(env, agent)

    if hasattr(trajectories[0][0], 'models_dists'):
        logging.info('Generating trajectories *with* model distribution sequence')

        # checks model distributions in trajectories (considered consistent)
        models_dists = trajectories[0][0].models_dists
        for other, models in team_config[agent.name].mental_models.items():
            assert other in models_dists[agent.name], \
                f'Agent {other} not present in trajectories\' mental model distribution ' \
                f'of agent {agent.name}: {list(models_dists.keys())}'
            for model in models.keys():
                assert f'{other}_{model}' in models_dists[agent.name][other], \
                    f'Model {model} of agent {other} not present in trajectories\' mental model distribution ' \
                    f'of agent {agent.name}: {list(models_dists[other].keys())}'

        feature_func = lambda s: agent_lrv.get_values(s)
        features, probs = estimate_feature_counts_tom(
            agent, trajectories,
            feature_func=feature_func,
            exact=False,
            num_mc_trajectories=args.trajectories,
            horizon=args.horizon,
            threshold=args.prune,
            unweighted=True,
            processes=args.processes,
            seed=args.seed,
            verbose=False,
            use_tqdm=True)
    else:
        logging.info('Generating trajectories *without* model distribution sequence')

        init_states: List[State] = [trajectory[0].state for trajectory in trajectories]
        feature_func = lambda s: agent_lrv.get_values(s)
        features, probs = estimate_feature_counts(
            agent, init_states,
            trajectory_length=len(trajectories[0]),  # assume same-length trajectories
            feature_func=feature_func,
            exact=False,
            num_mc_trajectories=args.trajectories,
            horizon=args.horizon,
            threshold=args.prune,
            unweighted=True,
            processes=args.processes,
            seed=args.seed,
            verbose=False,
            use_tqdm=True)

    plot_feature_counts(features, probs, title=f'Learner {agent.name} Estimated FCs',
                        labels=agent_lrv.names, output_dir=output_dir, img_format=args.img_format)

    return mean_feature_counts(features, probs)  # shape: (num_features, )
