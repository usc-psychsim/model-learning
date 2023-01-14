import argparse
import itertools as it
import logging
import os
import tqdm
from typing import List, Tuple, Dict

from model_learning.bin.sar.config import AgentProfiles, TeamConfig
from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld, AgentProfile
from model_learning.features.linear import LinearRewardFunction, add_linear_reward_model
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.util.cmd_line import str2bool, str2log_level
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'

# default env params
WORLD_NAME = 'SAR'
ENV_SIZE = 3
SEED = 48
NUM_VICTIMS = 3
VICS_CLEARED_FEATURE = False

# default agent params
DISCOUNT = 0.7
HORIZON = 2  # 0 for random actions
ACT_SELECTION = 'softmax'  # 'random'
RATIONALITY = 1 / 0.1

# default model params (make models less rational to get smoother (more cautious) inference over models)
MODEL_SELECTION = 'distribution'  # 'softmax'
MODEL_RATIONALITY = 1
MODEL_HORIZON = 1  # TODO HORIZON
OBSERVER_MODEL_NAME = 'OBSERVER'

# default common params
PRUNE_THRESHOLD = 1e-3  # 1e-2
PROCESSES = -1
CLEAR = False  # True  # False
PROFILES_FILE_PATH = os.path.abspath(os.path.dirname(__file__) + '/../../res/sar/profiles.json')

OBSERVER_NAME = 'Observer'


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

    parser.add_argument('--prune', '-pt', type=float, default=PRUNE_THRESHOLD,
                        help='Likelihood threshold for pruning outcomes during planning. `None` means no pruning.')
    parser.add_argument('--img-format', '-f', type=str, default='pdf', help='The format of image files.')

    parser.add_argument('--seed', '-s', type=int, default=SEED, help='Random seed to initialize world and agents.')
    parser.add_argument('--processes', '-p', type=int, default=PROCESSES,
                        help='The number of parallel processes to use. `-1` or `None` will use all available CPUs.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=logging.INFO, help='Logging verbosity level.')


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
    # world.setParallel()
    env = SearchRescueGridWorld(world, args.size, args.size, args.victims, WORLD_NAME,
                                vics_cleared_feature=args.vics_cleared_feature, seed=args.seed)
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
        agent.belief_threshold = args.get('prune', PRUNE_THRESHOLD)

    return env, team, team_config, profiles


def create_agent_models(env: SearchRescueGridWorld, team_config: TeamConfig, profiles: AgentProfiles):
    """
    Creates reward-based agent models according to the given team configuration and profiles.
    :param SearchRescueGridWorld env: the search-and-rescue environment.
    :param TeamConfig team_config: the team's configuration.
    :param AgentProfiles profiles: the set of agent profiles.
    """
    logging.info('Creating agent models...')
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        agent = env.world.agents[role]
        agent_lrv = SearchRescueRewardVector(env, agent)

        # create new agent reward models
        models = team_config.get_all_model_profiles(role, profiles)
        for model, profile in models.items():
            create_agent_model(agent, agent_lrv, model, profile)


def create_agent_model(agent: Agent, agent_lrv: SearchRescueRewardVector, model: str, profile: AgentProfile):
    rwd_function = LinearRewardFunction(agent_lrv, profile.to_array())
    model = f'{agent.name}_{model}'
    add_linear_reward_model(agent, rwd_function, model=model, parent=None,
                            rationality=MODEL_RATIONALITY,
                            selection=MODEL_SELECTION,
                            horizon=MODEL_HORIZON)
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
                logging.info(f'Set mental models to agent {role} of agent {other}:\n{dist}')

            zero_level = agent.zero_level(static=True, sample=True)

            # also, set mental model of other agents' models to a zero-level model
            # (exact same behavior as true agent) to avoid infinite recursion
            for other, models_probs in ag_conf.mental_models.items():
                for model in models_probs.keys():
                    # zero_level = f'{agent.name}_zero_{other_ag}_{model}'
                    # agent.addModel(zero_level, parent=agent.get_true_model())
                    # agent.setAttribute('level', 0, model=zero_level)
                    # agent.setAttribute('beliefs', True, model=zero_level)
                    # world.setMentalModel(agent.name, other_ag, f'{other_ag}_{model}', model=zero_level)

                    # if world.agents[other_ag].getAttribute('beliefs', model= f'{other_ag}_{model}') is True:
                    #     world.agents[other_ag].create_belief_state(model=f'{other_ag}_{model}')

                    model = f'{other}_{model}'
                    world.agents[other].create_belief_state(model=model)
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

    logging.info('Creating observers...')
    world = env.world
    observers: Dict[str, Agent] = {}
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        # create observer agent
        observer = world.addAgent(f'{OBSERVER_NAME}_{role}')
        observers[role] = observer
        observer.belief_threshold = PRUNE_THRESHOLD

    # make observers ignore each other
    for obs1, obs2 in it.combinations(list(observers.keys()), 2):
        observers[obs1].ignore(observers[obs2].name)
        observers[obs2].ignore(observers[obs1].name)

    logging.info('Setting observers\' mental models...')
    for role, ag_conf in tqdm.tqdm(team_config.items()):
        observer = observers[role]
        observer.set_observations()  # observer does not observe other agents' true models

        # set distribution over other agents' models for this observer
        if ag_conf.mental_models is not None:
            for other_ag, models_probs in ag_conf.mental_models.items():
                dist = Distribution({f'{other_ag}_{model}': prob for model, prob in models_probs.items()})
                dist.normalize()
                world.setMentalModel(observer.name, other_ag, dist, model=None)
                logging.info(f'Set mental models of agent {other_ag} to {observer.name}:\n{dist}')

        # create agent model just for observer (needs to have special action selection)
        if ag_conf.profile is not None:
            agent = world.agents[role]
            create_agent_model(agent, SearchRescueRewardVector(env, agent), OBSERVER_MODEL_NAME,
                               profiles[role][ag_conf.profile])
            world.setMentalModel(observer.name, role, f'{role}_{OBSERVER_MODEL_NAME}', model=None)

        # make all agents and their models ignore this observer
        for _role in team_config.keys():
            agent = world.agents[_role]
            for model in agent.models.keys():
                agent.ignore(observer.name, model=model)

    return observers
