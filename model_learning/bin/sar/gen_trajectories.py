import argparse
import logging
import numpy as np
import os
from typing import List, get_args

from model_learning import SelectionType
from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld, TeamConfig
from model_learning.features.linear import add_linear_reward_model, LinearRewardFunction
from model_learning.features.search_rescue import SearchRescueRewardVector
from model_learning.util.cmd_line import str2bool, save_args
from model_learning.util.io import create_clear_dir, save_object, get_file_name_without_extension
from model_learning.util.logging import change_log_handler
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World

__author__ = 'Haochen Wu, Pedro Sequeira'
__email__ = 'hcaawu@gmail.com, pedrodbs@gmail.com'
__maintainer__ = 'Pedro Sequeira'
__description__ = 'Collects a set of fixed-length trajectories in the search-and-rescue domain. ' \
                  'Saves the trajectories to a gif animation and pickle file.'

# default env params
WORLD_NAME = 'SAR'
ENV_SIZE = 3
SEED = 48
NUM_VICTIMS = 3
VICS_CLEARED_FEATURE = False

# default agent params
DISCOUNT = 0.7
HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'softmax'  # 'random'
RATIONALITY = 1 / 0.1

# default model params (make models less rational to get smoother (more cautious) inference over models)
MODEL_SELECTION = 'softmax'
MODEL_RATIONALITY = 1

# default common params
NUM_TRAJECTORIES = 1  # 5  # 10
TRAJ_LENGTH = 25  # 30
PROCESSES = -1
CLEAR = False  # True  # False


def main():
    assert os.path.isfile(args.team_config), \
        f'Could not find Json file with team\'s configuration in {args.team_config}'

    np.set_printoptions(precision=3)

    # create output
    output_dir = args.output
    create_clear_dir(output_dir, clear=args.clear)
    change_log_handler(os.path.join(output_dir, 'collect.log'), level=logging.INFO)
    save_args(args, os.path.join(output_dir, 'args.json'))

    world = World()
    # world.setParallel()
    env = SearchRescueGridWorld(world, args.size, args.size, args.victims, WORLD_NAME,
                                vics_cleared_feature=args.vics_cleared_feature, seed=args.seed)
    logging.info(f'Initialized World, h:{HORIZON}, x:{env.width}, y:{env.height}, v:{env.num_victims}')

    # team of two agents
    team_config = TeamConfig.load(args.team_config)
    team: List[Agent] = []
    for ag_name, ag_conf in team_config.items():
        # create and define agent dynamics
        agent = world.addAgent(ag_name)
        env.add_search_and_rescue_dynamics(agent, ag_conf.options)
        team.append(agent)

    env.world.setOrder([{agent.name for agent in team}])

    # collaboration dynamics
    env.add_collaboration_dynamics(team)

    # set agent rewards and attributes
    for agent in team:
        agent_lrv = SearchRescueRewardVector(env, agent)
        rwd_f_weights = team_config[agent.name].options.to_array()
        agent_lrv.set_rewards(agent, rwd_f_weights)

        logging.info(f'{agent.name} Reward Features')
        logging.info(agent_lrv.names)
        logging.info(rwd_f_weights)

        agent.setAttribute('selection', args.selection)
        agent.setAttribute('horizon', args.horizon)
        agent.setAttribute('rationality', args.rationality)
        agent.setAttribute('discount', args.discount)

    for ag_name, ag_conf in team_config.items():
        agent = world.agents[ag_name]
        agent_lrv = SearchRescueRewardVector(env, agent)

        if ag_conf.mental_models is not None:
            agent.create_belief_state()
            agent.set_observations()  # agent does not observe other agents' true models

        # create new agent reward models
        if ag_conf.models is not None:
            for name, options in ag_conf.models.items():
                rwd_function = LinearRewardFunction(agent_lrv, options.to_array())
                model = f'{agent.name}_{name}'
                add_linear_reward_model(agent, rwd_function, name=model, parent=None)
                agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
                agent.setAttribute('selection', MODEL_SELECTION, model=model)
                agent.create_belief_state(model=model)

    # add agents' mental models
    for ag_name, ag_conf in team_config.items():
        if ag_conf.mental_models is not None:
            agent = world.agents[ag_name]
            # set distribution over other agents' models
            for other_ag, models_probs in ag_conf.mental_models.items():
                dist = Distribution({f'{other_ag}_{model}': prob for model, prob in models_probs.items()})
                dist.normalize()
                world.setMentalModel(agent.name, other_ag, dist, model=None)

            zero_level = agent.zero_level(static=True, sample=True)

            # also, set mental model of other agents' models to a zero-level model
            # (exact same behavior as true agent) to avoid infinite recursion
            for other_ag, models_probs in ag_conf.mental_models.items():
                for model in models_probs.keys():
                    # zero_level = f'{agent.name}_zero_{other_ag}_{model}'
                    # agent.addModel(zero_level, parent=agent.get_true_model())
                    # agent.setAttribute('level', 0, model=zero_level)
                    # agent.setAttribute('beliefs', True, model=zero_level)
                    # world.setMentalModel(agent.name, other_ag, f'{other_ag}_{model}', model=zero_level)

                    # if world.agents[other_ag].getAttribute('beliefs', model= f'{other_ag}_{model}') is True:
                    #     world.agents[other_ag].create_belief_state(model=f'{other_ag}_{model}')

                    world.setMentalModel(other_ag, agent.name, zero_level, model=f'{other_ag}_{model}')

    world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    # generate trajectories using agents' policies and models
    logging.info(f'Generating {args.trajectories} trajectories of length {args.length}...')
    team_trajectories = env.generate_team_trajectories(
        team,
        trajectory_length=args.length,
        n_trajectories=args.trajectories,
        horizon=args.horizon,
        selection=args.selection,
        processes=args.processes,
        threshold=args.prune,
        seed=args.seed)

    # save trajectories to an animation (gif) file
    anim_path = os.path.join(output_dir, 'animations')
    create_clear_dir(anim_path, clear=False)
    logging.info(f'Saving animations for each trajectory to {anim_path}...')
    env.play_team_trajectories(team_trajectories, anim_path)

    # save all trajectories to pickle file
    file_path = os.path.join(output_dir, 'trajectories.pkl.gz')
    logging.info(f'Saving trajectories to {file_path}...')
    save_object(team_trajectories, file_path, compress_gzip=True)

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--team-config', '-cfg', type=str, required=True, help='Path to team configuration Json file.')
    parser.add_argument('--output', '-o', type=str, required=True, help='Directory in which to save results.')
    parser.add_argument('--size', '-sz', type=int, default=ENV_SIZE, help='Size of gridworld/environment.')
    parser.add_argument('--victims', '-nv', type=int, default=NUM_VICTIMS,
                        help='Number of victims in the environment.')
    parser.add_argument('--vics-cleared-feature', '-vcf', type=str2bool, default=VICS_CLEARED_FEATURE,
                        help='Whether to create a feature that counts the number of victims cleared.')

    parser.add_argument('--selection', '-as', type=str, choices=get_args(SelectionType), default=ACT_SELECTION,
                        help='Agents\' action selection criterion, to untie equal-valued actions.')
    parser.add_argument('--horizon', '-hz', type=int, default=HORIZON, help='Agents\' planning horizon.')
    parser.add_argument('--rationality', '-r', type=float, default=RATIONALITY,
                        help='Agents\' rationality when selecting actions under a probabilistic criterion.')
    parser.add_argument('--discount', '-d', type=float, default=DISCOUNT, help='Agents\' planning discount factor.')
    parser.add_argument('--prune', '-pt', type=float, default=PRUNE_THRESHOLD,
                        help='Likelihood threshold for pruning outcomes during planning. `None` means no pruning.')

    parser.add_argument('--trajectories', '-t', type=int, default=NUM_TRAJECTORIES,
                        help='Number of trajectories to generate.')
    parser.add_argument('--length', '-l', type=int, default=TRAJ_LENGTH, help='Length of the trajectories to generate.')

    parser.add_argument('--img-format', '-f', type=str, default='pdf', help='The format of image files.')

    parser.add_argument('--seed', '-s', type=int, default=SEED, help='Random seed to initialize world and agents.')
    parser.add_argument('--processes', '-p', type=int, default=PROCESSES,
                        help='The number of parallel processes to use. `-1` or `None` will use all available CPUs.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    main()
