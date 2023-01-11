import copy

import argparse
import logging
import os
import string
from typing import List, Optional, Dict

from model_learning import TeamTrajectory, TeamStateActionModelDistTuple, TeamModelDistTrajectory, ModelsDistributions
from model_learning.bin.sar import add_common_arguments, create_sar_world, create_observers, TeamConfig
from model_learning.trajectory import copy_world
from model_learning.util.io import load_object, save_object
from model_learning.util.mp import run_parallel
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, isModelKey, isActionKey, VectorDistributionSet, isRewardKey, isTurnKey, makeTree, \
    makeFuture, actionKey, CONSTANT, setToConstantMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'
__description__ = 'Performs reward model inference in the search-and-rescue domain.' \
                  'First, it loads a set of team trajectories corresponding to observed behavior between two ' \
                  'collaborative agents. ' \
                  'An observer agent then maintains beliefs (a probability distribution over reward models) about ' \
                  'each agent, which are updated via PsychSim inference according to the agents\' actions  .' \
                  'Trajectories with model inference are saved into a .pkl file.'

# common params
MODEL_RATIONALITY = 1  # .5
MODEL_SELECTION = 'distribution'  # 'random'  # 'distribution'


def get_belief(world: World, feature: str, ag: Agent, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(ag.name, unique=True)
    return world.getFeature(feature, state=ag.getBelief(model=model))


def update_state(world: World, state: VectorDistributionSet):
    # TODO the update() method of PsychSim state appears to not be working
    for key in state.keys():
        if (key in world.state and not isModelKey(key) and not isActionKey(key)
                and not isRewardKey(key) and not isTurnKey(key)):
            val = state.certain[key] if state.keyMap[key] is None else state[key]
            world.state.join(key, val)


def get_model_distributions(observers: Dict[str, Agent],
                            team_config: TeamConfig) -> ModelsDistributions:
    """
    Gets the distribution over models of other agents for all agents given the observers' current beliefs.
    :param dict[str, Agent] observers: the observer for each agent/role.
    :param TeamConfig team_config: the team configuration with references to the desired mental models.
    :return: a distribution over other agents' model names for each agent.
    """
    model_dists: ModelsDistributions = {}
    for role, observer in observers.items():
        # gets agent observer's beliefs about all other agents' models
        ag_model_dists: Dict[str, Distribution] = {}
        model_dists[role] = ag_model_dists

        for other, models in team_config[role].mental_models.items():
            # gets observer's current belief over the other agent's models
            dist = Distribution({f'{other}_{model}': 0 for model in models.keys()})
            ag_model_dists[other] = dist
            obs_dist = get_belief(observer.world, modelKey(other), observer)
            for model, prob in obs_dist.items():
                model = model.rstrip(string.digits)  # strip digits to get original model name
                dist[model] = prob
            dist.normalize()
    return model_dists


def team_trajectory_model_inference(world: World,
                                    team: List[Agent],
                                    trajectory: TeamTrajectory,
                                    observers: Dict[str, Agent],
                                    team_config: TeamConfig) -> TeamModelDistTrajectory:
    # reset to initial state in the given trajectory
    init_state = trajectory[0].world.state
    world = copy_world(world)
    update_state(world, init_state)
    team = [world.agents[agent.name] for agent in team]  # get new world's agents
    observers = {ag_name: world.agents[observer.name] for ag_name, observer in observers.items()}

    models_dists = get_model_distributions(observers, team_config)

    n_step = len(trajectory)
    team_trajectory: TeamModelDistTrajectory = []
    for step_i in range(n_step):
        # append state-action-models dists tuple
        _, team_action, prob = trajectory[step_i]
        team_trajectory.append(TeamStateActionModelDistTuple(copy.copy(world.state), team_action, models_dists, prob))

        if step_i == n_step - 1:
            break
        # m[makeFuture(action)][CONSTANT]

        team_action = {agent.name: team_action[agent.name].first() for agent in team}
        # team_action = {ag: makeTree(Distribution(
        #     {setToConstantMatrix(actionKey(ag), world.value2float(actionKey(ag), action)): prob
        #      for action, prob in action_dist.items()}))
        #     for ag, action_dist in team_action.items()}
        world.step(actions=team_action)

        models_dists = get_model_distributions(observers, team_config)

    return team_trajectory


def team_trajectories_model_inference(world: World,
                                      team: List[Agent],
                                      trajectories: List[TeamTrajectory],
                                      observers: Dict[str, Agent],
                                      team_config: TeamConfig,
                                      processes: Optional[int] = -1) -> List[TeamModelDistTrajectory]:
    args = [(world, team, trajectories[t], observers, team_config) for t in range(len(trajectories))]
    team_trajectories_with_model_dist: List[TeamModelDistTrajectory] = \
        run_parallel(team_trajectory_model_inference, args, processes=processes, use_tqdm=True)
    return team_trajectories_with_model_dist


def main():
    assert os.path.isfile(args.traj_file), f'Could not find the team trajectories file at {args.traj_file}'

    # create world and agents
    env, team, team_config, profiles = create_sar_world(args)

    # create and set observers' mental models
    observers = create_observers(env, team_config, profiles)

    env.world.dependency.getEvaluation()  # "compile" dynamics to speed up graph computation in parallel worlds

    # load trajectories
    logging.info('========================================')
    logging.info(f'Loading team trajectories from {args.traj_file}...')
    trajectories: List[TeamTrajectory] = load_object(args.traj_file)
    logging.info(f'Loaded {len(trajectories)} team trajectories of length {len(trajectories[0])}')

    logging.info('========================================')
    logging.info(f'Performing model inference over {len(trajectories)} trajectories...')
    team_model_dist_trajs = team_trajectories_model_inference(
        env.world, team, trajectories, observers, team_config, processes=args.processes)

    output_dir = args.output

    # save all trajectories to pickle file
    file_path = os.path.join(output_dir, 'trajectories.pkl.gz')
    logging.info(f'Saving trajectories to {file_path}...')
    save_object(team_model_dist_trajs, file_path, compress_gzip=True)

    logging.info('Done!')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    add_common_arguments(parser)

    parser.add_argument(
        '--traj-file', '-tf', type=str, required=True,
        help='Path to the .pkl.gz file containing the team trajectories over which to perform model inference.')

    args = parser.parse_args()

    main()
