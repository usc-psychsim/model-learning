import logging

from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import modelKey, makeTree, setToConstantMatrix, rewardKey, actionKey
from psychsim.reward import achieveFeatureValue, maximizeFeature
from psychsim.helper_functions import get_true_model_name
from model_learning.environments.grid_world import GridWorld
from model_learning.util.io import create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = ''

ENV_SIZE = 10

AGENT_NAME = 'Agent'
OBSERVER_NAME = 'Observer'
HORIZON = 2
SELECTION = 'random'  # stochastic over best actions

NUM_STEPS = 50

OUTPUT_DIR = 'output/examples/collect-trajectories'
DEBUG = True


def print_agent_loc():
    print('Agent is at: ({},{})'.format(world.getValue(x), world.getValue(y)))


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world and agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', SELECTION)

    # create grid-world and add world dynamics to agent
    env = GridWorld(world, ENV_SIZE, ENV_SIZE)
    env.add_agent_dynamics(agent)

    # set true reward function (achieve middle location)
    x, y = env.get_location_features(agent)
    agent.setReward(achieveFeatureValue(x, 4, agent.name), 1.)
    agent.setReward(achieveFeatureValue(y, 4, agent.name), 1.)

    # create observer
    observer = world.addAgent(OBSERVER_NAME)
    world.setOrder([{agent.name}])

    # ignore observer
    agent.ignore(observer.name)
    # agent.omega = {actionKey(agent.name)}  # todo should not need this

    # agent models and observer beliefs
    true_model = get_true_model_name(agent)
    agent.addModel('middle-pos', parent=true_model, rationality=.5)

    agent.addModel('maximize-pos', parent=true_model, rationality=.5)
    agent.setReward(maximizeFeature(x, agent.name), 1., 'maximize-pos')
    agent.setReward(maximizeFeature(y, agent.name), 1., 'maximize-pos')

    # agent.addModel('zero-rwd', parent=true_model, rationality=.5)
    # agent.setReward(makeTree(setToConstantMatrix(rewardKey(agent.name), 0)), model='zero-rwd')

    # observer does not model itself
    observer.resetBelief(ignore={modelKey(observer.name)})

    # observer has uniform prior distribution over possible agent models
    world.setMentalModel(observer.name, agent.name,
                         Distribution({name: 1. / (len(agent.models) - 1) for name in agent.models.keys()
                                       if name != true_model}))

    # observer observes everything except agent's reward received and true models
    observer.omega = {key for key in world.state.keys() if
                      key not in {modelKey(agent.name), modelKey(observer.name), rewardKey(agent.name)}}

    for t in range(NUM_STEPS):
        print_agent_loc()

        world.step()

        beliefs = agent.getBelief()
        assert len(beliefs) == 1  # Because we are dealing with a known-identity agent
        belief = next(iter(agent.getBelief().values()))
        print('Agent now models player as:')
        key = modelKey(agent.name)
        print(world.float2value(key, belief[key]))
