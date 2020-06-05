import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import modelKey, makeTree, setToConstantMatrix, rewardKey
from psychsim.reward import achieveFeatureValue, maximizeFeature
from psychsim.helper_functions import get_true_model_name
from model_learning.environments.grid_world import GridWorld
from model_learning.util.io import create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Perform reward model inference in the normal gridworld.' \
                  'There is one moving agent whose reward function is to reach the center of the grid.' \
                  'There is an observer agent that has 3 models of the moving agent (uniform prior):' \
                  '  - "middle_loc", i.e., the model with the true reward function;' \
                  '  - "maximize_loc", a model with a reward function that maximizes the coordinates values;' \
                  '  - "zero_rwd", a model with a zero reward function, resulting in a random behavior.' \
                  'The world is updated for some steps, where observer updates its belief over the models of the ' \
                  'moving agent via PsychSim inference. A plot is show with the inference evolution.'

ENV_SIZE = 10
NUM_STEPS = 100

OBSERVER_NAME = 'observer'
AGENT_NAME = 'agent'
MIDDLE_LOC_MODEL = 'middle_loc'
MAXIMIZE_LOC_MODEL = 'maximize_loc'
RANDOM_MODEL = 'zero_rwd'

HORIZON = 1  # TODO > 1 gives an error
MODEL_SELECTION = 'distribution'  # TODO 'consistent' or 'random' gives an error
AGENT_SELECTION = 'random'

OUTPUT_DIR = 'output/examples/reward-model-inference'
DEBUG = False
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _plot_evolution(probs):
    plt.figure()

    for i, model_name in enumerate(model_names):
        plt.plot(probs[i], label=model_name)

    plt.title('Evolution of Model Inference', fontweight='bold', fontsize=12)
    plt.xlabel('Time', fontweight='bold')
    plt.ylabel('Model probability', fontweight='bold')
    plt.xlim([0, NUM_STEPS])
    plt.legend()

    file_name = os.path.join(OUTPUT_DIR, 'inference.png')
    plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
    logging.info('Saved model inference evolution plot to:\n\t{}'.format(file_name))
    if SHOW:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world, agent and observer
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', AGENT_SELECTION)
    observer = world.addAgent(OBSERVER_NAME)

    # create grid-world and add world dynamics to agent
    env = GridWorld(world, ENV_SIZE, ENV_SIZE)
    env.add_agent_dynamics(agent)

    # set true reward function (achieve middle location)
    x, y = env.get_location_features(agent)
    agent.setReward(achieveFeatureValue(x, 4, agent.name), 1.)
    agent.setReward(achieveFeatureValue(y, 4, agent.name), 1.)

    world.setOrder([{agent.name}])

    # observer does not model itself
    observer.resetBelief(ignore={modelKey(observer.name)})

    # agent does not model itself and sees everything except true models and its reward
    agent.resetBelief(ignore={modelKey(observer.name)})
    agent.omega = {key for key in world.state.keys()
                   if key not in {modelKey(agent.name), rewardKey(agent.name), modelKey(observer.name)}}

    # get the canonical name of the "true" agent model
    true_model = get_true_model_name(agent)

    # agent's models
    agent.addModel(MIDDLE_LOC_MODEL, parent=true_model, rationality=.8, selection=MODEL_SELECTION)

    agent.addModel(MAXIMIZE_LOC_MODEL, parent=true_model, rationality=.8, selection=MODEL_SELECTION)
    agent.setReward(maximizeFeature(x, agent.name), 1., MAXIMIZE_LOC_MODEL)
    agent.setReward(maximizeFeature(y, agent.name), 1., MAXIMIZE_LOC_MODEL)

    if INCLUDE_RANDOM_MODEL:
        agent.addModel(RANDOM_MODEL, parent=true_model, rationality=.8, selection=MODEL_SELECTION)
        agent.setReward(makeTree(setToConstantMatrix(rewardKey(agent.name), 0)), model=RANDOM_MODEL)

    # observer has uniform prior distribution over possible agent models
    world.setMentalModel(observer.name, agent.name,
                         Distribution({name: 1. / (len(agent.models) - 1)
                                       for name in agent.models.keys() if name != true_model}))

    # observer sees everything except true models
    observer.omega = {key for key in world.state.keys()
                      if key not in {modelKey(agent.name), modelKey(observer.name)}}  # rewardKey(agent.name),

    random.seed(0)

    model_names = [name for name in agent.models.keys() if name != true_model]
    probs = np.zeros((NUM_STEPS, len(model_names)))

    for t in range(NUM_STEPS):
        logging.info('===============================================')
        logging.info('{}.\tlocation: ({},{})'.format(t, world.getValue(x), world.getValue(y)))

        beliefs = observer.getBelief()
        assert len(beliefs) == 1  # Because we are dealing with a known-identity agent
        belief = next(iter(observer.getBelief().values()))
        model_dist = world.getFeature(modelKey(agent.name), belief)
        for model in model_dist.domain():
            probs[t, model_names.index(model)] = model_dist[model]

        logging.info('Observer models agent as:')
        logging.info(model_dist)

        decision = agent.decide()
        action = decision[world.getValue(modelKey(agent.name))]['action']
        if isinstance(action, Distribution):
            action = random.choices(action.domain(), action.values())[0]
        logging.info(action)

        world.step(action)

    # create and save inference evolution plot
    _plot_evolution(probs.T)
