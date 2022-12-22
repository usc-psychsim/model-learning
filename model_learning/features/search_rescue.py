from typing import List

from model_learning.environments.gridworld import NOOP_ACTION
from model_learning.environments.search_rescue_gridworld import SearchRescueGridWorld, DIST_TO_VIC_FEATURE, \
    VICS_CLEARED_FEATURE, SEARCH_ACTION, TRIAGE_ACTION, EVACUATE_ACTION, CALL_ACTION, DIST_TO_HELP_FEATURE
from model_learning.features import LinearRewardVector
from model_learning.features.linear import LinearRewardFeature, NumericLinearRewardFeature, ActionLinearRewardFeature
from psychsim.agent import Agent

__author__ = 'Pedro Sequeira, Haochen Wu'
__email__ = 'pedrodbs@gmail.com, hcaawu@gmail.com'

DIST_TO_VIC_RWD_WEIGHT = 0.1
SEARCH_RWD_WEIGHT = 0.1
TRIAGE_RWD_WEIGHT = 0.3
EVAC_RWD_WEIGHT = 1
NOOP_RWD_WEIGHT = 0.05
CALL_RWD_WEIGHT = 0.05
DIST_TO_HELP_RWD_WEIGHT = 1
VICS_CLEAR_RWD_WEIGHT = 1


class SearchRescueRewardVector(LinearRewardVector):
    """
    Represents a linear reward vector for the search and rescue world.
    """

    def __init__(self, env: SearchRescueGridWorld, agent: Agent):
        """
        Creates a new search and rescue linear reward vector.
        :param SearchRescueGridWorld env: the environment with the features from which to create the reward features.
        :param Agent agent: the agent for which to create the reward vector.
        """
        reward_features: List[LinearRewardFeature] = []

        options = env.agent_options[agent.name]

        if options.dist_to_vic_feature:
            # inverse distance to victim reward, i.e., 1 - dist_to_vic
            d2v = env.get_dist_to_vic_feature(agent, key=True)
            r_d2v = NumericLinearRewardFeature(DIST_TO_VIC_FEATURE.title(), d2v, normalize_factor=-1, const_sum=1)
            reward_features.append(r_d2v)

        if options.dist_to_help_feature:
            # inverse distance to help request location, i.e., 1 - dist_to_help
            d2h = env.get_dist_to_help_feature(agent, key=True)
            r_d2h = NumericLinearRewardFeature(DIST_TO_HELP_FEATURE.title(), d2h, normalize_factor=-1, const_sum=1)
            reward_features.append(r_d2h)

        if env.vics_cleared_feature:
            # num victims cleared reward feature
            r_goal = NumericLinearRewardFeature(VICS_CLEARED_FEATURE.title(), env.get_vics_cleared_feature(key=True))
            reward_features.append(r_goal)

        # action execution reward features
        if options.search_action:
            search_action = agent.find_action({'action': SEARCH_ACTION})
            r_search = ActionLinearRewardFeature(SEARCH_ACTION.title(), agent, search_action)
            reward_features.append(r_search)

        if options.triage_action:
            triage_action = agent.find_action({'action': TRIAGE_ACTION})
            r_triage = ActionLinearRewardFeature(TRIAGE_ACTION.title(), agent, triage_action)
            reward_features.append(r_triage)

        evacuate_action = agent.find_action({'action': EVACUATE_ACTION})
        r_evacuate = ActionLinearRewardFeature(EVACUATE_ACTION.title(), agent, evacuate_action)
        reward_features.append(r_evacuate)

        if options.noop_action:
            wait_action = agent.find_action({'action': NOOP_ACTION})
            r_wait = ActionLinearRewardFeature(NOOP_ACTION.title(), agent, wait_action)
            reward_features.append(r_wait)

        if options.call_action:
            call_action = agent.find_action({'action': CALL_ACTION})
            r_call = ActionLinearRewardFeature(CALL_ACTION.title(), agent, call_action)
            reward_features.append(r_call)

        super().__init__(reward_features)
