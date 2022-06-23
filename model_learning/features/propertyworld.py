from model_learning.features.linear import *
from model_learning.environments.property_gridworld import *

GOAL_FEATURE = 'g'
VISIT_FEATURE = 'v'
MOVEMENT = ['right', 'left', 'up', 'down']


class PropertyActionComparisonLinearRewardFeature(LinearRewardFeature):
    """
    Represents a reward feature that returns `1` under a property-action pair and `0` otherwise.
    """

    def __init__(self, name: str, agent_name: str, env: PropertyGridWorld,
                 action_value: str, property_value: str, property_value_next: str,
                 comparison: str):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param Agent agent:
        :param PropertyGridWorld env: the PsychSim world capable of retrieving the feature's value given a state.
        :param str action_key: the named action key associated with this feature.
        :param str or int or float action_value: the value to be compared against the feature to determine its truth (boolean) value.
        :param str property_key: the named property key associated with this feature.
        :param int property_value: the value to be compared against the feature to determine its truth (boolean) value.
        :param str comparison: the comparison to be performed, one of `{'==', '>', '<'}`.
        """
        super().__init__(name)
        self.env = env
        self.world = env.world
        self.action_key = actionKey(agent_name)
        self.action_value = action_value
        self.property_key = get_property_features()
        self.property_value = property_value
        self.property_value_next = property_value_next
        self.locations = list(range(env.width*env.height))

        assert comparison in KeyedPlane.COMPARISON_MAP, \
            f'Invalid comparison provided: {comparison}; valid: {KeyedPlane.COMPARISON_MAP}'
        self.comparison = KeyedPlane.COMPARISON_MAP.index(comparison)
        self.comp_func = COMPARISON_OPS[comparison]

    def get_value(self, state: State) -> float:
        # TODO # not valid for property-action pair
        # collects feature value distribution and returns weighted sum
        dist = np.array(
            [[float(self.comp_func(self.world.float2value(self.action_key, kv[self.action_key]), self.action_value)), p]
             for kv, p in state.distributions[state.keyMap[self.action_key]].items()])
        return dist[:, 0].dot(dist[:, 1]) * self.normalize_factor

    def set_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        rwd_key = rewardKey(agent.name)
        x, y = self.env.get_location_features(agent)
        property_action_tree = {'if': KeyedPlane(KeyedVector({self.action_key: 1}), self.action_value, self.comparison)}
        property_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.env.width}), self.locations, self.comparison)}
        for i, loc in enumerate(self.locations):
            all_p_idx = self.env.get_possible_p_idx(loc, self.property_value)
            all_next_p_idx = [self.env.get_next_p_idx(p_idx, loc, self.property_value_next) for p_idx in all_p_idx]
            subproperty_tree = {'if': KeyedPlane(KeyedVector({self.property_key: 1}), all_next_p_idx, self.comparison)}
            for j, p_idx in enumerate(all_p_idx):
                subproperty_tree[j] = setToConstantMatrix(rwd_key, 1.)
            subproperty_tree[None] = setToConstantMatrix(rwd_key, 0.)
            property_tree[i] = subproperty_tree
        property_tree[None] = setToConstantMatrix(rwd_key, 0.)
        property_action_tree[True] = property_tree
        property_action_tree[False] = setToConstantMatrix(rwd_key, 0.)
        agent.setReward(makeTree(property_action_tree), weight * self.normalize_factor, model)


class AgentRoles(Agent):
    """
    setup agent roles and the corresponding reward features
    """

    def __init__(self, name: str, roles=None):
        super().__init__(name, world=None)
        if roles is None:
            roles = ['Goal']
        self.roles = roles

    def get_role_reward_vector(self, env: PropertyGridWorld):
        reward_features = []
        rf_weights = []
        if 'Goal' in self.roles:  # scale -1 to 1
            r_goal = NumericLinearRewardFeature(GOAL_FEATURE, stateKey(WORLD, GOAL_FEATURE))
            reward_features.append(r_goal)
            rf_weights.append(1)

            for move in MOVEMENT:
                move_action = self.find_action({'action': move})
                r_move = ActionLinearRewardFeature(move, self, move_action)
                reward_features.append(r_move)
                rf_weights.append(-.01)

        if 'Navigator' in self.roles:  # move
            # search_action = self.find_action({'action': 'search'})
            # r_search_unknown = PropertyActionComparisonLinearRewardFeature(
            #     'search_unknown_empty', self.name, env, search_action, PROPERTY_LIST[0], PROPERTY_LIST[4], '==')
            # reward_features.append(r_search_unknown)
            # rf_weights.append(.05)
            #
            # search_action = self.find_action({'action': 'search'})
            # r_search_unknown = PropertyActionComparisonLinearRewardFeature(
            #     'search_unknown_found', self.name, env, search_action, PROPERTY_LIST[0], PROPERTY_LIST[1], '==')
            # reward_features.append(r_search_unknown)
            # rf_weights.append(.1)

            search_action = self.find_action({'action': 'search'})
            r_search = ActionLinearRewardFeature('search', self, search_action)
            reward_features.append(r_search)
            rf_weights.append(.1)

            visits = env.get_visit_feature(self)
            for loc_i in range(env.width*env.height):
                r_visit = ValueComparisonLinearRewardFeature(
                    VISIT_FEATURE+f'{loc_i}', env.world, visits[loc_i], 2, '==')
                reward_features.append(r_visit)
                rf_weights.append(-.02)

            for move in MOVEMENT:
                move_action = self.find_action({'action': move})
                r_move = ActionLinearRewardFeature(move, self, move_action)
                reward_features.append(r_move)
                rf_weights.append(.01)

        if 'SubGoal' in self.roles:  # small reward for sub-goal: rescue when found
            rescue_action = self.find_action({'action': 'rescue'})
            r_rescue_found = PropertyActionComparisonLinearRewardFeature(
                'rescue_found', self.name, env, rescue_action, PROPERTY_LIST[1], PROPERTY_LIST[2], '==')
            reward_features.append(r_rescue_found)
            rf_weights.append(0.1)

            evacuate_action = self.find_action({'action': 'evacuate'})
            r_rescue_found = PropertyActionComparisonLinearRewardFeature(
                'evacuate_ready', self.name, env, evacuate_action, PROPERTY_LIST[2], PROPERTY_LIST[3], '==')
            reward_features.append(r_rescue_found)
            rf_weights.append(0.2)
        return reward_features, rf_weights


class AgentLinearRewardVector(LinearRewardVector):
    """
    same as LinearRewardVector now, automatically set reward
    """

    def __init__(self, agent: Agent, rf: List[LinearRewardFeature],
                 weights: np.ndarray, model: Optional[str] = None):
        super().__init__(rf)
        self.rwd_features = rf
        self.rwd_weights = weights
        self.set_rewards(agent, weights, model)

    # def set_rewards(self, agent: Agent, weights: np.ndarray, model: Optional[str] = None):
    #
    #     assert len(weights) == len(self.rwd_features), \
    #         'Provided weight vector\'s dimension does not match reward features'
    #
    #     agent.setAttribute('R', {}, model)  # make sure to clear agent's reward function
    #     for i, weight in enumerate(weights):
    #         self.rwd_features[i].set_reward(agent, weight, model)
