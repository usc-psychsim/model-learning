import json
from typing import NamedTuple, Optional, Dict

from model_learning import ModelsDistributions
from model_learning.environments.search_rescue_gridworld import AgentProfile
from psychsim.probability import Distribution

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class AgentProfiles(Dict[str, Dict[str, AgentProfile]]):
    """
    Represents a set of agent profiles for different roles in the search-and-rescue domain.
    """

    def save(self, file_path: str):
        """
        Saves the profiles to a Json format file.
        :param str file_path: the path to the file in which to save the profiles.
        """
        profiles = {role: {n: p._asdict() for n, p in self[role].items()} for role in self.keys()}
        with open(file_path, 'w') as fp:
            json.dump(profiles, fp, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """
        Loads a set of profiles from a file in the Json format.
        :param str file_path: the path to the file from which to load the profiles.
        :return: the loaded task model.
        """
        with open(file_path, 'r') as fp:
            profiles = json.load(fp)
            profiles = {role: {n: AgentProfile(**p) for n, p in profiles[role].items()}
                        for role in profiles.keys()}
            return AgentProfiles(profiles)


class AgentConfig(NamedTuple):
    """
    Configuration for an agent in the search-and-rescue domain.
    """
    profile: str
    mental_models: Optional[Dict[str, Dict[str, float]]] = None


class TeamConfig(Dict[str, AgentConfig]):
    """
    Configuration for a team of agents in the search-and-rescue domain, including the agent's true rewards and models,
    and mental models of the other agents in the team.
    """

    def check_profiles(self, profiles: AgentProfiles):
        """
        Verifies that all agent profiles referred to by this team configuration exist in the given profile set.
        Raises an assertion error whenever a profile is not defined for a role.
        :param AgentProfiles profiles: the set of existing agent profiles for each role.
        """
        for role, ag_conf in self.items():
            assert role in profiles, f'No profiles found for role: {role}'
            if ag_conf.mental_models is not None:
                for other, models in ag_conf.mental_models.items():
                    assert other in profiles, f'No profiles found for role: {role}'
                    for model in models.keys():
                        assert model in profiles[other], f'No profile found with name {model} for role {other}'

    def get_all_model_profiles(self, role: str, profiles: AgentProfiles) -> Dict[str, AgentProfile]:
        """
        Gets all model profiles for a given agent by searching in the mental model definition of all other agents.
        :param str role: the agent's role for which to retrieve all the model profiles..
        :param AgentProfiles profiles: the set of existing agent profiles for each role.
        :return: a dictionary containing an agent profile for each model.
        """
        assert role in self, f'Could not find role {role}'

        ag_models: Dict[str, AgentProfile] = {}
        for ag_conf in self.values():
            if ag_conf.mental_models is not None:
                for other, models in ag_conf.mental_models.items():
                    if other == role:
                        ag_models.update({f'{role}_{model}': profiles[other][model] for model in models.keys()})

        return ag_models

    def get_models_distributions(self) -> ModelsDistributions:
        """
        Creates distributions over other agents' models for each agent as specified in this configuration.
        :rtype:ModelsDistributions
        :return: a dictionary containing a PsychSim distribution over other agents' models for each agent.
        """
        return {
            role: {other_ag: Distribution({f'{other_ag}_{model}': prob for model, prob in models_probs.items()})
                   for other_ag, models_probs in ag_conf.mental_models.items() if ag_conf.mental_models is not None}
            for role, ag_conf in self.items()
        }

    def save(self, file_path: str):
        """
        Saves the team configuration to a Json format file.
        :param str file_path: the path to the file in which to save the task model.
        """
        with open(file_path, 'w') as fp:
            json.dump({n: ao._asdict() for n, ao in self.items()}, fp, indent=4)

    @classmethod
    def load(cls, file_path: str):
        """
        Loads a team configuration from a file in the Json format.
        :param str file_path: the path to the file from which to load the team config.
        :return: the loaded configuration.
        """
        with open(file_path, 'r') as fp:
            return TeamConfig({n: AgentConfig(**ao) for n, ao in json.load(fp).items()})
