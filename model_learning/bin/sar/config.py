import json
from typing import NamedTuple, Optional, Dict, List

from model_learning.environments.search_rescue_gridworld import AgentProfile

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
    models: Optional[List[str]] = None
    mental_models: Optional[Dict[str, Dict[str, float]]] = None


class TeamConfig(Dict[str, AgentConfig]):
    """
    Configuration for a team of agents in the search-and-rescue domain, including the agent's true rewards and models,
    and mental models of the other agents in the team.
    """

    def check_profiles(self, profiles: AgentProfiles):
        """
        Verifies that all agent profiles referred to by this team configuration exist in the given set.
        Raises an assertion error whenever a profile is not defined for a role.
        :param AgentProfiles profiles: the set of existing agent profiles for each role.
        """
        for role, ag_conf in self.items():
            assert role in profiles, f'No profiles found for role: {role}'
            if ag_conf.models is not None:
                for model in ag_conf.models:
                    assert model in profiles[role], f'No profile found with model {model} for role {role}'
            if ag_conf.mental_models is not None:
                for other, models in ag_conf.mental_models.items():
                    assert other in profiles, f'No profiles found for role: {role}'
                    for model in models.keys():
                        assert model in profiles[other], f'No profile found with model {model} for role {other}'

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
        :return: the loaded task model.
        """
        with open(file_path, 'r') as fp:
            return TeamConfig({n: AgentConfig(**ao) for n, ao in json.load(fp).items()})
