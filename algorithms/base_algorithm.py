from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):

    @abstractmethod
    def act(
        self,
        obs,
        shared_obs=None,
        training: bool = True,
        action_masks=None,
    ):
        """
        Return actions for all agents.
        """
        pass

    def observe(self, **transition):
        pass

    def update(self, **kwargs):
        return {}

    def save(self, path, **kwargs):
        pass

    def load(self, path, **kwargs):
        pass
