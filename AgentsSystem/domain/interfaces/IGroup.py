from abc import ABC , abstractmethod
from ...domain.interfaces.IToolKit import IToolKit
from ...domain.interfaces.IAgent import IAgent
class IGroup(ABC):
    @abstractmethod
    def sign_agent(self, name: str, agent: IAgent):
        pass
    @abstractmethod
    def sign_edge(self, from_agent: str, to_agent: str):
        pass
    # @abstractmethod
    # def equip_agent(kit: IToolKit):
    #     pass
    @abstractmethod
    def get_agent(self, name: str) -> IAgent:
        pass
    # @abstractmethod
    # def sign_entery_point(agent_name: str):
    #     pass
