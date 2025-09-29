from abc import ABC , abstractmethod
from ...domain.entities.PromptTemplate import PromptTemplate
from ...domain.interfaces.IToolKit import IToolKit
from typing import Dict, Any
class IAgent(ABC):
    # @abstractmethod
    # def add_tool_kit(kit: IToolKit):
    #     pass

    @abstractmethod
    def prepare(self, state) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def build_agent(self):
        pass