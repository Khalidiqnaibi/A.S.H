from abc import ABC, abstractmethod
from typing import Callable

class IToolKit(ABC):
    @abstractmethod
    def register(self, tool: Callable):
        pass
    # @abstractmethod
    # def get_tool(key: str):
    #     pass
    @abstractmethod
    def get_all_tools(self) -> list:
        pass
