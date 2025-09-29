from ...domain.interfaces.IToolKit import IToolKit
from typing import Callable

class ToolKit(IToolKit):
    def __init__(self):
        self._tools = []

    def register(self, tool:Callable):
        self._tools.append(tool)

    def get_all_tools(self):
        return self._tools