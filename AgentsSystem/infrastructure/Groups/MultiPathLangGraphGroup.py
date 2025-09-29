from .LangGraphGroup import LangGraphGroup
from typing import Dict, Any, List, Callable
from ...domain.interfaces.IGroup import IGroup

class MultiPathLangGraphGroup(IGroup):


    def __init__(self, state_paths: Dict[str, List[str]], conditions: Dict[str, Callable[[Dict[str, Any]], str]]):
      
        super().__init__()
        self.state_paths = state_paths
        self.conditions = conditions

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        current_state = state.get("current_state")

        if not current_state:
            raise ValueError("current_state not defined in state")

        possible_paths = self.state_paths.get(current_state, [])
        if not possible_paths:
            raise ValueError(f"No paths defined for state: {current_state}")

        if current_state in self.conditions:
            chosen_path = self.conditions[current_state](state)
            if chosen_path not in possible_paths:
                raise ValueError(f"Path {chosen_path} not valid for state {current_state}")
        else:
            chosen_path = possible_paths[0]

        state["next_path"] = chosen_path

        return super().run(state)
