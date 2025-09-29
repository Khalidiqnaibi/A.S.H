from ...infrastructure.Groups.LangGraphGroup import LangGraphGroup
from typing import Dict, Any
from ...app.AgentsStatus.BaseStatus import BaseStatus
class GroupsFactory:
    def __init__(self):
        pass

    def create_lang_graph_group(
        self,
        status: BaseStatus
        ):
        
        return LangGraphGroup(
            status = status
        )