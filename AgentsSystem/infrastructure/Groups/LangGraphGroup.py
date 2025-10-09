from ...domain.interfaces.IGroup import IGroup
from langgraph.graph import StateGraph #, End
from ...app.AgentsStatus.LangGraphChatbotGroupState import LangGraphChatbotGroupState
from typing import Dict, Any, TypedDict
from ...domain.interfaces.IAgent import IAgent
from ...domain.interfaces.IToolKit import IToolKit
from ...app.AgentsStatus.BaseStatus import BaseStatus

class LangGraphGroup(IGroup):
    def __init__(self, status: BaseStatus):
        """
        Initialize with separate substates for file and folder agents
        """
        self._registry: Dict[str, IAgent] = {}
        self._graph = StateGraph(type(status))
        self._status = status

    def sign_agent(self, name: str, agent: IAgent):
        agent.build_agent()
        self._graph.add_node(name, agent.prepare)
        self._registry[name] = agent

    def sign_edge(self, from_agent: str, to_agent: str):
        self._graph.add_edge(from_agent, to_agent)

    def sign_entry_point(self, agent_name: str):
        self._graph.set_entry_point(agent_name)

    def sign_exit_point(self, agent_name: str):
        self._graph.set_finish_point(agent_name)

    def get_agent(self, name: str) -> IAgent:
        return self._registry[name]

    def run(self, prompt: str) -> Any:
        # Ensure the graph gets what it expects
        if hasattr(self._status, "dict"):
            context = self._status.dict()
            context["input"] = prompt
        else:
            context = dict(self._status)
            context["input"] = prompt

        compiled_graph = self._graph.compile()
        result = compiled_graph.invoke(context)
        return result


