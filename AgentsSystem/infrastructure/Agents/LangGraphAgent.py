
from ...domain.interfaces.IAgent import IAgent
from ...domain.entities.PromptTemplate import PromptTemplate
from typing import Dict, Any
from ...domain.interfaces.IToolKit import IToolKit
from langchain.agents import initialize_agent, AgentType
from ...domain.entities.ToolKit import ToolKit

class LangGrpahAgent(IAgent):
    def __init__(self,
                llm,
                prompt: PromptTemplate,
                input_state: str,
                next_state: str,
                agent_type,
                verbose: bool,
                tools: IToolKit = ToolKit(),
                handle_parsing_errors: bool = True
    ):
        
        self._command = prompt.get_command()
        self._input_state = input_state
        self._next_state = next_state
        self._tools = tools
        self._llm = llm
        self._agent_type = agent_type
        self._verbose = verbose
        self._handle_parsing_errors = handle_parsing_errors

    def prepare(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = self._agent.invoke(state[self._input_state])
        state[self._next_state] = response['output']
        return state

    def build_agent(self):
        self._agent = initialize_agent(
            tools = self._tools.get_all_tools(),
            llm = self._llm,
            agent = self._agent_type,
            verbose = self._verbose,
            handle_parsing_errors = self._handle_parsing_errors
        )