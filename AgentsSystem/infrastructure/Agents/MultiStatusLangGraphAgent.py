from ...domain.interfaces.IAgent import IAgent
from ...domain.entities.PromptTemplate import PromptTemplate
from typing import Dict, Any, List
from ...domain.interfaces.IToolKit import IToolKit
from langchain.agents import initialize_agent, AgentType


class MultiStatusLangGraphAgent(IAgent):
    """
    Extended Agent that supports multiple input and output states.
    """

    def __init__(self,
                 prompt: PromptTemplate,
                 input_states: List[str],
                 output_states: List[str],
                 tools: IToolKit,
                 llm,
                 agent_type,
                 verbose: bool):
        
        self._command = prompt.get_command()
        self._input_states = input_states
        self._output_states = output_states
        self._tools = tools
        self._llm = llm
        self._agent_type = agent_type
        self._verbose = verbose
        self._agent = None

    def build_agent(self):
        self._agent = initialize_agent(
            tools=self._tools.get_all_tools(),
            llm=self._llm,
            agent=self._agent_type,
            verbose=self._verbose,
        )

    def prepare(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects all input states, passes them to the agent,
        and distributes the response across multiple output states.
        """
        if not self._agent:
            raise RuntimeError("Agent not built. Call build_agent() first.")

        inputs = {k: state[k] for k in self._input_states if k in state}

        response = self._agent.run(inputs)

        if isinstance(response, dict):
            for idx, key in enumerate(self._output_states):
                if key in response:
                    state[key] = response[key]
                elif idx == 0:
                    state[key] = response  
        else:
            state[self._output_states[0]] = response

        return state
