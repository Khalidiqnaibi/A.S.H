from ...infrastructure.Agents.LangGraphAgent import LangGrpahAgent
from ...domain.entities.PromptTemplate import PromptTemplate
from ...domain.interfaces.IToolKit import IToolKit

class AgentsFactory:
    def __init__(self):
        pass

    def create_lang_graph_agent(
            self,
            llm,
            prompt: PromptTemplate, 
            input_state: str,
            next_state: str,
            tools: IToolKit, 
            agent_type,
            verbose: bool,
            handle_parsing_errors: bool = True
            ):
        
        return LangGrpahAgent(
            prompt = prompt,
            input_state = input_state,
            next_state = next_state,
            tools = tools,
            llm = llm,
            agent_type = agent_type,
            verbose = verbose,
            handle_parsing_errors = handle_parsing_errors
        )