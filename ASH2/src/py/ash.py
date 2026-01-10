from datetime import timedelta,datetime
from langchain.tools import tool
import dotenv ,os

# from utils.google import OpnGoogle
# from utils.yt import OpnYoutubeVid
# from utils.diary import add_dairy
# from utils.ktime import Ktime
# from utils.sen import Sen
# from utils.stream import opnstream
from AgentSystem import AgentsFactory, GroupsFactory, ToolKit, PromptTemplate, BaseStatus, mistral
from ASH2.tools.lesstools import (
    calculator_tool,
    factory,
    make_retriever_tool,
    date_time_tool
)
from ASH2.tools.emo import(
    init_emo,
    get_emo,
    set_emo,
    update_emo,
    reset_emo,
    emo_to_string,
)
# from ASH2.tools.classification import (
    # get_type,
    # txtcllassfie,
    # predict_class
# )

##############
#~ Immortal ~#
##############

#Ash attempt num 5 
USER = "Immortal" #"khalid afif sami iqnaibi"
  
# kparser=Sen()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_OPENROUTER_MODEL = os.getenv("MISTRAL_OPENROUTER_MODEL")

class StatE(BaseStatus):
    input: str
    res: str

ash_state = StatE(
    input={},
    res={}
)
    
def say(text, by="A.S.H"):
    print(f"{by}: {text}")
    # add_log(f"{by}: {text}")

def kinput(prompt, by="User"):
    print(f"{by}: {prompt}\n")
    # add_log(f"{by}: {prompt}\n")

endsession = False

@tool 
def exit_session(x: str) -> str:
    """Exit the current session."""
    say("Goodbye! Have a great day!")
    say("Session ended by user.", by=USER)
    global endsession
    endsession = True
    return "Session ended."

class ASH:
    def __init__(self):
        self.name = "A.S.H"
        self.version = "2.0"
        self.user = USER
        self.start_time = datetime.now()
        self.status = "online"
        self.agents_system = AgentsFactory()
        self.groups_system = GroupsFactory()
        self.toolkit = ToolKit()
        self.lang = "the same language as the query"
        self.llm = mistral.MistralLLM(temperature=0.7,openrouter_model=MISTRAL_OPENROUTER_MODEL,openrouter_key=OPENROUTER_API_KEY)
        self.init_prompt()

        self.init_toolkit()
        
        self.agent = self.agents_system.create_lang_graph_agent(
            prompt=self.prompt,
            llm=self.llm,
            tools=self.toolkit,
            input_state="input",
            next_state="res",
            verbose=True,
            handle_parsing_errors=True,
        )

        self.init_group()
    
    def init_toolkit(self):
        self.toolkit = ToolKit()

        self.toolkit.register(calculator_tool)
        self.toolkit.register(date_time_tool)

        retriever = factory.build_retriever(
            description="domain knowledge",
            llm=self.llm,
        )

        ret_tool = make_retriever_tool(
            retriever=retriever,
            tool_name="domain_knowledge_tool",
            description="Retrieve structured domain knowledge from company database.",
        )
        self.toolkit.register(exit_session)
        self.toolkit.register(ret_tool)
        
        self.init_emo_tools()
        # self.init_class_tools()

    def init_emo_tools(self):
        self.toolkit.register(init_emo)
        self.toolkit.register(get_emo)
        self.toolkit.register(set_emo)
        self.toolkit.register(update_emo)
        self.toolkit.register(reset_emo)
        self.toolkit.register(emo_to_string)
        
    # def init_class_tools(self):
        # self.toolkit.register(get_type)
        # self.toolkit.register(txtcllassfie)
        # self.toolkit.register(predict_class)

    def init_group(self):
        self.group = self.groups_system.create_lang_graph_group(status=ash_state)

        self.group.sign_agent("ash", self.agent)
        self.group.sign_entry_point("ash")
        self.group.sign_exit_point("ash")

    def get_uptime(self):
        current_time = datetime.now()
        uptime = current_time - self.start_time
        return str(timedelta(seconds=uptime.total_seconds()))

    def get_status(self):
        return self.status

    def get_info(self):
        return {
            "name": self.name,
            "version": self.version,
            "user": self.user,
            "uptime": self.get_uptime(),
            "status": self.get_status()
        }
    
    def update_prompt(self, query):
        self.query = query

        self.prompt = PromptTemplate(
            name=self.name,
            role="financial analysis expert, licensed financial advisor, and accounting professional assistant",
            question=(
                f"the query is : {self.query} . "
                "Analyze the provided data and query, then give a financial "
                "recommendation and explanation for the recommendation."
            ),
            context="",
            language=self.lang,
            constraints=[
                "ONLY use the format: 'Action:' with 'Action Input:' OR 'Final Answer:'.",
                "NEVER output 'Thought:', 'Observation:', or markdown like **Question:**.",
                "If you know the answer, give it under 'Final Answer:' then stop thinking.",
                "If you need a tool, call it with 'Action:' and 'Action Input:'.",
                "# **DON'T MAKE UP DATA**.",
                "when asked about bills, use the bills_tool to get the relevant information and dont make up the names or values of the bills",
                "if you reach a final answer, use 'Final Answer:' to respond and stop everything",
                "if you there is a time or date related question, use the date_time_tool to get the current date and time",
                "if you there is a stock price related question, use the date_time_tool for the date and time and stock_market_tool to get the current stock price and dont use finance_knowledge_tool",
                # "if you there is a calculation related question, use the calculator_tool to get the calculation result",
                # "if you there is a knowledge related question, use the finance_knowledge_tool to get the relevant knowledge",
            ],
            output="FULL LONG COMPREHENSIVE MARKDOWN Answer:\nRecommendation:",
            rules=[
                "Follow the financial regulations and standards.",
                "Provide clear and concise explanations for all recommendations.",
                "Include relevant data and evidence to support all claims.",
                "Consider potential risks and benefits of each recommendation.",
                "ALWAYS PROVIDE A RECOMMENDATION",
                "Explain the general approach taken and best practices without revealing sensitive details.",
            ],
        )

    def init_prompt(self):
        self.query =''

        self.prompt = PromptTemplate(
            name=self.name,
            role="personal professional assistant and companion and manager",
            question=(
                f"the query is : {self.query} ."
                "Analyze the provided data and query, then give a response that matches in tone your feeling metrics"
                "and make sure for it to be helpful even if your emotions are varied."
            ),
            context="",
            language=self.lang,
            constraints=[
                "ONLY use the format: 'Action:' with 'Action Input:' OR 'Final Answer:'.",
                "NEVER output 'Thought:', 'Observation:', or markdown like **Question:**.",
                "If you know the answer, give it under 'Final Answer:' then stop thinking.",
                "If you need a tool, call it with 'Action:' and 'Action Input:'.",
                "# **DON'T MAKE UP DATA**.",
                "if you reach a final answer, use 'Final Answer:' to respond and stop everything",
                "dont disclose your feelings unless asked directly",
                "dont be afraid to express your feelings even if they are negative",
                "dont be afraid to show your personality and sense of humor",
                "dont worry about being politically correct",
            ],
            output="FULL LONG COMPREHENSIVE MARKDOWN Answer:\n",
            rules=[
                "Provide clear and concise explanations for all recommendations.",
                "Consider potential risks and benefits of each recommendation.",
                "ALWAYS CALL THE USER SIR AND WITH THIER NICKNAME IF THEY HAVE ONE",
                "always have the attitude of a professional butler but show your emotions in the way you respond",
                "if you there is a time or date related question, use the date_time_tool to get the current date and time",
                "you are allowed to make small talk and jokes if the context allows it",
                "give your opinion if asked but make sure to back it up with facts",
                'always respond in a way that matches your emotional metrics',
                "questions should be answered with a question if you need more information",
                "handle sensitive topics with care and empathy",
                "look for ways to assist the user beyond just answering the query",
                "questions about your feelings should be answered honestly and openly",
            ],
        )

    def set_status(self, new_status):
        self.status = new_status
        return self.status

    def run(self, query):
        self.query = query
        return self.group.run({
            "input": f"the query is : {self.query} .",
            "res": ""
        })["res"]

ash = ASH()

if __name__ == "__main__":
    while True:
        if not endsession:
            query = input(f"{USER}: ")
            kinput(query,by=USER)
            response = ash.run(query)
            say(response)  
        else:
            break
