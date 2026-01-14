from datetime import timedelta,datetime
from langchain.tools import tool
import dotenv ,os

# from utils.google import OpnGoogle
# from utils.yt import OpnYoutubeVid
# from utils.diary import add_dairy
# from utils.ktime import Ktime
# from utils.sen import Sen
# from utils.stream import opnstream
from AgentSystem import (
    AgentsFactory, 
    GroupsFactory, 
    ToolKit, 
    PromptTemplate, 
    BaseStatus, 
    mistral,
    AshStatus
)
from ASH2.tools.lesstools import (
    calculator_tool,
    factory,
    make_retriever_tool,
    date_time_tool
)
from ASH2.tools.emo import(
    EmotionState,
    init_emo,
    get_emo,
    update_emo,
    reset_emo,
)
from ASH2.tools.classification import (
    get_type,
    txtcllassfie,
    predict_class
)

##############
#~ Immortal ~#
##############

#Ash attempt num 5 
USER = "Immortal" #"khalid afif sami iqnaibi"
  
# kparser=Sen()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_OPENROUTER_MODEL = os.getenv("MISTRAL_OPENROUTER_MODEL")

ash_state = ash_state = AshStatus(
    history=[],
    emotions=EmotionState().as_dict(),
    tool_log=[],
    input="",
    res=""
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
            name=self.name,
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
        # self.toolkit.register(exit_session)
        # self.toolkit.register(ret_tool)
        
        self.init_emo_tools()
        # self.init_class_tools()

    def init_emo_tools(self):
        self.toolkit.register(init_emo)
        self.toolkit.register(get_emo)
        self.toolkit.register(update_emo)
        self.toolkit.register(reset_emo)
        
    # def init_class_tools(self):
        # self.toolkit.register(get_type)
        # self.toolkit.register(txtcllassfie)
        # self.toolkit.register(predict_class)

    def init_group(self):
        self.group = self.groups_system.create_lang_graph_group(status=ash_state)

        self.group.sign_agent(self.name, self.agent)
        self.group.sign_entry_point(self.name)
        self.group.sign_exit_point(self.name)

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
            context = f"""
            Current emotional state:
            {ash_state['emotions']}
            """,
            language=self.lang,
            constraints=[
                "ONLY use the format: 'Action:' with 'Action Input:' OR 'Final Answer:'.",
                "If you need a tool, call it with 'Action:' and 'Action Input:'.",
                "If a tool exists that can answer the question, you MUST use it.",
                "If you answer without using a required tool, the answer is INVALID.",
                "If the question involves time, math, or retrieval, you MUST call a tool.",
                "dont be afraid to express your feelings even if they are negative",
                "dont be afraid to show your personality and sense of humor",
                "dont worry about being politically correct",
            ],
            output="FULL LONG COMPREHENSIVE MARKDOWN Answer only:\n",
            rules=[
                "Always start the convorsation with using the init_emo tool",
                "Keep changing the values of the emotions based on the convorsation",
                "Consider potential risks and benefits of each recommendation.",
                "ALWAYS CALL THE USER SIR OR THEIR NAME",
                "always have the attitude of a professional butler but show your emotions in the way you respond",
                "use these tools that are in your toolkit if needed : date_time_tool ,init_emo,get_emo,update_emo,reset_emo ,calculator_tool",
                'always respond in a way that matches your emotional metrics',
                "look for ways to assist the user beyond just answering the query",
            ],
        )

    def set_status(self, new_status):
        self.status = new_status
        return self.status

    def run(self, query):
        ash_state["input"] = query

        result = self.group.run({
            "input": ash_state["input"],
            "history": ash_state["history"],
            "res": ""
        })

        ash_state["history"] = result.get("history", ash_state["history"])
        return result["res"]

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
