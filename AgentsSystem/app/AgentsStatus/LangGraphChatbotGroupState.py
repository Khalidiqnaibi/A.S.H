from typing import TypedDict

# --- Define the state type ---
class LangGraphChatbotGroupState(TypedDict, total=False):
    question: str
    plan: str
    answer: str
