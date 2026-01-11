from langchain.tools import tool
from dataclasses import dataclass, asdict
from ASH2.tools.log_tools import log_tool_use

EMO_KEYS = [
    "happy", "sad", "angry", "fear", "surprise", "disgust",
    "love", "trust", "anticipation", "excitement", "boredom",
    "anxiety", "confidence", "frustration", "relief", "pride",
    "shame", "guilt", "envy", "jealousy", "optimism",
    "pessimism", "curiosity"
]

@dataclass
class EmotionState:
    happy: int = 0
    sad: int = 0
    angry: int = 0
    fear: int = 0
    surprise: int = 0
    disgust: int = 0
    love: int = 0
    trust: int = 0
    anticipation: int = 0
    excitement: int = 0
    boredom: int = 0
    anxiety: int = 0
    confidence: int = 0
    frustration: int = 0
    relief: int = 0
    pride: int = 0
    shame: int = 0
    guilt: int = 0
    envy: int = 0
    jealousy: int = 0
    optimism: int = 0
    pessimism: int = 0
    curiosity: int = 0

    def clamp(self, min_val=-10, max_val=10):
        for k in EMO_KEYS:
            setattr(self, k, max(min(getattr(self, k), max_val), min_val))

    def as_dict(self):
        return asdict(self)

@tool
def init_emo(state: dict) -> dict:
    """Initialize emotion state."""
    print('used init_emo')
    in_state = state
    
    state["emotions"] = EmotionState().as_dict()
    log_tool_use(
        state=state,
        tool_name="init_emo",
        tool_input=in_state,
        tool_output=state,
    )
    return state

@tool
def get_emo(state: dict) -> str:
    """Get current emotional state."""
    print('used get_emo')
    emo = state.get("emotions", {})
    res = ", ".join(f"{k}: {v}" for k, v in emo.items())
    log_tool_use(
        state=state,
        tool_name="get_emo",
        tool_input=state,
        tool_output=res,
    )
    return res


@tool
def update_emo(state: dict, emo: str, val: int) -> dict:
    """
    Update an emotion by value.
    emo: emotion name
    val: delta (-5 to +5 recommended)
    """
    print('used update_emo')
    inputs = [
        state,
        emo,
        val
    ]
    emotions = EmotionState(**state.get("emotions", {}))

    if not hasattr(emotions, emo):
        return state

    setattr(emotions, emo, getattr(emotions, emo) + val)
    emotions.clamp()

    state["emotions"] = emotions.as_dict()
    log_tool_use(
        state=state,
        tool_name="update_emo",
        tool_input=inputs,
        tool_output=state,
    )
    return state

@tool
def reset_emo(state: dict) -> dict:
    """
    resets the value of all emotions
    to the nautral value.
    """
    in_state = state
    print('used reset_emo')
    state["emotions"] = EmotionState().as_dict()
    log_tool_use(
        state=state,
        tool_name="reset_emo",
        tool_input=in_state,
        tool_output=state,
    )
    return state

