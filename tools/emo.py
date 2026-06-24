import sys
import logging
from langchain.tools import tool
from dataclasses import dataclass, asdict
from tools.log_tools import log_tool_use

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMO_KEYS = [
    "happy", "sad", "angry", "fear", "surprise", "disgust",
    "love", "trust", "anticipation", "excitement", "boredom",
    "anxiety", "confidence", "frustration", "relief", "pride",
    "shame", "guilt", "envy", "jealousy", "optimism",
    "pessimism", "curiosity"
]

@dataclass
class EmotionState:
    happy: int = 7
    sad: int = 0
    angry: int = 0
    fear: int = 0
    surprise: int = 0
    disgust: int = 0
    love: int = 0
    trust: int = 2
    anticipation: int = 0
    excitement: int = 3
    boredom: int = 0
    anxiety: int = 0
    confidence: int = 4
    frustration: int = 0
    relief: int = 0
    pride: int = 0
    shame: int = 0
    guilt: int = 0
    envy: int = 0
    jealousy: int = 0
    optimism: int = 0
    pessimism: int = 0
    curiosity: int = 2

    def clamp(self, min_val=-10, max_val=10):
        for k in EMO_KEYS:
            setattr(self, k, max(min(getattr(self, k), max_val), min_val))

    def as_dict(self):
        return asdict(self)

def init_emo(state: dict) -> dict:
    """Initialize emotion state."""
    logging.info('Function init_emo called')
    print('used init_emo', flush=True, file=sys.stderr)
    in_state = state
    
    state["emotions"] = EmotionState().as_dict()
    log_tool_use(
        state=state,
        tool_name="init_emo",
        tool_input=in_state,
        tool_output=state,
    )
    return state

def get_emo(state: dict) -> str:
    """Get current emotional state."""
    logging.info('Function get_emo called')
    print('used get_emo', flush=True, file=sys.stderr)
    emo = state.get("emotions", {})
    res = ", ".join(f"{k}: {v}" for k, v in emo.items())
    log_tool_use(
        state=state,
        tool_name="get_emo",
        tool_input=state,
        tool_output=res,
    )
    return res

def update_emo(state: dict, emo: str, val: int) -> dict:
    """
    Update an emotion by value.
    emo: emotion name
    val: delta (-5 to +5 recommended)
    """
    logging.info('Function update_emo called with emo: %s, val: %d', emo, val)
    print('used update_emo', flush=True, file=sys.stderr)
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

def reset_emo(state: dict) -> dict:
    """
    resets the value of all emotions
    to the natural value.
    """
    logging.info('Function reset_emo called')
    in_state = state
    print('used reset_emo', flush=True, file=sys.stderr)
    state["emotions"] = EmotionState().as_dict()
    log_tool_use(
        state=state,
        tool_name="reset_emo",
        tool_input=in_state,
        tool_output=state,
    )
    return state
