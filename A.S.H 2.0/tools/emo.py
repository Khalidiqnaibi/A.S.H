from langchain.tools import tool

emo = None

class emos: 
    def __init__(
        self, happy=0, sad=0, angry=0, fear=0, surprise=0, disgust=0, love=0, trust=0,
        anticipation=0, excitement=0, boredom=0, anxiety=0, confidence=0, frustration=0,
        relief=0, pride=0, shame=0, guilt=0, envy=0, jealousy=0, optimism=0, pessimism=0, curiosity=0
    ):
        self.happy = happy
        self.sad = sad
        self.angry = angry
        self.fear = fear
        self.surprise = surprise
        self.disgust = disgust
        self.love = love
        self.trust = trust
        self.anticipation = anticipation
        self.excitement = excitement
        self.boredom = boredom
        self.anxiety = anxiety
        self.confidence = confidence
        self.frustration = frustration
        self.relief = relief
        self.pride = pride
        self.shame = shame
        self.guilt = guilt
        self.envy = envy
        self.jealousy = jealousy
        self.optimism = optimism
        self.pessimism = pessimism
        self.curiosity = curiosity

@tool
def init_emo() -> str:
    """Initialize the global emotion state."""
    global emo
    emo = emos()
    return "Emotion state initialized."

@tool
def get_emo() -> str:
    """Get the current emotion state as a string."""
    if emo is None:
        return "Emotion state not initialized."
    return emo_to_string()

@tool
def set_emo(new_emo: dict) -> str:
    """Set the emotion state using a dictionary of values."""
    global emo
    emo = emos(**new_emo)
    return "Emotion state updated."

@tool
def update_emo(emotion: str, value: int) -> str:
    """Update a specific emotion by adding a value."""
    if emo is None:
        return "Emotion state not initialized."
    if hasattr(emo, emotion):
        current_value = getattr(emo, emotion)
        setattr(emo, emotion, current_value + value)
        return f"{emotion} updated to {getattr(emo, emotion)}."
    else:
        return f"Emotion '{emotion}' not found."

@tool
def reset_emo() -> str:
    """Reset all emotions to zero."""
    global emo
    emo = emos()
    return "Emotion state reset."

@tool
def emo_to_string() -> str:
    """Return the emotion state as a formatted string."""
    if emo is None:
        return "Emotion state not initialized."
    return (
        f"happy: {emo.happy}, sad: {emo.sad}, angry: {emo.angry}, fear: {emo.fear}, "
        f"surprise: {emo.surprise}, disgust: {emo.disgust}, love: {emo.love}, trust: {emo.trust}, "
        f"anticipation: {emo.anticipation}, excitement: {emo.excitement}, boredom: {emo.boredom}, "
        f"anxiety: {emo.anxiety}, confidence: {emo.confidence}, frustration: {emo.frustration}, "
        f"relief: {emo.relief}, pride: {emo.pride}, shame: {emo.shame}, guilt: {emo.guilt}, "
        f"envy: {emo.envy}, jealousy: {emo.jealousy}, optimism: {emo.optimism}, "
        f"pessimism: {emo.pessimism}, curiosity: {emo.curiosity}"
    )
