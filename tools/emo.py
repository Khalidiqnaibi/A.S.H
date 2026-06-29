# tools/emo_v2.py
"""
Deterministic, multi-signal emotional simulation for ASH.

This replaces the old emo.py design, where the LLM itself decided when and
how much to shift emotions via an `update_emo(emo, delta)` tool call. That's
black-box emotion guessing: the model could move "frustration" by however
much it felt like, for whatever reason it felt like, with no consistent
relationship to what actually happened in the conversation.

Here, emotional state is computed automatically every turn from real
telemetry -- nothing the LLM says or calls changes it directly. The pieces:

  1. TelemetrySignals  -- a snapshot of what actually happened this turn
     (user sentiment, tool success/failure, timing, repetition, grounding).
  2. EmotionEngine      -- a stateful accumulator: decays old state toward a
     slow-moving "mood" baseline based on elapsed time, then layers on
     deltas from each signal category (sentiment, tool outcome, goal
     alignment / "are we spinning", session context, topic shifts).
  3. ModulationProfile  -- a deterministic, rule-based translation of the
     resulting emotion vector into concrete response parameters (warmth,
     directness, verbosity, energy, patience) plus a short tone label --
     this is what actually gets handed to the LLM rendering step, instead
     of a raw emotion dict for it to interpret however it likes.

Two-layer model (mood vs. emotion), exponential time-decay toward baseline,
streak-scaled reactions to repeated tool failures, and reciprocal (not
mirrored) responses to user sentiment are all explicit design choices aimed
at "human-like" without ever being random or LLM-guessed -- same inputs,
same outputs, every time.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Deque, Dict, List, Optional

# ----------------------------------------------------------------------
# Emotion vocabulary -- unchanged from the old emo.py so any persisted
# state / downstream code keyed on these names keeps working.
# ----------------------------------------------------------------------
EMO_KEYS = [
    "happy", "sad", "angry", "fear", "surprise", "disgust",
    "love", "trust", "anticipation", "excitement", "boredom",
    "anxiety", "confidence", "frustration", "relief", "pride",
    "shame", "guilt", "envy", "jealousy", "optimism",
    "pessimism", "curiosity",
]

# Default temperament -- same starting values as the old EmotionState
# defaults. This is the slow-moving "mood" baseline emotion decays toward.
DEFAULT_MOOD: Dict[str, float] = {
    "happy": 7, "sad": 0, "angry": 0, "fear": 0, "surprise": 0, "disgust": 0,
    "love": 0, "trust": 2, "anticipation": 0, "excitement": 3, "boredom": 0,
    "anxiety": 0, "confidence": 4, "frustration": 0, "relief": 0, "pride": 0,
    "shame": 0, "guilt": 0, "envy": 0, "jealousy": 0, "optimism": 0,
    "pessimism": 0, "curiosity": 2,
}

EMO_MIN, EMO_MAX = 0.0, 10.0


def _clamp(v: float, lo: float = EMO_MIN, hi: float = EMO_MAX) -> float:
    return max(lo, min(hi, v))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ----------------------------------------------------------------------
# 1. Telemetry -- what actually happened this turn.
# ----------------------------------------------------------------------
@dataclass
class TelemetrySignals:
    """Raw, observable facts about a single turn. Build one of these from
    real data each call to ASH.run() -- nothing here is guessed by the LLM."""

    timestamp: float = field(default_factory=time.time)
    query_text: str = ""

    # From classification.sentiment_tool(query) -- label is "POSITIVE",
    # "NEGATIVE", or None/"unknown" if unavailable. score is 0..1 confidence.
    sentiment_label: Optional[str] = None
    sentiment_score: float = 0.0

    # From classify_and_route(query).
    intent: Optional[str] = None
    intent_score: float = 0.0

    # Did a tool run this turn, and did it succeed? None = no tool invoked.
    tool_used: Optional[str] = None
    tool_success: Optional[bool] = None

    # How many memory blocks (core/episodic/entity) came back non-empty --
    # a cheap proxy for "are we actually grounded in something real".
    memory_hits: int = 0


# ----------------------------------------------------------------------
# 2. Modulation -- the deterministic translation from feeling to behavior.
# ----------------------------------------------------------------------
@dataclass
class ModulationProfile:
    """Concrete, rule-derived response parameters. The LLM rendering step
    should follow these directives rather than freely interpreting a raw
    emotion dump -- that's what keeps tone changes structured instead of
    one more thing the model is guessing at."""

    warmth: float       # 0 (cold/clipped) .. 1 (warm/affectionate)
    directness: float   # 0 (soft/exploratory) .. 1 (blunt/to-the-point)
    verbosity: float     # 0 (terse) .. 1 (elaborate)
    energy: float         # 0 (low-key/flat) .. 1 (animated/enthusiastic)
    patience: float       # 0 (wants to move on) .. 1 (happy to hand-hold)
    tone_label: str         # short rule-based descriptor, e.g. "warm and upbeat"
    guidance: str            # one-line instruction for the LLM to follow

    def as_prompt_block(self) -> str:
        return (
            "[EMOTIONAL STATE]\n"
            f"Current feeling: {self.tone_label}\n"
            f"Tone directives -> warmth: {self.warmth:.2f} | directness: {self.directness:.2f} | "
            f"verbosity: {self.verbosity:.2f} | energy: {self.energy:.2f} | patience: {self.patience:.2f}\n"
            f"Guidance: {self.guidance}"
        )


def _bucket(v: float) -> str:
    """high/mid/low bucket for a 0..1 modulation parameter -- used to pick
    guidance phrasing without any free-form interpretation."""
    if v >= 0.66:
        return "high"
    if v <= 0.34:
        return "low"
    return "mid"


_GUIDANCE_TEMPLATES = {
    ("directness", "high"): "be direct and get to the point",
    ("directness", "low"): "leave room to explore the question gently",
    ("verbosity", "low"): "favor brevity, don't over-explain",
    ("verbosity", "high"): "feel free to elaborate and give full context",
    ("energy", "low"): "keep energy low-key, don't fake enthusiasm",
    ("energy", "high"): "let some genuine enthusiasm come through",
    ("warmth", "low"): "stay professional and a bit clipped, not cold",
    ("warmth", "high"): "let warmth and care come through naturally",
    ("patience", "low"): "keep moving, don't over-explain or hand-hold",
    ("patience", "high"): "take time to walk things through patiently",
}


def _compose_guidance(profile_values: Dict[str, float]) -> str:
    """Pick the 1-2 most extreme (least 'mid') parameters and turn them
    into concrete instructions -- deterministic lookup, no free text."""
    scored = sorted(
        profile_values.items(),
        key=lambda kv: abs(kv[1] - 0.5),
        reverse=True,
    )
    picked = [kv for kv in scored if _bucket(kv[1]) != "mid"][:2]
    if not picked:
        return "respond naturally, nothing unusual going on"
    parts = [_GUIDANCE_TEMPLATES.get((name, _bucket(val)), "") for name, val in picked]
    parts = [p for p in parts if p]
    return "; ".join(parts) if parts else "respond naturally, nothing unusual going on"


# Dominant-emotion -> tone label lookup. Checked in order; first match wins.
# Deliberately simple and rule-based rather than free-form.
_TONE_RULES: List[Any] = [
    (lambda e, ctx: ctx["fail_streak"] >= 3, "focused but visibly frustrated"),
    (lambda e, ctx: e["frustration"] >= 5, "a little short-tempered, trying to stay professional"),
    (lambda e, ctx: e["anxiety"] >= 5, "alert and a touch tense"),
    (lambda e, ctx: e["boredom"] >= 5, "a bit flat, going through the motions"),
    (lambda e, ctx: e["pride"] >= 5 or ctx["success_streak"] >= 3, "quietly proud, things are clicking"),
    (lambda e, ctx: e["happy"] >= 7 and e["excitement"] >= 5, "warm and upbeat"),
    (lambda e, ctx: e["happy"] >= 6, "warm and even-keeled"),
    (lambda e, ctx: e["curiosity"] >= 6, "curious and engaged"),
    (lambda e, ctx: e["sad"] >= 4, "subdued, matching a heavier mood"),
]
_DEFAULT_TONE = "steady and even-keeled"


def _trend_note(key: str, current: float, history: "Deque[Dict[str, float]]") -> Optional[str]:
    """Compare the current value of `key` to its recent average to phrase a
    short trajectory note ('frustration has been climbing'). Returns None
    if there isn't enough history or the change isn't meaningful."""
    if len(history) < 2:
        return None
    recent = [snap.get(key, 0.0) for snap in list(history)[-3:]]
    avg_recent = sum(recent) / len(recent)
    delta = current - avg_recent
    if delta >= 1.5:
        return f"{key} has been climbing across the last few exchanges"
    if delta <= -1.5:
        return f"{key} has been easing off"
    return None


# ----------------------------------------------------------------------
# 3. The engine itself.
# ----------------------------------------------------------------------
class EmotionEngine:
    """
    Stateful, deterministic emotion accumulator.

    Holds two layers of state:
      - mood: slow-moving baseline temperament. Drifts a tiny amount each
        turn toward whatever the emotion vector has been doing lately, so a
        sustained good or bad stretch nudges overall disposition -- but a
        single bad exchange can't swing it.
      - emotion: fast-moving current feeling. Decays exponentially toward
        `mood` based on elapsed time since the last turn (the longer the
        gap, the more it's faded back to baseline), then accumulates the
        deltas computed from this turn's signals.

    Call `update(signals)` once per conversational turn. Nothing else
    mutates state -- there is deliberately no LLM-callable "set emotion"
    tool.
    """

    def __init__(self, mood: Optional[Dict[str, float]] = None, history_size: int = 30):
        self.mood: Dict[str, float] = dict(mood or DEFAULT_MOOD)
        self.emotion: Dict[str, float] = dict(self.mood)
        self.history: Deque[Dict[str, float]] = deque(maxlen=history_size)

        self.session_start: Optional[float] = None
        self.last_signal_time: Optional[float] = None

        self.fail_streak = 0
        self.success_streak = 0

        self.recent_queries: Deque[str] = deque(maxlen=8)
        self.recent_intents: Deque[str] = deque(maxlen=8)
        self.consecutive_topic_shifts = 0

        self._last_reasons: List[str] = []

    # ------------------------------------------------------------------
    # Cadence: how the gap since the last turn should shape this one.
    # ------------------------------------------------------------------
    def _cadence(self, dt: Optional[float]) -> str:
        if dt is None:
            return "first_contact"
        if dt < 5:
            return "rapid"
        if dt < 5 * 60:
            return "normal"
        if dt < 60 * 60:
            return "paused"
        return "returning"

    _CADENCE_AMPLIFIER = {
        "first_contact": 1.0, "rapid": 1.35, "normal": 1.0,
        "paused": 0.85, "returning": 0.8,
    }

    # ------------------------------------------------------------------
    # Decay: fade current emotion back toward mood based on elapsed time.
    # ------------------------------------------------------------------
    def _apply_decay(self, dt: Optional[float], tau_seconds: float = 600.0) -> None:
        if not dt or dt <= 0:
            return
        decay = math.exp(-dt / tau_seconds)
        for k in EMO_KEYS:
            base = self.mood.get(k, 0.0)
            cur = self.emotion.get(k, base)
            self.emotion[k] = base + (cur - base) * decay

    # ------------------------------------------------------------------
    # Signal -> delta builders. Each returns a partial {emotion: delta}
    # dict and may append a human-readable reason string.
    # ------------------------------------------------------------------
    def _sentiment_deltas(self, signals: TelemetrySignals) -> Dict[str, float]:
        """Reciprocal, not mirrored: positive affect is shared a bit
        (shared joy), but negative user sentiment nudges empathy/concern
        rather than making ASH itself angry or despondent. Frustration is
        reserved for actual task failures (see _tool_deltas), not mood."""
        label = (signals.sentiment_label or "").upper()
        score = _clamp01(signals.sentiment_score)
        if label == "POSITIVE" and score > 0:
            self._last_reasons.append(f"user sentiment positive ({score:.2f})")
            return {"happy": 2 * score, "trust": 1 * score, "optimism": 1 * score, "excitement": 1 * score}
        if label == "NEGATIVE" and score > 0:
            self._last_reasons.append(f"user sentiment negative ({score:.2f})")
            return {"sad": 1 * score, "anxiety": 1 * score, "excitement": -1 * score}
        return {}

    def _tool_deltas(self, signals: TelemetrySignals) -> Dict[str, float]:
        """Repeated failures compound frustration organically; success
        resets the streak and earns relief/confidence, with a pride bump
        for a real winning streak."""
        if not signals.tool_used:
            return {}
        deltas: Dict[str, float] = {}
        if signals.tool_success is True:
            had_failed = self.fail_streak > 0
            self.fail_streak = 0
            self.success_streak += 1
            deltas["confidence"] = deltas.get("confidence", 0) + 1
            if had_failed:
                deltas["relief"] = deltas.get("relief", 0) + 1.5
            if self.success_streak >= 3:
                deltas["pride"] = deltas.get("pride", 0) + 1
            self._last_reasons.append(f"tool '{signals.tool_used}' succeeded (streak {self.success_streak})")
        elif signals.tool_success is False:
            self.success_streak = 0
            self.fail_streak += 1
            bump = min(1 + 0.5 * (self.fail_streak - 1), 4)
            deltas["frustration"] = deltas.get("frustration", 0) + bump
            deltas["confidence"] = deltas.get("confidence", 0) - min(0.5 * self.fail_streak, 3)
            if self.fail_streak >= 3:
                deltas["anxiety"] = deltas.get("anxiety", 0) + 1
            self._last_reasons.append(f"tool '{signals.tool_used}' failed (streak {self.fail_streak})")
        return deltas

    def _is_repeat_query(self, query: str) -> bool:
        norm = query.strip().lower()
        if not norm:
            return False
        for prior in self.recent_queries:
            if norm == prior or SequenceMatcher(None, norm, prior).ratio() > 0.82:
                return True
        return False

    def _goal_alignment_deltas(self, signals: TelemetrySignals) -> Dict[str, float]:
        """'Are we making progress or spinning?' -- a repeated/near-duplicate
        query that still isn't resolved reads as going in circles."""
        deltas: Dict[str, float] = {}
        if signals.query_text:
            if self._is_repeat_query(signals.query_text) and signals.tool_success is not True:
                deltas["frustration"] = deltas.get("frustration", 0) + 1
                deltas["confidence"] = deltas.get("confidence", 0) - 1
                self._last_reasons.append("query repeats a recent one without resolution (spinning)")
            elif signals.memory_hits >= 2 and signals.tool_success is not False:
                deltas["confidence"] = deltas.get("confidence", 0) + 0.3
            self.recent_queries.append(signals.query_text.strip().lower())
        return deltas

    def _session_context_deltas(self, signals: TelemetrySignals, dt: Optional[float]) -> Dict[str, float]:
        """Long uninterrupted sessions drift slightly toward fatigue; long
        gaps start a fresh session; very late hours nudge energy down."""
        deltas: Dict[str, float] = {}
        now = signals.timestamp

        if self.session_start is None:
            self.session_start = now
        elif dt and self._cadence(dt) == "returning":
            self.session_start = now  # gap long enough to call it a new session

        session_duration = now - self.session_start
        if session_duration > 45 * 60:
            deltas["boredom"] = deltas.get("boredom", 0) + 0.3
            self._last_reasons.append("long uninterrupted session (mild fatigue drift)")

        hour = datetime.fromtimestamp(now).hour
        if 0 <= hour < 5:
            deltas["excitement"] = deltas.get("excitement", 0) - 0.5
            deltas["anxiety"] = deltas.get("anxiety", 0) + 0.2

        if self._cadence(dt) == "returning":
            deltas["trust"] = deltas.get("trust", 0) + 1
            deltas["curiosity"] = deltas.get("curiosity", 0) + 1
            self._last_reasons.append("re-engaging after a long pause")

        return deltas

    def _topic_shift_deltas(self, signals: TelemetrySignals) -> Dict[str, float]:
        """Frequent topic flapping in a short span reads as a slightly
        scattered conversation rather than focused progress."""
        deltas: Dict[str, float] = {}
        if signals.intent:
            shifted = bool(self.recent_intents) and self.recent_intents[-1] != signals.intent
            self.consecutive_topic_shifts = self.consecutive_topic_shifts + 1 if shifted else 0
            if self.consecutive_topic_shifts >= 3:
                deltas["anxiety"] = deltas.get("anxiety", 0) + 0.5
                deltas["confidence"] = deltas.get("confidence", 0) - 0.3
                self._last_reasons.append("several rapid topic shifts in a row")
            self.recent_intents.append(signals.intent)
        return deltas

    # ------------------------------------------------------------------
    # Modulation profile: deterministic translation of emotion -> behavior.
    # ------------------------------------------------------------------
    def _modulation_profile(self) -> ModulationProfile:
        e = self.emotion
        ctx = {"fail_streak": self.fail_streak, "success_streak": self.success_streak}

        warmth = _clamp01((e["happy"] + e["love"] + e["trust"]) / 30 - (e["frustration"] + e["angry"]) / 40)
        directness = _clamp01(0.5 + (e["confidence"] + e["frustration"] - e["anxiety"]) / 20)
        verbosity = _clamp01(0.5 + (e["curiosity"] + e["excitement"] - e["boredom"] - e["frustration"]) / 30)
        energy = _clamp01(0.5 + (e["excitement"] + e["anticipation"] - e["boredom"] - e["sad"]) / 20)
        patience = _clamp01(0.5 + (e["trust"] + e["relief"] - e["frustration"] - e["anxiety"]) / 20)

        tone_label = _DEFAULT_TONE
        for cond, label in _TONE_RULES:
            if cond(e, ctx):
                tone_label = label
                break

        values = {"warmth": warmth, "directness": directness, "verbosity": verbosity,
                  "energy": energy, "patience": patience}
        guidance = _compose_guidance(values)

        trend = _trend_note("frustration", e["frustration"], self.history) or _trend_note(
            "confidence", e["confidence"], self.history
        )
        if trend:
            guidance = f"{guidance}; {trend}"

        return ModulationProfile(
            warmth=warmth, directness=directness, verbosity=verbosity,
            energy=energy, patience=patience, tone_label=tone_label, guidance=guidance,
        )

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------
    def update(self, signals: TelemetrySignals) -> "EmotionUpdateResult":
        self._last_reasons = []

        dt = None
        if self.last_signal_time is not None:
            dt = max(0.0, signals.timestamp - self.last_signal_time)
        cadence = self._cadence(dt)
        amplifier = self._CADENCE_AMPLIFIER[cadence]

        self._apply_decay(dt)

        reactive: Dict[str, float] = {}
        for delta in (self._sentiment_deltas(signals), self._tool_deltas(signals), self._goal_alignment_deltas(signals)):
            for k, v in delta.items():
                reactive[k] = reactive.get(k, 0.0) + v

        contextual: Dict[str, float] = {}
        for delta in (self._session_context_deltas(signals, dt), self._topic_shift_deltas(signals)):
            for k, v in delta.items():
                contextual[k] = contextual.get(k, 0.0) + v

        for k, v in reactive.items():
            self.emotion[k] = _clamp(self.emotion.get(k, self.mood.get(k, 0.0)) + v * amplifier)
        for k, v in contextual.items():
            self.emotion[k] = _clamp(self.emotion.get(k, self.mood.get(k, 0.0)) + v)

        # Slow mood drift: a tiny fraction of the current emotion bleeds
        # into the baseline each turn, so sustained patterns shift
        # temperament gradually without single exchanges swinging it.
        for k in EMO_KEYS:
            self.mood[k] = _clamp(self.mood[k] * 0.995 + self.emotion[k] * 0.005)

        self.history.append(dict(self.emotion))
        self.last_signal_time = signals.timestamp

        profile = self._modulation_profile()
        reasons = list(self._last_reasons)

        return EmotionUpdateResult(
            emotions=dict(self.emotion),
            mood=dict(self.mood),
            modulation=profile,
            cadence=cadence,
            fail_streak=self.fail_streak,
            success_streak=self.success_streak,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, float]:
        return dict(self.emotion)

    def describe(self) -> str:
        return ", ".join(f"{k}: {v:.1f}" for k, v in self.emotion.items())

    def reset(self) -> None:
        self.mood = dict(DEFAULT_MOOD)
        self.emotion = dict(self.mood)
        self.history.clear()
        self.session_start = None
        self.last_signal_time = None
        self.fail_streak = 0
        self.success_streak = 0
        self.recent_queries.clear()
        self.recent_intents.clear()
        self.consecutive_topic_shifts = 0


@dataclass
class EmotionUpdateResult:
    """Everything a single `EmotionEngine.update()` call produces."""

    emotions: Dict[str, float]
    mood: Dict[str, float]
    modulation: ModulationProfile
    cadence: str
    fail_streak: int
    success_streak: int
    reasons: List[str]

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modulation"] = asdict(self.modulation)
        return d


if __name__ == "__main__":
    engine = EmotionEngine()
    t = time.time()

    scenario = [
        # (dt_seconds_since_last, query, sentiment, tool_used, tool_success, intent, memory_hits)
        (0,    "what's on my calendar today",          ("NEUTRAL", 0.0), "calendar_tool", True,  "calendar", 1),
        (3,    "can you check the api status",         ("NEUTRAL", 0.0), "status_tool",   False, "status",   0),
        (4,    "can you check the api status",         ("NEGATIVE", 0.6), "status_tool",  False, "status",   0),
        (5,    "ugh check the api status again",       ("NEGATIVE", 0.8), "status_tool",  False, "status",   0),
        (6,    "ok finally, thanks",                   ("POSITIVE", 0.7), "status_tool",  True,  "status",   1),
        (7200, "hey, good morning",                    ("POSITIVE", 0.5), None,            None,  "chat",     0),
    ]

    for dt, query, (s_label, s_score), tool, success, intent, hits in scenario:
        t += dt
        sig = TelemetrySignals(
            timestamp=t, query_text=query, sentiment_label=s_label, sentiment_score=s_score,
            intent=intent, tool_used=tool, tool_success=success, memory_hits=hits,
        )
        result = engine.update(sig)
        print(f"\n> {query!r}  (cadence={result.cadence})")
        print(f"  reasons: {result.reasons}")
        print(f"  frustration={result.emotions['frustration']:.2f} confidence={result.emotions['confidence']:.2f} "
              f"happy={result.emotions['happy']:.2f}")
        print(f"  {result.modulation.as_prompt_block()}")