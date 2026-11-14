"""What ASH does with a free moment.

A free moment is a pause after a stretch of work: the user is not typing, not
in a conversation, not busy. It is rare, so it is worth having a few things
that could fill it, and choosing between them rather than always doing the
same one. Closing a gap in what ASH knows (see `wondering`) is one such thing.
Others are a light joke, asking how the user is, a note on how ASH itself is
running, a question that calibrates a prediction.

Each is an `Activity`: it looks at the moment and may propose one thing to
say, with a value (how much it is worth saying now). The registry scores each
proposal as value x weight x fit, where fits are pluggable functions of the
proposal and the moment, and picks the best. It spends a budget (a gap between
free-moment remarks, a cap per hour), gives each activity its own cool-down,
and stays quiet whenever the user should not be disturbed.

Adding a new kind of thing to do in a free moment is one `Activity`, or one
function handed to `ProviderActivity`, and `FreeMoments.register`.

Mood comes from the sentiment model the brain already runs on every query
(`mood_of`), not from word lists: the answer to "how are you holding up?" is
read by the same model, and so is everything the user asks in between.

Privacy: what the user replies to a mood question is read for a sign and
dropped; only a number and a time are kept, in memory, never written out.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .wondering import Environment

logger = logging.getLogger("ash.freemoment")


@dataclass
class Candidate:
    activity: str
    text: str
    value: float = 0.5                 # 0..1: how worth saying this is now
    data: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0                 # filled in by the registry


FitFn = Callable[[Candidate, Environment], float]
# The brain's sentiment model: text -> {"sentiment": "POSITIVE"|"NEGATIVE"|..., "confidence": 0..1}
SentimentFn = Callable[[str], Dict[str, Any]]


class Activity:
    """One thing ASH might do in a free moment. Subclass, or use
    `ProviderActivity`."""
    name = "activity"
    weight = 1.0                       # standing preference over other activities
    cooldown_s = 3600.0                # at least this long between two of its own

    def propose(self, env: Environment, now: float) -> Optional[Candidate]:
        return None

    def spoken(self, now: float) -> None:
        """What it proposed was actually said (the attention gate let it
        through). Most activities need not care."""
        return None

    def hear(self, text: str, now: float) -> Optional[str]:
        """The user's next utterance after this activity spoke. Return a line
        to say back, or None to say nothing (the utterance is then handled
        as an ordinary one)."""
        return None


@dataclass
class FreeMomentsConfig:
    enabled: bool = True
    min_gap_s: float = 600.0           # between two free-moment remarks
    max_per_hour: int = 2
    answer_window_s: float = 120.0
    disabled: Tuple[str, ...] = ()     # activities to leave out, by name
    min_score: float = 0.1             # below this, silence beats the remark
    mood_ttl_s: float = 4 * 3600.0     # how long a mood reading is believed

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FreeMomentsConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                continue
            try:
                cur = getattr(c, k)
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, tuple):
                    setattr(c, k, tuple(str(x) for x in (v or ())))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                else:
                    setattr(c, k, float(v))
            except (TypeError, ValueError):
                pass
        return c


class FreeMoments:
    def __init__(self, cfg: Optional[FreeMomentsConfig] = None,
                 clock: Callable[[], float] = time.time,
                 sentiment: Optional[SentimentFn] = None):
        self.cfg = cfg or FreeMomentsConfig()
        self.clock = clock
        self.sentiment = sentiment         # the brain's sentiment model; None: moods are not read
        self.activities: Dict[str, Activity] = {}
        self._fits: List[FitFn] = []
        self._last_by: Dict[str, float] = {}
        self._spoken: Deque[float] = deque(maxlen=32)
        self._awaiting: Optional[Tuple[str, float]] = None
        self._mood: Optional[Tuple[float, float]] = None    # (value, when)
        from .backoff import Backoff
        self.backoff = Backoff(clock=lambda: self.clock())  # activities waved off lately sit out
        self.stats = {"picked": 0, "spoken": 0, "replies": 0}

    # -- setup ----------------------------------------------------------
    def register(self, act: Activity) -> Activity:
        self.activities[act.name] = act
        return act

    def add_fit(self, fn: FitFn):
        self._fits.append(fn)

    # -- mood: the one signal free moments can also produce --------------
    def set_mood(self, value: float, now: Optional[float] = None):
        self._mood = (max(-1.0, min(1.0, float(value))), self.clock() if now is None else now)

    def read(self, text: str) -> Optional[float]:
        """-1..1 for a sentence, from the sentiment model; None if there is no
        model, it is unsure, or it fails."""
        return read_with(self.sentiment, text)

    def note_mood(self, value: Optional[float], now: Optional[float] = None, weight: float = 0.5):
        """Fold a reading into the current mood: a single sentence moves it,
        it does not replace it (unless the old one has gone stale)."""
        if value is None:
            return
        now = self.clock() if now is None else now
        cur = self.mood(now)
        self.set_mood(value if cur is None else (1 - weight) * cur + weight * value, now)

    def mood(self, now: Optional[float] = None) -> Optional[float]:
        if self._mood is None:
            return None
        now = self.clock() if now is None else now
        v, t = self._mood
        return v if now - t <= self.cfg.mood_ttl_s else None

    # -- choosing -------------------------------------------------------
    def _blocked(self, env: Environment) -> str:
        if env.muted:
            return "muted"
        if env.quiet_hours:
            return "quiet hours"
        if env.paused:
            return "privacy pause"
        if env.conversation_open:
            return "in conversation"
        if env.plan_active:
            return "plan active"
        if env.speaking:
            return "speaking"
        if env.idle_s is None:
            return "idle unknown"
        if not env.free_moment:
            return "not a free moment"
        return ""

    def can_speak(self, env: Environment, now: Optional[float] = None) -> Tuple[bool, str]:
        now = self.clock() if now is None else now
        if not self.cfg.enabled:
            return False, "off"
        why = self._blocked(env)
        if why:
            return False, why
        if self._spoken and now - self._spoken[-1] < self.cfg.min_gap_s:
            return False, "too soon after the last"
        if sum(1 for t in self._spoken if now - t <= 3600) >= self.cfg.max_per_hour:
            return False, "hourly budget spent"
        if self._awaiting is not None and now - self._awaiting[1] < self.cfg.answer_window_s:
            return False, "waiting for an answer"
        return True, ""

    def fit(self, c: Candidate, env: Environment) -> float:
        f = 1.0
        for fn in self._fits:
            try:
                f *= max(0.0, float(fn(c, env)))
            except Exception:
                logger.exception("free-moment fit failed")
            if f == 0.0:
                break
        return f

    def pick(self, env: Environment, now: Optional[float] = None) -> Optional[Candidate]:
        now = self.clock() if now is None else now
        ok, _ = self.can_speak(env, now)
        if not ok:
            return None
        if env.mood is None:
            env = _with_mood(env, self.mood(now))
        best: Optional[Candidate] = None
        for name, act in self.activities.items():
            if name in self.cfg.disabled or self.backoff.benched(name, now):
                continue
            if now - self._last_by.get(name, -1e12) < act.cooldown_s:
                continue
            try:
                c = act.propose(env, now)
            except Exception:
                logger.exception("free-moment activity %s failed", name)
                continue
            if c is None or not c.text:
                continue
            c.activity = name
            c.score = max(0.0, min(1.0, c.value)) * act.weight * self.fit(c, env)
            if c.score >= self.cfg.min_score and (best is None or c.score > best.score):
                best = c
        if best is not None:
            self.stats["picked"] += 1
        return best

    def mark_spoken(self, name: str, now: Optional[float] = None):
        now = self.clock() if now is None else now
        self._last_by[name] = now
        self._spoken.append(now)
        self._awaiting = (name, now)
        self.stats["spoken"] += 1
        act = self.activities.get(name)
        if act is not None:
            try:
                act.spoken(now)
            except Exception:
                logger.exception("free-moment activity %s failed on spoken", name)

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """An utterance right after a free-moment remark. Returns a reply line
        if the activity took it as an answer; None leaves it to others."""
        now = self.clock() if now is None else now
        if self._awaiting is None:
            return None
        name, t = self._awaiting
        if now - t > self.cfg.answer_window_s:
            self._awaiting = None
            return None
        act = self.activities.get(name)
        if act is None:
            return None
        try:
            r = act.hear(str(text or ""), now)
        except Exception:
            logger.exception("free-moment answer failed")
            return None
        if r is not None:
            self._awaiting = None
            self.stats["replies"] += 1
        return r

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        return {
            "enabled": self.cfg.enabled,
            "activities": sorted(a for a in self.activities if a not in self.cfg.disabled),
            "spoken_last_hour": sum(1 for t in self._spoken if now - t <= 3600),
            "budget": self.cfg.max_per_hour,
            "mood": self.mood(now),
            "backoff": self.backoff.status(now),
            **self.stats,
        }


def _with_mood(env: Environment, mood: Optional[float]) -> Environment:
    if mood is None:
        return env
    from dataclasses import replace
    return replace(env, mood=mood)


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------
class ProviderActivity(Activity):
    """An activity made of one function: `provider(env, now)` returns
    (text, value) or None. This is how a new kind of free-moment remark is
    added: a system assessment, a prediction-calibration question."""

    def __init__(self, name: str, provider: Callable[[Environment, float], Optional[Tuple[str, float]]],
                 cooldown_s: float = 3600.0, weight: float = 1.0,
                 on_answer: Optional[Callable[[str, float], Optional[str]]] = None):
        self.name = name
        self.provider = provider
        self.cooldown_s = cooldown_s
        self.weight = weight
        self._on_answer = on_answer

    def propose(self, env, now):
        r = self.provider(env, now)
        if not r:
            return None
        text, value = r
        return Candidate(self.name, str(text), float(value))

    def hear(self, text, now):
        return self._on_answer(text, now) if self._on_answer else None


class GapClosure(Activity):
    """Ask about something ASH noticed it does not know. The questions come
    from the Wonderer, which has its own limits; this only offers the best one
    to the free-moment contest, so it competes with the other activities."""
    name = "gap_closure"
    weight = 1.0
    cooldown_s = 0.0                   # the Wonderer paces itself

    def __init__(self, wonderer):
        self.wonderer = wonderer

    def propose(self, env, now):
        w = self.wonderer
        if w is None:
            return None
        q = w.next_question(env, now)
        if q is None:
            return None
        # Closing a gap is worth more the more it is wanted; an old, patient
        # question still beats a joke.
        return Candidate(self.name, w.render(q, now),
                         value=max(0.5, min(1.0, 0.5 + 0.5 * q.urgency)),
                         data={"wonder": q})


_JOKES = (
    "A programmer's partner says: go to the shop, buy a loaf of bread, and if they have eggs, get a dozen. "
    "They come back with twelve loaves.",
    "There are only two hard things in computing: cache invalidation, naming things, and off-by-one errors.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "It works on my machine. So we are shipping your machine.",
    "A SQL query walks into a bar, sees two tables, and asks: can I join you?",
    "I tried to write a recursive joke, but first I had to write a recursive joke.",
    "My code has no bugs. It just develops random features.",
)


class Joke(Activity):
    """Something light, for when things are going fine. Never when the user is
    low, never twice in a row, and the same joke only after the rest."""
    name = "joke"
    weight = 0.8
    cooldown_s = 6 * 3600.0

    def __init__(self, jokes=_JOKES):
        self.jokes = tuple(jokes)
        self._i = 0

    def propose(self, env, now):
        if not self.jokes:
            return None
        if env.mood is not None and env.mood < 0.0:
            return None
        value = 0.25 + (0.15 if (env.mood or 0.0) > 0.3 else 0.0)
        return Candidate(self.name, self.jokes[self._i % len(self.jokes)], value)

    def hear(self, text, now):
        self._i += 1                    # told once; the next is a different one
        return None


def mood_of(result: Any, min_confidence: float = 0.6) -> Optional[float]:
    """-1..1 from the sentiment model's result ({"sentiment", "confidence"}),
    or None when it has nothing to say: no result, an unknown label, or too
    unsure to count. A two-way model (positive/negative) is scaled by how
    sure it is, so a 0.6 reading moves the mood far less than a 0.99 one; a
    three-way model's neutral is 0."""
    if not isinstance(result, dict):
        return None
    label = str(result.get("sentiment") or result.get("label") or "").upper()
    try:
        conf = float(result.get("confidence", result.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return None
    if label in ("NEUTRAL", "LABEL_1_NEUTRAL"):
        return 0.0 if conf >= min_confidence else None
    sign = 1.0 if label in ("POSITIVE", "POS") else -1.0 if label in ("NEGATIVE", "NEG") else 0.0
    if sign == 0.0 or conf < min_confidence:
        return None
    return sign * min(1.0, max(0.0, (conf - 0.5) * 2.0))


def read_with(sentiment: Optional[SentimentFn], text: str) -> Optional[float]:
    """The model's mood for a sentence; None without a model or on failure."""
    if sentiment is None or not isinstance(text, str) or not text.strip():
        return None
    try:
        return mood_of(sentiment(text))
    except Exception:
        logger.exception("sentiment model failed on a reply")
        return None


class MoodCheck(Activity):
    """Ask how the user is, when ASH has no recent idea. The answer sets the
    mood that other activities (and the wonderer's fit) read."""
    name = "mood_check"
    weight = 0.9
    cooldown_s = 8 * 3600.0

    def __init__(self, moments: FreeMoments):
        self.moments = moments

    def propose(self, env, now):
        if self.moments.mood(now) is not None:
            return None                 # already has a recent reading
        return Candidate(self.name, "How are you holding up today?", 0.4)

    def hear(self, text, now):
        m = self.moments.read(text)
        if m is None:
            return None                 # not an answer; handle it as any utterance
        self.moments.set_mood(m, now)   # asked directly: this reading replaces the old one
        if m <= -0.3:
            return "Ah, that sounds rough. I'll go easy."
        if m >= 0.3:
            return "Good. Hold onto that."
        return "Fair enough."


def default_moments(cfg: Optional[Dict[str, Any]] = None, wonderer=None,
                    clock: Callable[[], float] = time.time,
                    jokes: bool = True, mood_check: bool = True,
                    sentiment: Optional[SentimentFn] = None) -> FreeMoments:
    fm = FreeMoments(FreeMomentsConfig.from_dict(cfg), clock, sentiment=sentiment)
    if wonderer is not None:
        fm.register(GapClosure(wonderer))
    if jokes:
        fm.register(Joke())
    if mood_check:
        fm.register(MoodCheck(fm))
    return fm
