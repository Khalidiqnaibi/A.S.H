"""
src/daemon/social.py

The social loop: noticing when to speak, and learning from how it lands.

The attention gate decides whether something is worth saying. Until now every
number in it was picked by hand and nothing ever found out whether a remark
was welcome. This module closes that loop, in three small parts.

    SocialLearner     Keeps a track record of how each KIND of remark lands in
                      each kind of SITUATION (busy/idle, time of day), and turns
                      it into a shift of the gate's bar: raised where remarks
                      were unwelcome, lowered where they were. Few observations
                      stay close to the prior; many move it. Where it has
                      little evidence it adds a small exploration bonus, or it
                      would never speak and so never learn that it may.

    ReactionTracker   Remembers what was just said and reads what the user does
                      next. A reply is welcome or not; a dismissal is a cost;
                      saying nothing is a weak negative (ignoring is
                      ambiguous). "Not now" is a strong one and snoozes that
                      kind of remark for a while.

    FlowDetector      Spots the two moments worth speaking up in, from events
                      the loop already sees: a FREE MOMENT (a stretch of real
                      work just ended) and being STUCK (the same error again,
                      the same question asked three ways, a long time in one
                      window). It turns them into events carrying an offer.

What it cannot change: the hourly budget, quiet hours, and the rule that a
person typing right now is not interrupted live in the gate and sit outside
the learner. Only the soft bar moves, and only inside fixed limits, scaled
down by urgency so that nothing it learns can silence a real emergency.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.senses.base import Modality, SensorEvent

logger = logging.getLogger("ash.daemon.social")

# --- how a reaction is scored -------------------------------------------------
REWARD = {
    "accepted": 1.0,      # "yes please", "thanks", "go ahead"
    "engaged": 0.8,       # answered, or went on to talk about something
    "silence": 0.3,       # nothing happened -- ambiguous, so only mildly negative
    "deferred": 0.25,     # "not now", "later": not a no, but not welcome either
    "dismissed": 0.0,     # swiped the notification away
    "rejected": 0.0,      # "be quiet", "no thanks"
}
WEIGHT = {"accepted": 1.0, "engaged": 1.0, "silence": 0.6, "deferred": 1.0,
          "dismissed": 1.0, "rejected": 2.0}

# A reply this long is a new request, not a verdict on the offer.
VERDICT_MAX_WORDS = 8

_REJECT = re.compile(
    r"\b(?:go\s+away|leave\s+me\s+alone|shut\s+up|be\s+quiet|"
    r"stop\s+(?:it|that|talking|interrupting|bothering|nagging)|don'?t\s+(?:interrupt|bother|disturb)|"
    r"no\s+thanks?|not\s+interested)\b|^\s*(?:no|nope|nah|quiet)\W*$", re.I)
_DEFER = re.compile(r"\b(?:not\s+now|not\s+right\s+now|later|i'?m\s+busy|busy)\b", re.I)
_ACCEPT = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please|go\s+ahead|do\s+it|thanks?|thank\s+you|"
    r"that\s+would\s+help|sounds\s+good|show\s+me|help|why\s+not|alright)\b", re.I)


def classify_reply(text: str) -> str:
    """'accepted' | 'deferred' | 'rejected' | 'engaged' for something said to ASH."""
    t = (text or "").strip()
    if len(t.split()) <= VERDICT_MAX_WORDS:
        if _REJECT.search(t):
            return "rejected"
        if _DEFER.search(t):
            return "deferred"
        if _ACCEPT.search(t):
            return "accepted"
    return "engaged"


# --- the check-in line --------------------------------------------------------
# A free-moment remark that could be said to anyone at any time ("how can I help
# you today?") is worse than none: it asks the user to do the work of thinking
# of something. The model is asked for something specific; if what comes back is
# generic, long or a string of questions, a plain line built from the facts
# (how long, in what) is said instead.
_GENERIC = re.compile(
    r"how can i (?:help|assist)|how may i (?:help|assist)|(?:need|want) any help|anything (?:i can|else)|"
    r"(?:don'?t|do not) hesitate|feel free to ask|here to help|let me know if|what can i do for you|"
    r"just wanted to (?:check in|see)|hey there|how are you (?:doing )?today|how can i be of", re.I)

_APPS = {
    "code": "VS Code", "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "claude": "Claude",
    "windowsterminal": "the terminal", "cmd": "the terminal", "powershell": "the terminal",
    "pwsh": "the terminal", "python": "Python", "pythonw": "Python", "pycharm64": "PyCharm",
    "idea64": "IntelliJ", "explorer": "File Explorer", "slack": "Slack", "discord": "Discord",
    "spotify": "Spotify", "notepad": "Notepad", "winword": "Word", "excel": "Excel",
    "powerpnt": "PowerPoint", "githubdesktop": "GitHub Desktop",
}


def friendly_app(process: str) -> str:
    """'claude.exe' -> 'Claude'; unknown names are tidied, empty stays empty."""
    p = re.sub(r"\.(?:exe|app|bin)$", "", (process or "").strip(), flags=re.I)
    if not p:
        return ""
    return _APPS.get(p.lower(), p[:1].upper() + p[1:])


def where_phrase(process: str = "", subject: str = "", project: str = "") -> str:
    """" in VS Code", " on brain.py in A.S.H", or nothing.

    The subject leads because it is the part that says what the work was. An
    hour "in Code" is a fact about a window; an hour "on brain.py in A.S.H"
    is a fact about the evening. The project beats the app as the container
    for the same reason -- it is the more specific of the two, and it is the
    one the user would use themselves.
    """
    app = friendly_app(process)
    subject = (subject or "").strip()
    inside = (project or "").strip() or app
    if subject and inside:
        return f" on {subject} in {inside}"
    if subject:
        return f" on {subject}"
    return f" in {inside}" if inside else ""


def is_generic_checkin(text: str) -> bool:
    t = (text or "").strip()
    return (not t or len(t) > 220 or t.count("?") > 1 or bool(_GENERIC.search(t)))


def compose_check_in(minutes: float, process: str = "", hour: Optional[int] = None,
                     subject: str = "", project: str = "") -> str:
    """A short, specific line from the facts alone: used when the model has
    nothing better than a generic one."""
    m = max(1, int(round(minutes)))
    unit = "minute" if m == 1 else "minutes"
    where = where_phrase(process, subject, project)
    late = hour is not None and (hour >= 23 or hour < 5)
    if m >= 90:
        h = m / 60.0
        line = f"That's about {h:.1f} hours straight{where}. Worth a stretch and some water?"
    elif m >= 30:
        # "at it" only when there is nothing more specific to name: "You've
        # been at it on brain.py" reads like two openings stapled together.
        line = (f"You've been{where or ' at it'} for about {m} {unit} "
                f"-- a short break might do you good.")
    elif m >= 8:
        line = f"That was a solid {m} {unit}{where}. I'm around if you want a second opinion on anything."
    else:
        line = f"Nice focus -- {m} {unit}{where} without stopping."
    return ("It's late. " + line) if late else line


def context_key(user_busy: float, now: Optional[float] = None) -> str:
    """The situation a remark lands in, coarse enough to learn from quickly."""
    if user_busy >= 0.9:
        busy = "typing"
    elif user_busy >= 0.5:
        busy = "active"
    elif user_busy >= 0.2:
        busy = "idle"
    else:
        busy = "away"
    h = time.localtime(now if now is not None else time.time()).tm_hour
    part = ("morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17
            else "evening" if 17 <= h < 22 else "night")
    return f"{busy}/{part}"


# ==============================================================================
# SocialLearner
# ==============================================================================
class SocialLearner:
    """Per-kind, per-situation track record -> a shift of the attention bar.

    Estimates are hierarchical: everything shrinks toward the overall rate
    (what this user is like in general), a kind toward that, and a kind in a
    situation toward its kind. Each level is `(sum_reward + s * parent) /
    (count + s)`, so a single bad reaction in a new situation moves it a little
    and ten move it a lot. Cells forget slowly (`decay`), so the user changing
    their mind is followed rather than averaged away.
    """

    GLOBAL = "*"

    def __init__(self, path: Optional[str] = None, prior: float = 0.5,
                 prior_strength: float = 3.0, decay: float = 0.97,
                 gain: float = 0.6, max_shift: float = 0.3,
                 explore: float = 0.10, snooze_s: float = 1800.0,
                 half_life_s: float = 2 * 86400.0, clock=time.time):
        self.path = path
        self.prior = prior
        self.prior_strength = prior_strength
        self.decay = decay
        self.gain = gain
        self.max_shift = max_shift
        self.explore = explore
        self.snooze_s = snooze_s
        self.half_life_s = half_life_s       # what was learned fades back to the prior
        self.clock = clock
        self.cells: Dict[str, List[float]] = {}      # key -> [reward_sum, count, updated_at]
        self.snoozes: Dict[str, float] = {}          # kind -> until
        self.updates = 0
        self.load()

    # -- estimates ---------------------------------------------------------
    @staticmethod
    def _key(kind: str, ctx: str = "") -> str:
        return f"{kind}|{ctx}" if ctx else kind

    def _cell(self, key: str, now: float) -> Tuple[float, float]:
        """(reward_sum, count), faded for the time since it was last touched.
        Without the fade one bad reaction could mute a kind of remark for good:
        silence produces no new evidence to undo it."""
        c = self.cells.get(key)
        if not c:
            return 0.0, 0.0
        f = 0.5 ** (max(0.0, now - c[2]) / self.half_life_s) if self.half_life_s else 1.0
        return c[0] * f, c[1] * f

    def _shrink(self, key: str, parent: float, now: float) -> float:
        a, n = self._cell(key, now)
        return (a + self.prior_strength * parent) / (n + self.prior_strength)

    def _now(self, now: Optional[float]) -> float:
        return self.clock() if now is None else now

    def p_general(self, now: Optional[float] = None) -> float:
        return self._shrink(self.GLOBAL, self.prior, self._now(now))

    def p_kind(self, kind: str, now: Optional[float] = None) -> float:
        now = self._now(now)
        return self._shrink(self._key(kind), self.p_general(now), now)

    def p(self, kind: str, ctx: str = "", now: Optional[float] = None) -> float:
        """Estimated welcome rate (0..1) for this kind of remark in this situation."""
        now = self._now(now)
        return self._shrink(self._key(kind, ctx), self.p_kind(kind, now), now)

    def evidence(self, kind: str, ctx: str = "", now: Optional[float] = None) -> float:
        return self._cell(self._key(kind, ctx), self._now(now))[1]

    def bar_shift(self, kind: str, ctx: str = "", urgency: float = 0.0,
                  now: Optional[float] = None) -> float:
        """Added to the gate's threshold: positive where this was unwelcome."""
        raw = self.gain * (0.5 - self.p(kind, ctx, now))
        shift = max(-self.max_shift, min(self.max_shift, raw))
        return shift * (1.0 - max(0.0, min(1.0, urgency)))

    def exploration(self, kind: str, ctx: str = "", urgency: float = 0.0,
                    now: Optional[float] = None) -> float:
        """Bonus to the score while there is little evidence either way."""
        return (self.explore / math.sqrt(1.0 + self.evidence(kind, ctx, now))
                * (1.0 - max(0.0, min(1.0, urgency))))

    # -- learning ----------------------------------------------------------
    def update(self, kind: str, ctx: str, reward: float, weight: float = 1.0,
               now: Optional[float] = None):
        now = self._now(now)
        reward = max(0.0, min(1.0, float(reward)))
        for key in (self.GLOBAL, self._key(kind), self._key(kind, ctx)):
            a, n = self._cell(key, now)
            self.cells[key] = [a * self.decay + weight * reward, n * self.decay + weight, now]
        self.updates += 1
        self.save()

    def snooze(self, kind: str, now: Optional[float] = None, seconds: Optional[float] = None):
        now = self._now(now)
        self.snoozes[kind] = now + (self.snooze_s if seconds is None else seconds)
        self.save()

    def snoozed(self, kind: str, now: Optional[float] = None) -> bool:
        now = self._now(now)
        until = self.snoozes.get(kind)
        if until is None:
            return False
        if now >= until:
            self.snoozes.pop(kind, None)
            return False
        return True

    # -- persistence -------------------------------------------------------
    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "cells": self.cells, "snoozes": self.snoozes,
                           "updates": self.updates}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save the social record")

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.cells = {str(k): [float(v[0]), float(v[1]),
                                   float(v[2]) if len(v) > 2 else self.clock()]
                          for k, v in d.get("cells", {}).items()}
            self.snoozes = {str(k): float(v) for k, v in d.get("snoozes", {}).items()}
            self.updates = int(d.get("updates", 0))
        except Exception:
            logger.exception("Social record unreadable; starting fresh")
            self.cells, self.snoozes, self.updates = {}, {}, 0

    def status(self) -> Dict[str, Any]:
        kinds = sorted({k.split("|")[0] for k in self.cells if k != self.GLOBAL})
        return {
            "updates": self.updates,
            "welcome_rate": round(self.p_general(), 3),
            "kinds": {k: {"p": round(self.p_kind(k), 3), "n": round(self.evidence(k), 1)}
                      for k in kinds},
            "snoozed": sorted(k for k in self.snoozes if self.snoozed(k)),
        }


# ==============================================================================
# ReactionTracker
# ==============================================================================
@dataclass
class Offer:
    kind: str
    ctx: str
    text: str
    ts: float
    topic: str = ""
    accept_context: str = ""


@dataclass
class Reaction:
    offer: Offer
    outcome: str                  # a key of REWARD


class ReactionTracker:
    """Reads what the user does after ASH volunteered something."""

    def __init__(self, learner: SocialLearner, window_s: float = 120.0, max_pending: int = 8):
        self.learner = learner
        self.window_s = window_s
        self.pending: Deque[Offer] = deque(maxlen=max_pending)
        self.counts = {k: 0 for k in REWARD}

    def register(self, kind: str, ctx: str, text: str, now: float,
                 topic: str = "", accept_context: str = "") -> Offer:
        o = Offer(kind=kind, ctx=ctx, text=text, ts=now, topic=topic, accept_context=accept_context)
        self.pending.append(o)
        return o

    def _resolve(self, offer: Offer, outcome: str, now: float) -> Reaction:
        self.counts[outcome] += 1
        self.learner.update(offer.kind, offer.ctx, REWARD[outcome], WEIGHT[outcome], now=now)
        if outcome == "rejected":
            self.learner.snooze(offer.kind, now, self.learner.snooze_s * 4)
        elif outcome == "deferred":
            self.learner.snooze(offer.kind, now)
        return Reaction(offer, outcome)

    def expire(self, now: float) -> List[Reaction]:
        """Offers nobody reacted to within the window were, mildly, unwelcome."""
        out = []
        while self.pending and now - self.pending[0].ts > self.window_s:
            out.append(self._resolve(self.pending.popleft(), "silence", now))
        return out

    def on_user_event(self, ev, now: float) -> Optional[Reaction]:
        """Call with every event. Returns the reaction if this event is one."""
        self.expire(now)
        if not self.pending or getattr(ev, "source", "") == "flow":
            return None
        meta = getattr(ev, "meta", None) or {}
        if getattr(ev, "addressed", False):
            outcome = classify_reply(getattr(ev, "text", ""))
        elif meta.get("dismissed"):
            outcome = "dismissed"
        else:
            return None
        return self._resolve(self.pending.pop(), outcome, now)      # the latest one

    def status(self) -> Dict[str, Any]:
        return {"pending": len(self.pending), "outcomes": dict(self.counts)}


# ==============================================================================
# FlowDetector
# ==============================================================================
_STOP = frozenset("the a an and or but for with this that what how why can you could would "
                  "please just have has had was were are not its it's into from does did".split())
_ERROR = re.compile(r"traceback|exception|\berror\b|\bfailed\b|\bfailure\b|fatal|segfault|"
                    r"undefined|exit\s+code\s+[1-9]|\bpanic\b|cannot\s+\w+", re.I)


def _tokens(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9_']{3,}", (text or "").lower())
                     if w not in _STOP)


def similarity(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def error_signature(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[a-z]:[\\/][^\s]*|/[\w./-]+", "<path>", t)
    t = re.sub(r"0x[0-9a-f]+|\d+", "#", t)
    return re.sub(r"\s+", " ", t).strip()[:80]


class FlowDetector:
    """Notices free moments and being stuck, from events the loop already sees.

    Rule-based on purpose: which moments are worth speaking up in is for the
    SocialLearner to learn from reactions, but whether the evidence is there at
    all should be inspectable and testable.
    """

    def __init__(self, free_idle_s: float = 20.0, free_max_idle_s: float = 120.0,
                 min_burst_s: float = 120.0, burst_gap_s: float = 45.0,
                 stuck_bar: float = 0.7, signal_window_s: float = 600.0,
                 stuck_cooldown_s: float = 1800.0, free_cooldown_s: float = 1200.0,
                 long_dwell_s: float = 1200.0, pause_s: float = 6.0):
        self.free_idle_s = free_idle_s
        self.free_max_idle_s = free_max_idle_s
        self.min_burst_s = min_burst_s
        self.burst_gap_s = burst_gap_s
        self.stuck_bar = stuck_bar
        self.signal_window_s = signal_window_s
        self.stuck_cooldown_s = stuck_cooldown_s
        self.free_cooldown_s = free_cooldown_s
        self.long_dwell_s = long_dwell_s
        self.pause_s = pause_s

        self.errors: Dict[str, Deque[float]] = {}
        self.error_text: Dict[str, str] = {}
        self.queries: Deque[Tuple[float, frozenset]] = deque(maxlen=32)
        self.process = ""
        self.subject = ""       # what is open in it: the file, the document
        self.project = ""       # and what that belongs to
        self.private = False    # a chat or a mailbox: never "a stretch of work"
        self.since = 0.0
        self.burst_start: Optional[float] = None
        self.last_active: float = 0.0
        self.free_emitted = False
        self.last_free = -1e12
        self.last_stuck: Dict[str, float] = {}
        self.emitted = {"free_moment": 0, "stuck_help": 0}

    # -- input -------------------------------------------------------------
    def observe(self, ev, now: Optional[float] = None):
        now = now if now is not None else time.time()
        if getattr(ev, "source", "") == "flow":
            return
        text = getattr(ev, "text", "") or ""
        if getattr(ev, "addressed", False):
            toks = _tokens(text)
            if toks:
                self.queries.append((now, toks))
            return
        meta = getattr(ev, "meta", None) or {}
        proc = meta.get("process")
        if proc:
            dwell = float(meta.get("dwell_s", 0.0) or 0.0)
            if proc != self.process:
                self.process, self.since = str(proc), now - dwell
                self.subject = self.project = ""
            elif dwell:
                self.since = min(self.since, now - dwell)
            self.private = bool(meta.get("private"))
            # The subject moves within a stretch of work -- switching file is
            # not switching task -- so it tracks the latest one and does not
            # restart the dwell clock the way a process change does.
            self.subject = str(meta.get("subject") or "") or self.subject
            self.project = str(meta.get("project") or "") or self.project
        if text and _ERROR.search(text):
            sig = error_signature(text)
            self.errors.setdefault(sig, deque(maxlen=16)).append(now)
            self.error_text[sig] = re.sub(r"(?i)^\s*traceback[^:\n]*:\s*", "", text.strip())[:70]

    # -- evidence ----------------------------------------------------------
    def _stuck(self, now: float):
        """(score, reasons, topic, strongest) from the last signal_window_s."""
        horizon = now - self.signal_window_s
        score, reasons, topic, best = 0.0, [], "", ""

        best_sig, best_n = None, 0
        for sig, ts in list(self.errors.items()):
            while ts and ts[0] < horizon:
                ts.popleft()
            if not ts:
                del self.errors[sig]
                continue
            if len(ts) > best_n:
                best_sig, best_n = sig, len(ts)
        if best_sig and best_n >= 2:
            s = {2: 0.2, 3: 0.5, 4: 0.65}.get(best_n, 0.8)
            score += s
            topic = self.error_text.get(best_sig, "")
            reasons.append(f"the same error {best_n} times")
            best = "error"

        while self.queries and self.queries[0][0] < horizon:
            self.queries.popleft()
        if self.queries:
            latest = self.queries[-1][1]
            k = sum(1 for _, t in self.queries if similarity(t, latest) >= 0.5)
            if k >= 2:
                score += 0.45 if k >= 3 else 0.2
                reasons.append(f"the same question {k} times")
                best = best or "question"

        recently_active = self.burst_start is not None and now - self.last_active <= self.burst_gap_s
        if recently_active and self.process and now - self.since >= self.long_dwell_s:
            score += 0.2
            reasons.append(f"{(now - self.since) / 60:.0f} minutes in {self.process}")
            best = best or "dwell"
        return min(1.0, score), reasons, topic, best

    # -- output ------------------------------------------------------------
    def poll(self, idle_s: float, now: Optional[float] = None) -> List[SensorEvent]:
        """Call once per tick with seconds since the user last touched the machine."""
        now = now if now is not None else time.time()

        if idle_s < 5.0:
            if self.burst_start is None or now - self.last_active > self.burst_gap_s:
                self.burst_start, self.free_emitted = now, False
            self.last_active = now

        # Stuck wins over a free moment, and is only raised at a natural pause:
        # never over someone mid-keystroke.
        if idle_s >= self.pause_s and idle_s < self.free_max_idle_s * 2.5:
            score, reasons, topic, best = self._stuck(now)
            if score >= self.stuck_bar - 1e-9:
                key = topic or best
                if now - self.last_stuck.get(key, -1e12) >= self.stuck_cooldown_s:
                    self.last_stuck[key] = now
                    self.emitted["stuck_help"] += 1
                    return [self._stuck_event(score, reasons, topic, best, now)]

        if (self.burst_start is not None and not self.free_emitted
                and not self.private
                and self.free_idle_s <= idle_s < self.free_max_idle_s
                and self.last_active - self.burst_start >= self.min_burst_s
                and now - self.last_free >= self.free_cooldown_s):
            self.free_emitted = True
            self.last_free = now
            self.emitted["free_moment"] += 1
            return [self._free_event(now)]
        return []

    def _stuck_event(self, score, reasons, topic, best, now) -> SensorEvent:
        if best == "error" and topic:
            say = f"That error keeps coming back ({topic[:50]}) -- want me to take a look?"
        elif best == "question":
            say = "You've asked about this a few different ways -- want me to try another angle?"
        else:
            say = f"You've been at this a while ({reasons[-1]}) -- want a second pair of eyes?"
        strength = min(1.0, 0.6 + 0.4 * (score - self.stuck_bar) / max(1e-6, 1.0 - self.stuck_bar))
        return SensorEvent(
            source="flow", modality=Modality.TEXT, text=say, salience=0.6, addressed=False,
            ts=now, persistable=False,
            meta={"offer": "stuck_help", "offer_strength": round(strength, 3), "say": say,
                  "topic": topic, "reasons": "; ".join(reasons),
                  "accept_context": f"You offered help because: {'; '.join(reasons)}. "
                                    f"Topic: {topic or 'what they are working on'}."})

    def _free_event(self, now) -> SensorEvent:
        mins = (self.last_active - (self.burst_start or self.last_active)) / 60.0
        where = where_phrase(self.process, self.subject, self.project)
        return SensorEvent(
            source="flow", modality=Modality.TEXT, ts=now, persistable=False, addressed=False,
            text=f"the user just finished about {mins:.0f} minutes of work{where} and paused",
            salience=0.5,
            meta={"offer": "free_moment", "offer_strength": round(min(0.8, 0.6 + 0.01 * mins), 3),
                  "minutes": round(mins, 1), "process": self.process,
                  "subject": self.subject, "project": self.project,
                  "accept_context": f"You had just checked in after {mins:.0f} minutes of their work{where}."})

    def status(self) -> Dict[str, Any]:
        return {"emitted": dict(self.emitted), "process": self.process,
                "subject": self.subject, "project": self.project,
                "tracked_errors": len(self.errors)}
