"""
src/daemon/context.py

Context carry-over: "it", "that" and "who was that?", resolved against what
was just said or just on screen.

    open it / close that / switch to it   -> the app last opened, closed or
                                             in front (not ASH's own window)
    remind me about that [later|in ...]   -> what you said last: "I need to
                                             call the bank" becomes "remind me
                                             to call the bank in 1 hour"
    look it up / look that up             -> the last thing talked about
    who was that? / who's that?           -> the last person named, and who
                                             they are to you, if ASH knows
    say that again / what did you say?    -> ASH's last line, again

Rule-based. A sentence is rewritten into the full request and then goes
through the usual handlers ("open it" becomes "open Chrome", which the
desktop part opens), so nothing here does anything by itself except answer
"who was that" and "say that again". What it cannot resolve it leaves
alone: the brain may still make sense of it.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

_LEAD = r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?ash[,\s]+)?(?:(?:please|can you|could you|would you)[,\s]+)*"
_END = r"(?:[,\s]+(?:please|ash|again|now))*\s*[.!?]*\s*$"

_APP = re.compile(_LEAD + r"(?P<verb>open|launch|start|close|quit|exit|switch\s+to|bring\s+up|go\s+to)\s+"
                  r"(?:it|that|this|that\s+app|the\s+same\s+(?:app|one))" + _END, re.I)
_REMIND = re.compile(_LEAD + r"remind\s+me\s+(?:about|of)\s+(?:that|it|this)(?P<when>\s+.*?)?\s*[.!]?\s*$", re.I)
_LOOKUP = re.compile(_LEAD + r"(?:look\s+(?:it|that|this)\s+up|look\s+up\s+(?:it|that|this)|research\s+(?:that|it|this)|"
                     r"find\s+out\s+(?:more\s+)?about\s+(?:that|it|this))" + _END, re.I)
_WHO = re.compile(_LEAD + r"(?:who(?:'s|\s+is|\s+was)\s+(?:that|he|she|they|this)|who\s+did\s+you\s+(?:mean|say))" + _END,
                  re.I)
_AGAIN = re.compile(_LEAD + r"(?:say\s+(?:that|it)\s+again|repeat\s+(?:that|it|yourself)|what\s+did\s+you\s+(?:just\s+)?say|"
                    r"come\s+again|sorry\s*,?\s*what\s+was\s+that|pardon(?:\s+me)?|i\s+didn'?t\s+(?:catch|hear)\s+(?:that|you))"
                    + _END, re.I)

# What someone says they need to do, as the thing to be reminded of.
_TODO = re.compile(r"^\s*(?:i\s+(?:need|have|ought|want)\s+to|i\s+(?:should|must|gotta|got\s+to)|i'?m\s+supposed\s+to|"
                   r"(?:don'?t\s+)?(?:let\s+me\s+)?(?:forget|remember)\s+to|i\s+still\s+(?:need|have)\s+to|"
                   r"i'?ve\s+got\s+to)\s+(?P<what>.+?)\s*[.!]?\s*$", re.I)
_WHEN_WORDS = re.compile(r"\b(?:in\s+(?:an?\s+|\d+\s*)|at\s+\d|tomorrow|tonight|this\s+(?:evening|afternoon|morning)|"
                         r"on\s+(?:mon|tue|wed|thu|fri|sat|sun))", re.I)
_ABOUT = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:what(?:'s|\s+is|\s+are|\s+was|\s+were)|who(?:'s|\s+is|\s+was)|"
                    r"tell\s+me\s+about|what\s+do\s+you\s+know\s+about|how\s+(?:does|do|did)|where\s+is|"
                    r"when\s+(?:is|was|did))\s+(?P<t>.{2,80}?)\s*\??\s*$", re.I)
_LATER = re.compile(r"^\s*(?:later(?:\s+on)?|in\s+a\s+(?:bit|while)|some\s*time|afterwards?)?\s*(?:today)?\s*$", re.I)

_NOT_NAMES = {"I", "I'm", "I've", "I'll", "I'd", "ASH", "Ash", "OK", "Okay", "Monday", "Tuesday", "Wednesday",
              "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December", "God", "Sure", "Yes", "No", "The",
              "Hey", "Hi", "Hello", "Thanks", "What", "Who", "When", "Where", "Why", "How", "Please", "Sorry"}
_SELF_PROCS = ("python", "pythonw", "msedge", "ash", "windowsterminal", "cmd", "powershell", "explorer")


class Context:
    def __init__(self, clock: Callable[[], float] = time.time, knowledge=None, app_names: Optional[Dict[str, str]] = None,
                 window_s: float = 600.0):
        self.clock = clock
        self.kb = knowledge                         # wondering's Knowledge, for people
        self.app_names = app_names or {}            # process ("chrome.exe") -> name ("Chrome")
        self.window_s = window_s
        self.user: Deque[Tuple[float, str]] = deque(maxlen=8)    # what the user said, newest last
        self.ash: Optional[Tuple[float, str]] = None             # ASH's last line
        self.app: Optional[Tuple[float, str]] = None             # the app last opened, closed or in front
        self.people: Deque[Tuple[float, str]] = deque(maxlen=8)  # names, newest last
        self.stats = {"resolved": 0, "repeated": 0, "who": 0, "unresolved": 0}
        self.on_name: Optional[Callable[[str, float], None]] = None   # a name the user said (the weekly review)

    # -- noting ----------------------------------------------------------------
    def heard(self, text: str, now: Optional[float] = None):
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or self.is_referring(t):
            return
        self.user.append((now, t))
        self._names(t, now, user=True)

    def said(self, text: str, now: Optional[float] = None):
        now = self.clock() if now is None else now
        if text:
            self.ash = (now, text)
            self._names(text, now)
            m = re.match(r"^(?:Opening|Closing|Switching to)\s+(?P<app>[^.]{1,40})\.", text)
            if m:
                self.app = (now, m.group("app").strip())

    def window(self, process: str, now: Optional[float] = None):
        """The app in front, from the active-window sensor."""
        p = (process or "").lower()
        if not p or any(p.startswith(x) for x in _SELF_PROCS):
            return
        name = self.app_names.get(p) or re.sub(r"\.exe$", "", p)
        self.app = (self.clock() if now is None else now, name)

    def _names(self, text: str, now: float, user: bool = False):
        for i, w in enumerate(re.findall(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?\b", text)):
            first = w.split()[0]
            if first in _NOT_NAMES:
                continue
            known = False
            if self.kb is not None:
                try:
                    known = self.kb.person_seen(w) or self.kb.person_seen(first)
                except Exception:
                    known = False
            # a capital at the start of a sentence is not a name unless ASH knows it
            starts = re.search(r"(?:^|[.!?]\s+)" + re.escape(w) + r"\b", text) is not None
            if known or not starts:
                self.people.append((now, w))
                if user and self.on_name is not None:
                    try:
                        self.on_name(w, now)
                    except Exception:
                        pass

    # -- resolving -------------------------------------------------------------
    @staticmethod
    def is_referring(text: str) -> bool:
        t = text or ""
        return any(p.match(t) for p in (_APP, _REMIND, _LOOKUP, _WHO, _AGAIN))

    def _fresh(self, item: Optional[Tuple[float, str]], now: float) -> Optional[str]:
        return item[1] if item is not None and now - item[0] <= self.window_s else None

    def _topic(self, now: float) -> Optional[str]:
        for t, text in reversed(self.user):
            if now - t <= self.window_s and len(text.split()) >= 2:
                return text
        return None

    def _subject(self, now: float) -> Optional[str]:
        """What a recent question was about ("what's a quasar" -> "quasar"),
        or a short phrase said on its own; not a whole sentence."""
        for t, text in reversed(self.user):
            if now - t > self.window_s:
                break
            m = _ABOUT.match(text)
            if m:
                return m.group("t").strip()
            if len(text.split()) <= 4 and not re.match(r"^\s*(?:i|i'm|we|you|it|ok|okay|yes|no)\b", text, re.I):
                return text.rstrip(" .!?")
        return None

    def resolve(self, text: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """{"text": rewritten} to run as the full request, {"say": ...} to
        answer, or None if this is not a reference (or cannot be resolved)."""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or len(t) > 120:
            return None
        m = _AGAIN.match(t)
        if m:
            last = self._fresh(self.ash, now)
            if last is None:
                return None
            self.stats["repeated"] += 1
            return {"say": last, "repeat": True}
        m = _WHO.match(t)
        if m:
            for when, name in reversed(self.people):
                if now - when > self.window_s:
                    break
                self.stats["who"] += 1
                return {"say": self._who(name)}
            self.stats["unresolved"] += 1
            return None
        m = _APP.match(t)
        if m:
            app = self._fresh(self.app, now)
            if app is None:
                self.stats["unresolved"] += 1
                return None
            self.stats["resolved"] += 1
            return {"text": f"{m.group('verb').lower()} {app}"}
        m = _REMIND.match(t)
        if m:
            topic = self._topic(now)
            if topic is None:
                self.stats["unresolved"] += 1
                return None
            todo = _TODO.match(topic)
            what = ("to " + todo.group("what")) if todo else ("about " + topic.rstrip(" .!?"))
            when = (m.group("when") or "").strip()
            if _LATER.match(when) or not _WHEN_WORDS.search(when):
                when = "in 1 hour"
            self.stats["resolved"] += 1
            return {"text": f"remind me {what} {when}"}
        m = _LOOKUP.match(t)
        if m:
            topic = self._subject(now)
            if topic is None:
                self.stats["unresolved"] += 1
                return None
            self.stats["resolved"] += 1
            return {"text": "look up " + topic}
        return None

    def _who(self, name: str) -> str:
        rel, desc = "", ""
        if self.kb is not None:
            try:
                rel = self.kb.relation_of(name) or ""
                p = (getattr(self.kb, "people", {}) or {}).get(re.sub(r"\s+", " ", name.lower()).strip()) or {}
                desc = p.get("desc", "") or ""
            except Exception:
                pass
        if rel:
            return f"That was {name} -- your {rel}."
        if desc:
            return f"That was {name} -- {desc}."
        return f"That was {name}. I don't know who they are to you yet."

    def status(self) -> Dict[str, Any]:
        return {"app": self.app[1] if self.app else None, "people": [n for _, n in self.people][-3:], **self.stats}
