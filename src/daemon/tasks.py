"""
src/daemon/tasks.py

Multi-step tasks, with a plan to approve.

    "open Outlook, set a timer for 25 minutes and turn off the desk lamp"
        -> "Here's the plan: 1) open Outlook, 2) set a timer for 25 minutes,
            3) turn off the desk lamp. Go ahead?"
    "yes" -> each step is run by the part of ASH that already does it
             (desktop, reminders, media, smart home, watches, research), and
             what came of each is said back in one line.

    "when I say get me ready for the meeting, open the agenda, mute the
     sound and set a 25 minute timer"
        -> kept as a named task. Saying "get me ready for the meeting" later
           shows the same plan and asks once.

Rule-based. A request is a plan only when it splits into two or more parts
that each start like a command ASH can carry out ("open", "set a timer",
"turn off", "remind me" ...). A part that does not ("... and eggs") is put
back onto the one before it, so "remind me to buy milk and eggs" stays one
reminder. Nothing runs before a yes; a plan not answered within a minute is
dropped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.tasks")

_LEAD = re.compile(r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?ash[,\s]+)?(?:(?:please|can you|could you|would you)[,\s]+)*",
                   re.I)
_VERB = re.compile(r"^\s*(?:please\s+)?(?:open|launch|start|fire\s+up|bring\s+up|close|quit|exit|set|remind|wake|"
                   r"play|pause|resume|unpause|stop|next|skip|turn|switch|put|dim|brighten|mute|unmute|volume|"
                   r"take|snap|tell\s+me\s+when|let\s+me\s+know\s+when|ping\s+me\s+when|look\s+up|research|"
                   r"find\s+out|run)\b", re.I)
_SEP = re.compile(r"\s*(?:,\s*(?:and\s+)?then\s+|,\s*and\s+|\s+and\s+then\s+|\s+then\s+|;\s*|,\s+|\s+and\s+)", re.I)
_TEACH = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:when(?:ever)?|if)\s+i\s+say\s+"
                    r"(?:[\"'“](?P<qname>[^\"'”]{3,50})[\"'”]\s*[,:]?|"
                    r"(?P<name>[^,\"'”]{3,50}?)\s*(?:,|:|\s+(?=you\s+should|please|then|just|i\s+want\s+you\s+to)))\s*"
                    r"(?:you\s+should\s+|please\s+|then\s+|just\s+|i\s+want\s+you\s+to\s+)?(?P<steps>.+)$", re.I)
_LIST = re.compile(r"^\s*(?:what|which)\s+(?:tasks|routines|shortcuts)\s+(?:do\s+you\s+know|have\s+i\s+(?:set|taught\s+you))"
                   r"\s*\??\s*$|^\s*list\s+(?:my\s+)?tasks\s*$", re.I)
_FORGET = re.compile(r"^\s*(?:forget|delete|remove)\s+(?:the\s+)?(?:task\s+)?[\"'“]?(?P<name>[^\"'”]{3,50}?)"
                     r"[\"'”]?(?:\s+task)?\s*[.!]?\s*$", re.I)
_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|go(?:\s+ahead)?|do\s+it|please\s+do|go\s+for\s+it|"
                  r"sounds\s+good|correct)\b", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|don'?t|cancel|never\s*mind|stop|not\s+now|wait)\b", re.I)

# "Save that as start work": what was just done, kept under a name (a macro).
_NUMW = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a few": 3, "few": 3}
_WHAT = r"(?P<what>that|this|it|those|the\s+last\s+(?:(?P<n>\d|two|three|four|five|a\s+few)\s+)?(?:things?|steps?|commands?|actions?))"
_NAME = r"[\"'“]?(?P<name>[a-z][a-z0-9 '\-]{2,40}?)[\"'”]?\s*[.!]?\s*$"
_SAVE = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?(?:save|keep|remember|store|make|turn)\s+" + _WHAT +
                   r"\s+(?:as|into)\s+(?:an?\s+)?(?:(?:macro|shortcut|task|routine)\s+)?(?:(?:called|named)\s+)?" + _NAME, re.I)
_SAVE2 = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?make\s+" + _WHAT +
                    r"\s+(?:an?\s+)?(?:macro|shortcut|task|routine)\s+(?:called|named)\s+" + _NAME, re.I)
RECENT_S = 600.0

MAX_STEPS = 8


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def split_steps(text: str) -> Optional[List[str]]:
    """The steps in a compound command, or None if it is not one."""
    t = (text or "").strip()
    if not t or len(t) > 240 or t.endswith("?"):
        return None
    t = _LEAD.sub("", t).strip().rstrip(".!")
    parts: List[str] = []
    pos = 0
    for m in _SEP.finditer(t):
        parts.append(t[pos:m.start()])
        parts.append(m.group(0))
        pos = m.end()
    parts.append(t[pos:])
    steps: List[str] = []
    for i in range(0, len(parts), 2):
        frag = parts[i].strip()
        if not frag:
            continue
        if steps and not _VERB.match(frag):
            steps[-1] = steps[-1] + parts[i - 1] + frag        # "... milk and eggs": back where it was
        else:
            steps.append(frag)
    if len(steps) < 2 or len(steps) > MAX_STEPS or not all(_VERB.match(s) for s in steps):
        return None
    return steps


def describe(steps: List[str]) -> str:
    return ", ".join(f"{i}) {s}" for i, s in enumerate(steps, 1))


class Tasks:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 confirm_window_s: float = 60.0, enabled: bool = True):
        self.path = path
        self.clock = clock
        self.enabled = enabled
        self.confirm_window_s = confirm_window_s
        self.saved: Dict[str, Dict[str, Any]] = {}       # norm(name) -> {"name", "steps", "runs"}
        self.pending: Optional[Dict[str, Any]] = None    # {"steps", "t", "name"}
        self.stats = {"planned": 0, "run": 0, "declined": 0, "taught": 0, "kept": 0}
        self.last_run: Optional[tuple] = None             # (steps, t) of the plan just approved
        self.recent: List[tuple] = []                      # (t, text) of single commands just given
        self._lock = threading.RLock()
        self._load()

    # -- persistence -----------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.saved = {_norm(k): {"name": str(v["name"]), "steps": [str(s) for s in v["steps"]],
                                     "runs": int(v.get("runs", 0))}
                          for k, v in (raw.get("tasks") or {}).items() if v.get("steps")}
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with self._lock, open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"tasks": self.saved}, fh, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    # -- hearing ---------------------------------------------------------------
    def awaiting(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = self.clock() if now is None else now
        p = self.pending
        return p if p is not None and now - p["t"] <= self.confirm_window_s else None

    def hear(self, text: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """What to do about `text`:
            {"say": ...}                     a reply, nothing to run
            {"run": [steps], "name": ...}    approved: run these now
        or None if it is not about tasks."""
        if not self.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t:
            return None

        p = self.awaiting(now)
        if p is not None:
            self.pending = None
            if _YES.match(t):
                self.stats["run"] += 1
                self.last_run = (list(p["steps"]), now)
                if p.get("name"):
                    with self._lock:
                        rec = self.saved.get(_norm(p["name"]))
                        if rec is not None:
                            rec["runs"] = rec.get("runs", 0) + 1
                    self.save()
                return {"run": list(p["steps"]), "name": p.get("name")}
            if _NO.match(t):
                self.stats["declined"] += 1
                return {"say": "Okay, I won't."}
            # Anything else: the plan is dropped and this is heard as itself.
        else:
            self.pending = None

        m = _SAVE.match(t) or _SAVE2.match(t)
        if m:
            return self.save_recent(m.group("name"), m.group("what"), m.group("n"), now)
        m = _TEACH.match(t)
        if m:
            name = (m.group("qname") or m.group("name")).strip()
            steps = split_steps(m.group("steps")) or (
                [m.group("steps").strip().rstrip(".!")] if _VERB.match(m.group("steps")) else None)
            if not steps:
                return {"say": "I can only keep steps I can do myself -- opening and closing apps, timers and "
                               "reminders, music, lights, watches."}
            with self._lock:
                self.saved[_norm(name)] = {"name": name, "steps": steps, "runs": 0}
            self.save()
            self.stats["taught"] += 1
            return {"say": f"Got it. When you say “{name}”, I'll {describe(steps)} -- "
                           f"after checking with you."}
        if _LIST.match(t):
            with self._lock:
                rows = list(self.saved.values())
            if not rows:
                return {"say": "You haven't taught me any tasks yet. Try: “when I say start work, open "
                               "Outlook and set a timer for 25 minutes”."}
            return {"say": "I know: " + "; ".join(f"“{r['name']}” ({len(r['steps'])} steps)"
                                                  for r in rows) + "."}
        m = _FORGET.match(t)
        if m:
            with self._lock:
                rec = self.saved.pop(_norm(m.group("name")), None)
            if rec is not None:
                self.save()
                return {"say": f"Forgotten: “{rec['name']}”."}
            # not one of ours: "forget Omar" is for forget.py
        key = _norm(_LEAD.sub("", t))
        with self._lock:
            rec = self.saved.get(key)
        if rec is not None:
            return self._plan(rec["steps"], now, name=rec["name"])
        steps = split_steps(t)
        if steps:
            return self._plan(steps, now)
        self._note(t, now)
        return None

    def _note(self, text: str, now: float):
        """A single command, remembered for a few minutes so "save that as ..." has something to save."""
        t = _LEAD.sub("", text).strip().rstrip(".!")
        if t and not t.endswith("?") and len(t) <= 120 and _VERB.match(t):
            self.recent = [(a, b) for a, b in self.recent if now - a <= RECENT_S][-9:] + [(now, t)]

    def save_recent(self, name: str, what: str, n: Optional[str], now: float) -> Dict[str, Any]:
        """Keep what was just done under a name. `that` is the plan just run, or
        the last command; "the last three things" is the last three commands."""
        what = (what or "").lower()
        steps: List[str] = []
        if what.startswith("the last"):
            k = _NUMW.get((n or "").lower().strip()) or (int(n) if (n or "").isdigit() else 1 if not n else 2)
            steps = [b for a, b in self.recent if now - a <= RECENT_S][-k:]
        elif self.last_run is not None and now - self.last_run[1] <= RECENT_S and (
                not self.recent or self.last_run[1] >= self.recent[-1][0]):
            steps = list(self.last_run[0])
        else:
            steps = [b for a, b in self.recent if now - a <= RECENT_S][-1:]
        if not steps:
            return {"say": "I haven't seen anything you did just now that I can keep. Do it first, then say "
                           "“save that as ...”."}
        name = name.strip()
        with self._lock:
            self.saved[_norm(name)] = {"name": name, "steps": steps, "runs": 0}
        self.save()
        self.stats["kept"] += 1
        return {"say": f"Saved. When you say “{name}”, I'll {describe(steps)} -- after checking with you.",
                "saved": name}

    def hear_is_task(self, text: str) -> bool:
        """Would `text` start a plan or a kept task (without changing anything)?"""
        t = (text or "").strip()
        if not self.enabled or not t:
            return False
        with self._lock:
            if _norm(_LEAD.sub("", t)) in self.saved:
                return True
        return bool(_TEACH.match(t) or _SAVE.match(t) or _SAVE2.match(t) or split_steps(t))

    def _plan(self, steps: List[str], now: float, name: Optional[str] = None) -> Dict[str, Any]:
        self.pending = {"steps": list(steps), "t": now, "name": name}
        self.stats["planned"] += 1
        return {"say": f"Here's the plan: {describe(steps)}. Go ahead?", "asking": True}

    @staticmethod
    def summary(results: List[Dict[str, Any]]) -> str:
        """One line on what came of each step."""
        done = [r["said"].rstrip(".") for r in results if r["said"]]
        missed = [r["step"] for r in results if not r["said"]]
        out = ". ".join(done) + "." if done else ""
        if missed:
            out += (" " if out else "") + "I couldn't do: " + ", ".join(missed) + "."
        return out or "Nothing came of that."

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "saved": sorted(r["name"] for r in self.saved.values()),
                    "pending": self.pending is not None, **self.stats}
