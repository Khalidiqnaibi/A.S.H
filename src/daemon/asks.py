"""
src/daemon/asks.py

What ASH asked, and what the user said back.

Before this, a question ASH put to the user lived for two minutes in the reaction tracker, was scored as "silence" if
nobody answered in time, and left no trace anywhere else: not in the memory the conversation reads, not in a file you
could open. So "I have a question for you when you have a minute" was followed, hours later, by "what questions do you
have?" -- and ASH had no idea it had said anything.

This keeps three things, all by rule:

  THE LEDGER    every ask, when it was made, in what situation (time of day, the program in front, how long the
                user had been in it, whether they were typing or reading), and what became of it: answered, said yes,
                said no, said later, moved on to something else, or nobody replied. One line per ask in
                state/asks/asks.jsonl (the reply is kept as said; a guest's never is).

  THE LINK      an ask stays open until it is settled -- minutes for a question, hours for "I have a question for you
                when you have a minute" -- so a yes, a "go ahead", or a bare "what is it?" is read as the answer it
                is, and the line the model is given says what it had asked. The pair also goes into episodic memory
                ("ASH asked: ... Immortal answered: yes"), which is where the conversation looks.

  THE TIMING    from every answer, how welcome an ask is in a situation: by part of the day, by the program in front,
                by typing or reading, by how long they have been at it. A no, a "later", or being ignored lowers it
                for that situation; a yes or an answer raises it. It is read back as a number the free-moment chooser
                multiplies in (`fit`), and in words ("when do you like to be asked things?").

Nothing here calls a model.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .social import classify_reply, friendly_app

logger = logging.getLogger("ash.daemon.asks")

# How long an ask stays open, by what it asked for.
WINDOW_S = {"answer": 180.0, "yes_no": 180.0, "permission": 8 * 3600.0}

# What a permission ask ("I have a question for you when you have a minute") is answered by, besides a plain yes.
_GO = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|sure|yes|yeah|yep|alright|fine|so)[,.!\s]+)*(?:go\s+(?:ahead|on)|ask(?:\s+(?:away|me))?|shoot|fire\s+away|"
    r"what(?:'s|\s+is|\s+was)?\s+(?:it|the\s+question|your\s+question|that|those)|what\s+(?:questions?|things?)\s+(?:do\s+you\s+have|did\s+you\s+have|have\s+you\s+got)|"
    r"what\s+(?:do\s+you\s+)?(?:want\s+to\s+(?:know|ask)|wanna\s+(?:know|ask))|i'?m\s+(?:listening|free|here)|now\s+is\s+fine|"
    r"now'?s\s+(?:good|fine|ok(?:ay)?)|i\s+have\s+a\s+minute|let'?s\s+hear\s+it)\b", re.I)

_PARTS = ((5, 12, "morning"), (12, 17, "afternoon"), (17, 22, "evening"))


def part_of_day(hour: int) -> str:
    for a, b, name in _PARTS:
        if a <= hour < b:
            return name
    return "night"


def idle_bucket(idle_s: Optional[float]) -> str:
    if idle_s is None:
        return "unknown"
    if idle_s < 5:
        return "typing"
    if idle_s < 120:
        return "pausing"
    return "reading"


def dwell_bucket(minutes: float) -> str:
    return "just_started" if minutes < 5 else "a_while" if minutes < 25 else "long"


def context_of(now: float, process: str = "", subject: str = "", idle_s: Optional[float] = None,
               dwell_min: float = 0.0, mood: Optional[float] = None) -> Dict[str, Any]:
    lt = time.localtime(now)
    return {"hour": lt.tm_hour, "part": part_of_day(lt.tm_hour), "weekend": lt.tm_wday >= 5,
            "app": friendly_app(process) or "", "subject": (subject or "")[:60], "mode": idle_bucket(idle_s),
            "dwell": dwell_bucket(dwell_min), "dwell_min": round(float(dwell_min), 1),
            "mood": None if mood is None else round(float(mood), 2)}


def keys_of(ctx: Dict[str, Any]) -> List[str]:
    """The situation at several levels of detail, coarsest first."""
    part, app, mode, dwell = ctx.get("part", ""), ctx.get("app", ""), ctx.get("mode", ""), ctx.get("dwell", "")
    out = ["*"]
    if part:
        out.append(f"part:{part}")
    if mode:
        out.append(f"mode:{mode}")
    if app:
        out.append(f"app:{app}")
    if dwell:
        out.append(f"dwell:{dwell}")
    if part and mode:
        out.append(f"part:{part}|mode:{mode}")
    if part and app:
        out.append(f"part:{part}|app:{app}")
    if app and mode:
        out.append(f"app:{app}|mode:{mode}")
    return out


# What each outcome says about the situation it was asked in: (reward 0..1, weight).
OUTCOME = {"answered": (1.0, 1.0), "accepted": (1.0, 1.0), "declined": (0.0, 2.0), "later": (0.15, 1.5),
           "moved_on": (0.4, 0.5), "ignored": (0.3, 0.6)}


class TimingModel:
    """How welcome an ask is, by situation. Each key is a count of welcomed and unwelcome, forgetting slowly."""

    def __init__(self, path: Optional[str] = None, decay: float = 0.97, clock: Callable[[], float] = time.time):
        self.path, self.decay, self.clock = path, float(decay), clock
        self.cells: Dict[str, List[float]] = {}          # key -> [welcomed, total]
        self.updates = 0
        self._load()

    def update(self, ctx: Dict[str, Any], outcome: str) -> None:
        if outcome not in OUTCOME:
            return
        reward, weight = OUTCOME[outcome]
        for k in keys_of(ctx):
            a, n = self.cells.get(k, [0.0, 0.0])
            self.cells[k] = [a * self.decay + weight * reward, n * self.decay + weight]
        self.updates += 1
        self.save()

    def p(self, ctx: Dict[str, Any]) -> Tuple[float, float]:
        """(welcome 0..1, evidence): the finer the situation, the more it counts once there is evidence for it."""
        prior, num, den = 0.55, 0.0, 0.0
        ev = 0.0
        keys = keys_of(ctx)
        specific = [k for k in keys if k != "*" and self.cells.get(k, [0, 0])[1] > 0]
        for k in (specific or ["*"]):
            a, n = self.cells.get(k, [0.0, 0.0])
            if n <= 0:
                continue
            w = n * (1.0 + 1.5 * k.count("|"))          # "at night in Edge" speaks louder than "at night"
            num += w * (a / n)
            den += w
            ev = max(ev, n)
        if den <= 0:
            return prior, 0.0
        est = num / den
        shrink = den / (den + 1.5)                       # little evidence: stay near the prior
        return prior + (est - prior) * shrink, ev

    def fit(self, ctx: Dict[str, Any]) -> float:
        """A multiplier for a free-moment candidate: 0.35 where asking has always been unwelcome, up to 1.5 where it is wanted."""
        p, ev = self.p(ctx)
        return 0.35 + 1.15 * p if ev >= 2.0 else 1.0

    def describe(self, min_evidence: float = 2.0) -> str:
        rows = []
        for k, (a, n) in self.cells.items():
            if k == "*" or n < min_evidence:
                continue
            rows.append((a / n, n, k))
        if not rows:
            return ""
        rows.sort()
        low = [r for r in rows if r[0] <= 0.35][:3]
        high = [r for r in rows[::-1] if r[0] >= 0.7][:3]

        def name(k: str) -> str:
            return " and ".join(x.split(":", 1)[1].replace("_", " ") for x in k.split("|"))
        parts = []
        if high:
            parts.append("You're usually fine being asked things " + "; ".join(f"{name(k)}" for _p, _n, k in high))
        if low:
            parts.append("I've learned to leave you alone " + "; ".join(f"{name(k)}" for _p, _n, k in low))
        return ". ".join(parts) + "."

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"v": 1, "cells": self.cells, "updates": self.updates}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save the timing record")

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.cells = {str(k): [float(v[0]), float(v[1])] for k, v in d.get("cells", {}).items()}
            self.updates = int(d.get("updates", 0))
        except Exception:
            logger.exception("Timing record unreadable; starting fresh")
            self.cells, self.updates = {}, 0


@dataclass
class Ask:
    id: int
    kind: str                       # wonder | moment | briefing | check_in | ...
    text: str
    ts: float
    expects: str = "answer"         # answer | yes_no | permission
    topic: str = ""
    ctx: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "open"            # open | answered | accepted | declined | later | moved_on | ignored
    reply: str = ""
    replied_at: float = 0.0


@dataclass
class Settled:
    ask: Ask
    outcome: str                    # answered | accepted | declined | later | moved_on | go (a permission taken up)
    reply: str = ""


class Asks:
    def __init__(self, directory: Optional[str] = None, clock: Callable[[], float] = time.time, episodic: Any = None,
                 user: str = "", scrub: Callable[[str], str] = lambda s: s, keep_replies: bool = True,
                 max_open: int = 6):
        self.dir = directory
        self.clock = clock
        self.episodic = episodic
        self.user = user or "The user"
        self.scrub = scrub
        self.keep_replies = bool(keep_replies)
        self.max_open = int(max_open)
        self.open: List[Ask] = []
        self.recent: List[Ask] = []                   # settled, newest last (kept in memory for the page and the reports)
        self._n = 0
        self._lock = threading.RLock()
        self.timing = TimingModel(os.path.join(directory, "timing.json") if directory else None, clock=clock)
        self.stats = {"asked": 0, "answered": 0, "accepted": 0, "declined": 0, "later": 0, "moved_on": 0, "ignored": 0}
        self._load_recent()

    # -- paths ----------------------------------------------------------------------------------------------
    @property
    def ledger_path(self) -> Optional[str]:
        return os.path.join(self.dir, "asks.jsonl") if self.dir else None

    def _append(self, row: Dict[str, Any]) -> None:
        p = self.ledger_path
        if not p:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Could not write the ask ledger")

    def _load_recent(self, n: int = 60) -> None:
        p = self.ledger_path
        if not p or not os.path.exists(p):
            return
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()[-400:]
            settled: Dict[int, Ask] = {}
            maxid = 0
            for ln in lines:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                maxid = max(maxid, int(r.get("id", 0)))
                if r.get("event") == "asked":
                    settled[r["id"]] = Ask(id=r["id"], kind=r.get("kind", ""), text=r.get("text", ""), ts=r.get("ts", 0.0),
                                           expects=r.get("expects", "answer"), topic=r.get("topic", ""), ctx=r.get("ctx", {}))
                elif r.get("event") == "settled" and r.get("id") in settled:
                    a = settled[r["id"]]
                    a.status, a.reply, a.replied_at = r.get("status", ""), r.get("reply", ""), r.get("ts", 0.0)
            self._n = maxid
            self.recent = [a for a in settled.values() if a.status != "open"][-n:]
        except Exception:
            logger.exception("Could not read the ask ledger")

    # -- asking ---------------------------------------------------------------------------------------------
    def asked(self, kind: str, text: str, now: Optional[float] = None, topic: str = "", expects: str = "answer",
              ctx: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Ask:
        now = self.clock() if now is None else now
        with self._lock:
            self._n += 1
            a = Ask(id=self._n, kind=kind, text=str(text)[:300], ts=now, expects=expects, topic=topic,
                    ctx=dict(ctx or {}), data=dict(data or {}))
            self.open.append(a)
            while len(self.open) > self.max_open:
                self._settle(self.open.pop(0), "ignored", "", now)
            self.stats["asked"] += 1
            self._append({"event": "asked", "id": a.id, "ts": now, "kind": kind, "text": a.text, "expects": expects,
                          "topic": topic, "ctx": a.ctx})
            return a

    def current(self, now: Optional[float] = None) -> Optional[Ask]:
        """The ask a reply right now would be an answer to: the latest still open."""
        now = self.clock() if now is None else now
        with self._lock:
            self.expire(now)
            return self.open[-1] if self.open else None

    def standing(self, now: Optional[float] = None) -> Optional[Ask]:
        """An open permission ask ("I have a question for you when you have a minute")."""
        now = self.clock() if now is None else now
        with self._lock:
            self.expire(now)
            for a in reversed(self.open):
                if a.expects == "permission":
                    return a
        return None

    def expire(self, now: float) -> List[Ask]:
        out = []
        with self._lock:
            for a in list(self.open):
                if now - a.ts > WINDOW_S.get(a.expects, 180.0):
                    self.open.remove(a)
                    self._settle(a, "ignored", "", now)
                    out.append(a)
        return out

    # -- the reply ------------------------------------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None, addressed: bool = True, guest: bool = False) -> Optional[Settled]:
        """Read a line from the user against what is open. None when nothing was asked, or the line was not for ASH."""
        now = self.clock() if now is None else now
        t = str(text or "").strip()
        if not t or not addressed:
            return None
        with self._lock:
            self.expire(now)
            if not self.open:
                return None
            a = self.open[-1]
            verdict = classify_reply(t)
            if a.expects == "permission":
                if _GO.match(t) or verdict == "accepted":
                    outcome = "accepted"
                elif verdict == "rejected":
                    outcome = "declined"
                elif verdict == "deferred":
                    outcome = "later"
                else:
                    return None                           # something else entirely: the ask stays standing
            else:
                short = len(t.split()) <= 8
                if verdict == "rejected" and short:
                    outcome = "declined"
                elif verdict == "deferred" and short:
                    outcome = "later"
                elif verdict == "accepted" and a.expects == "yes_no":
                    outcome = "accepted"
                elif t.endswith("?") or len(t.split()) > 25:
                    outcome = "moved_on"                  # they asked something of their own
                else:
                    outcome = "answered"
            self.open.remove(a)
            self._settle(a, outcome, t, now, guest=guest)
            return Settled(a, outcome, t)

    def _settle(self, a: Ask, outcome: str, reply: str, now: float, guest: bool = False) -> None:
        a.status = outcome if outcome in self.stats else "answered"
        a.reply = reply if (self.keep_replies and not guest) else ""
        a.replied_at = now
        self.stats[a.status] = self.stats.get(a.status, 0) + 1
        self.recent.append(a)
        del self.recent[:-60]
        if not guest:
            self.timing.update(a.ctx, a.status)
        self._append({"event": "settled", "id": a.id, "ts": now, "status": a.status, "reply": a.reply,
                      "latency_s": round(now - a.ts, 1)})
        if a.status in ("answered", "accepted", "declined", "later") and not guest:
            self._episode(a, now)

    def _episode(self, a: Ask, now: float) -> None:
        if self.episodic is None:
            return
        said = {"answered": "answered", "accepted": "said yes", "declined": "said no", "later": "said later"}[a.status]
        when = time.strftime("%H:%M", time.localtime(a.ts))
        where = f" while on {a.ctx.get('subject') or a.ctx.get('app')}" if (a.ctx.get("subject") or a.ctx.get("app")) else ""
        shown = f": {a.reply}" if a.reply and a.status == "answered" else ""
        try:
            self.episodic.add_episode(
                summary=self.scrub(f"A.S.H asked at {when}{where}: \"{a.text}\" -- {self.user} {said}{shown}"),
                event_type="ask", related_entities=[], importance=0.5 if a.status == "answered" else 0.35)
        except Exception:
            logger.exception("Could not write the ask to episodic memory")

    # -- what the ledger says -------------------------------------------------------------------------------
    def context_line(self, now: Optional[float] = None) -> str:
        """For the model: what ASH last asked, if it is still waiting for an answer."""
        a = self.current(now)
        if a is None:
            return ""
        return f"[You had asked the user: \"{a.text}\" -- a message now may be their answer to that.]"

    def history(self, n: int = 12) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [asdict(a) for a in self.open] + [asdict(a) for a in self.recent[-n:]]
        rows.sort(key=lambda r: r["ts"])
        return rows[-n:]

    def report(self) -> str:
        s = self.stats
        if not s["asked"]:
            return "I haven't asked you anything yet."
        bits = []
        for k, word in (("answered", "answered"), ("accepted", "said yes to"), ("declined", "said no to"),
                        ("later", "put off"), ("moved_on", "talked past"), ("ignored", "didn't reply to")):
            if s[k]:
                bits.append(f"you {word} {s[k]}")
        line = f"I've asked {s['asked']} thing{'s' if s['asked'] != 1 else ''}: " + ", ".join(bits) + "."
        learned = self.timing.describe()
        return line + (" " + learned if learned else " I haven't asked enough yet to know when you like to be asked.")

    def status(self) -> Dict[str, Any]:
        return {"open": len(self.open), "timing_updates": self.timing.updates, **self.stats}
