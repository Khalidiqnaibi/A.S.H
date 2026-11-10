"""
src/daemon/uxreport.py

What ASH has learned about when it is welcome -- said back, and resettable.

ASH already learns this: every time it volunteers something, the social learner
(social.py) notes how the user reacted (a thanks, a "not now", a swipe, silence)
and shifts how readily it speaks up again, per kind of remark and per situation
(busy or idle, time of day, and what the user is doing: coding, gaming,
leisure...). That is a contextual bandit with a prior, a forgetting rate and a
cap on how far it can move; this makes it visible, because a thing that learns
from you ought to be able to say what it learned.

    "how are your nudges landing?"            welcome rate overall, best and weakest kind
    "when are you welcome / when should you stay quiet?"   by activity and time of day
    "forget how I react" / "reset what you learned about my reactions"

No model; arithmetic on the learner's own cells, with the evidence in the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_A = r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"\s*[.!?]*\s*$"
_RX = [(k, re.compile(p, re.I)) for k, p in (
    ("reset", _A + r"(?:forget|reset|clear|unlearn)\s+(?:how\s+i\s+react|what\s+you(?:'ve|\s+have)?\s+learn(?:ed|t)\s+(?:about\s+)?(?:my\s+reactions|when\s+to\s+(?:talk|speak)(?:\s+to\s+me)?)|"
                    r"(?:my\s+)?reactions|(?:your\s+)?(?:nudge|reaction)\s+(?:learning|history))" + _Z),
    ("report", _A + r"(?:how\s+(?:are|were)\s+(?:your\s+)?(?:nudges|remarks|interruptions|suggestions|comments)\s+(?:landing|going|received)|"
                    r"how\s+welcome\s+(?:are|is)\s+(?:your\s+)?(?:nudges|remarks|interruptions)|"
                    r"what\s+(?:have\s+you|did\s+you)\s+learn(?:ed|t)?\s+about\s+(?:when\s+to\s+(?:talk|speak)(?:\s+to\s+me)?|my\s+reactions|how\s+i\s+react)|"
                    r"when\s+(?:am\s+i|are\s+you)\s+(?:welcome|most\s+welcome)|when\s+should\s+you\s+(?:stay\s+quiet|talk|speak)(?:\s+to\s+me)?)" + _Z),
)]

_KIND = {"moment": "free-moment remarks", "wonder": "questions", "briefing": "briefings", "care": "health nudges",
         "stuck": "offers of help", "joke": "jokes", "game": "game offers", "flow": "check-ins", "cheer": "cheering"}
_MIN_N = 3.0


def parse(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    for kind, rx in _RX:
        if rx.match(t):
            return kind
    return None


def _pct(p: float) -> str:
    return f"{int(round(100 * p))}%"


def _name(kind: str) -> str:
    return _KIND.get(kind, kind.replace("_", " "))


def describe(learner: Any, now: Optional[float] = None) -> str:
    if learner is None:
        return "I'm not learning from how you react -- that's switched off."
    if int(getattr(learner, "updates", 0)) < 5:
        return (f"I've only seen {int(getattr(learner, 'updates', 0))} reactions so far, which isn't enough to say what "
                f"lands. Each thanks, \"not now\" or silence after I speak up teaches me a little.")
    now = learner._now(now)
    kinds = sorted({k.split("|")[0] for k in learner.cells if k != learner.GLOBAL and "|" not in k})
    rows: List[Tuple[str, float, float]] = []
    for k in kinds:
        n = learner.evidence(k, "", now)
        if n >= _MIN_N:
            rows.append((k, learner.p_kind(k, now), n))
    out = [f"About {_pct(learner.p_general(now))} of what I volunteer lands well, from {int(learner.updates)} reactions."]
    if len(rows) >= 2:
        rows.sort(key=lambda r: -r[1])
        best, worst = rows[0], rows[-1]
        if best[1] - worst[1] >= 0.1:
            out.append(f"Best: {_name(best[0])} ({_pct(best[1])}). Weakest: {_name(worst[0])} ({_pct(worst[1])}).")
    # by what the user was doing, across kinds: each "kind|@activity" cell holds reward and count
    acts: Dict[str, List[float]] = {}
    for key in learner.cells:
        if "|@" in key:
            a, n = learner._cell(key, now)
            act = key.split("|@", 1)[1]
            s = acts.setdefault(act, [0.0, 0.0])
            s[0] += a
            s[1] += n
    found = sorted(((act, s[0] / s[1], s[1]) for act, s in acts.items() if s[1] >= 5), key=lambda r: -r[1])
    if len(found) >= 2 and found[0][1] - found[-1][1] >= 0.1:
        hi, lo = found[0], found[-1]
        out.append(f"I'm most welcome while you're {_act(hi[0])} ({_pct(hi[1])} of {int(hi[2])}) and least while {_act(lo[0])} "
                   f"({_pct(lo[1])} of {int(lo[2])}).")
    snoozed = [k for k in getattr(learner, "snoozes", {}) if learner.snoozed(k, now)]
    if snoozed:
        out.append("I'm holding off on " + ", ".join(_name(k) for k in snoozed[:3]) + " for now because you said so.")
    out.append("Say \"forget how I react\" and I start again from neutral.")
    return " ".join(out)


def _act(a: str) -> str:
    return {"leisure": "relaxing online", "gaming": "gaming", "coding": "coding", "writing": "writing", "working": "working",
            "reading": "reading", "designing": "designing", "studying": "studying"}.get(a, a)


def reset(learner: Any) -> str:
    if learner is None:
        return "I'm not learning from how you react, so there's nothing to forget."
    n = int(getattr(learner, "updates", 0))
    learner.cells.clear()
    learner.snoozes.clear()
    learner.updates = 0
    learner.save()
    return f"Done -- I've forgotten what I'd learned from {n} reactions and I'm starting from neutral." if n else \
        "I hadn't learned anything yet, so I'm already neutral."
