"""
src/brain/gist.py

Deciding what to say before deciding how to say it.

A Gist is the brain's committed intent for a reply: the speech act, the one to
three points that have to land, the facts that must survive, the tone. Wording
is a later, separate step -- which buys three things:

  * Plain cases need no language model. A blocked action, an ambiguous
    request, a failed tool and a short factual result all have an obvious
    sentence, so `render_gist()` produces it directly. That is the common case
    for an always-on assistant and it costs microseconds, not a model call
    that may take minutes on a small local machine.
  * The rest are checkable. When a model does the phrasing, `verify()` asks
    whether the reply actually carries the gist: required facts present,
    misleading phrases absent, meaning close to what was intended. A reply
    that fails gets one retry told exactly what was wrong, then a safe
    fallback -- never an unchecked claim that an action happened when it did
    not.
  * Open conversation is left alone. Intent "converse" carries no points and
    no checks; free generation stays free.

Everything here is rule-based and pure: no model, no I/O. An embedding
function may be passed in for the meaning check; without one that check is
skipped.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .signals import Gist, Point, Stance

# How close, in embedding space, a reply must stay to the intended meaning. A
# floor against replies that wandered off, not a measure of quality. Measured
# with MiniLM on a handful of pairs: on-topic paraphrases scored 0.24-0.72 and
# off-topic replies 0.01-0.05, so 0.12 sits in the gap. Tunable per machine.
GIST_MIN_COS = float(os.environ.get("ASH_GIST_MIN_COS", "0.12"))

# Phrases that would misstate what happened when the action did NOT succeed.
_FALSE_SUCCESS = ["successfully", "all done", "i've done", "i have done", "completed", "done!"]

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_DATE = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b")
_INT = re.compile(r"(?<![\d.])\d+(?!\d|\.\d)")
_PLAIN_MAX = 160


def readable(intent: Optional[str]) -> str:
    return (intent or "that").replace("_", " ").strip() or "that"


def number_words(n: int) -> List[str]:
    """The spoken forms of 0..999, as a model would plausibly write them."""
    if n < 0 or n > 999:
        return []
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [_TENS[t]] if o == 0 else [f"{_TENS[t]}-{_ONES[o]}", f"{_TENS[t]} {_ONES[o]}"]
    h, rest = divmod(n, 100)
    base = f"{_ONES[h]} hundred"
    if rest == 0:
        return [base]
    out = []
    for r in number_words(rest):
        out += [f"{base} and {r}", f"{base} {r}"]
    return out


def extract_facts(body: str, limit: int = 5) -> List[str]:
    """Whole numbers in a tool's output that a reply must not lose.

    Timestamps, clock times and dates are skipped: those are meant to be
    SPOKEN ("ten past two"), so demanding the digits would reject exactly the
    natural phrasing we want. Decimals and anything over 999 are skipped too,
    because they have no single reliable spoken form to check for.
    """
    s = _ISO.sub(" ", body or "")
    s = _DATE.sub(" ", s)
    s = _CLOCK.sub(" ", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    out: List[str] = []
    for m in _INT.findall(s):
        if len(m) > 3 and int(m) > 999:
            continue
        v = str(int(m)) if m.isdigit() else m
        if v not in out:
            out.append(v)
        if len(out) >= limit:
            break
    return out


def contains_fact(text: str, fact: str) -> bool:
    low = (text or "").lower()
    flat = re.sub(r"(?<=\d),(?=\d{3})", "", low)
    if re.search(rf"(?<![\d.]){re.escape(fact)}(?!\d|\.\d)", flat):
        return True
    if fact.isdigit():
        for w in number_words(int(fact)):
            if re.search(rf"\b{re.escape(w)}\b", low):
                return True
    return False


def is_plain(body: str) -> bool:
    """Short, human-readable text that can be said as it stands."""
    b = (body or "").strip()
    if not b or len(b) > _PLAIN_MAX or "\n" in b:
        return False
    if b[0] in "{[" or "http" in b.lower():
        return False
    return not (_ISO.search(b) or _CLOCK.search(b) or _DATE.search(b))


def _datumy(body: str) -> bool:
    """Data rather than prose: a spoken paraphrase of a timestamp or a run of
    numbers sits far from the written form in embedding space, so a meaning
    check on such a result would reject exactly the natural reply."""
    b = body or ""
    if _ISO.search(b) or _CLOCK.search(b) or _DATE.search(b):
        return True
    return bool(b) and sum(c.isdigit() for c in b) / len(b) > 0.3


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def stance_from_drives(drives: Any) -> Stance:
    return Stance(
        warmth=_clip(getattr(drives, "social", 0.5)),
        energy=_clip(1.0 - getattr(drives, "fatigue", 0.0)),
        brevity=_clip(getattr(drives, "urgency", 0.5)),
    )


_VETO_REASON = {
    "constitution": "it goes against one of my hard rules",
    "vla": "I can't see enough to do that safely",
}


def _veto_reason(vetoed_by: Sequence[str]) -> str:
    for who in vetoed_by:
        if who in _VETO_REASON:
            return _VETO_REASON[who]
    return "the board blocked it"


def build_gist(*, chosen=None, tool_out=None, tool_ok: Optional[bool] = None,
               vetoed_by: Sequence[str] = (), deadlocked: bool = False,
               attempted=None, alternatives: Sequence[Optional[str]] = (),
               drives: Any = None, body: str = "",
               embed_fn: Optional[Callable[[str], Any]] = None) -> Gist:
    """Decide what the reply is for, from what actually happened this turn.

    `body` is the tool output rendered as text. Order matters: a tool that ran
    is reported as it went; only if nothing ran does a veto or an ambiguity
    become the point of the reply.
    """
    stance = stance_from_drives(drives) if drives is not None else Stance()
    conf = float(getattr(chosen, "confidence", 0.5) or 0.5) if chosen is not None else 0.5
    ran = tool_ok is not None

    if ran and tool_ok:
        say = body.strip() if is_plain(body) else None
        g = Gist(
            intent="inform",
            points=[Point(f"report the result: {body.strip()[:300]}", "answer", say)],
            stance=stance, must_include=extract_facts(body), confidence=conf)
        if stance.warmth > 0.65 and stance.brevity <= 0.65:
            g.follow_up = "anything else you need on that?"
    elif ran:
        err = ""
        if isinstance(tool_out, dict):
            err = str(tool_out.get("error") or "").strip()
        say = f"That didn't work{': ' + err if err else ''}."
        g = Gist(
            intent="report_failure",
            points=[Point("say plainly that it did not work" + (f" and why: {err}" if err else ""),
                          "answer", say if is_plain(err or "x") else None)],
            stance=stance, must_avoid=list(_FALSE_SUCCESS), confidence=conf)
    elif vetoed_by:
        what = readable(getattr(attempted, "intent", None))
        reason = _veto_reason(vetoed_by)
        g = Gist(
            intent="refuse",
            points=[Point(f"say you did not do '{what}'", "answer", f"I didn't {what}."),
                    Point(f"explain why: {reason}", "reason", f"{reason[0].upper()}{reason[1:]}.")],
            stance=stance, must_avoid=list(_FALSE_SUCCESS), confidence=1.0)
    elif deadlocked:
        opts = [readable(a) for a in alternatives if a][:2]
        if len(opts) == 2:
            # Two concrete readings: the question writes itself.
            point = Point(f"ask which they meant: {opts[0]} or {opts[1]}", "question",
                          f"Did you mean {opts[0]}, or {opts[1]}?")
        else:
            # Nothing concrete to offer. A canned "say more" would be worse than
            # the model's question, which can use the conversation -- so leave
            # the sentence to it and only fix the purpose.
            point = Point("ask one short clarifying question about what they want", "question")
        g = Gist(intent="ask", points=[point], stance=stance, confidence=0.4)
    else:
        return Gist(intent="converse", stance=stance, confidence=conf)

    # The meaning check compares a reply with what the reply is ABOUT, so the
    # vector comes from the content itself -- never from the brief written for
    # the phrasing model ("ask one clarifying question..."), which is an
    # instruction, not something a good reply would resemble.
    if embed_fn is not None and g.intent == "inform" and not _datumy(body):
        try:
            g.vector = embed_fn(body.strip()[:300])
        except Exception:
            g.vector = None
    return g


def render_gist(gist: Gist) -> Optional[str]:
    """The reply, with no model -- or None when a person-sounding phrasing is
    needed (open conversation, or a point with no ready sentence)."""
    if gist is None or gist.intent == "converse" or not gist.points:
        return None
    if any(not p.say for p in gist.points):
        return None
    text = " ".join(p.say.strip() for p in gist.points)
    if gist.follow_up:
        text = f"{text} — {gist.follow_up}"
    return text


def tone_words(stance: Stance) -> str:
    warm = "warm" if stance.warmth > 0.65 else ("reserved" if stance.warmth < 0.35 else "even")
    length = "very brief" if stance.brevity > 0.65 else "brief"
    return f"{warm}, {length}"


def gist_prompt(gist: Gist) -> str:
    """The brief handed to the model that does the phrasing."""
    lines = ["[WHAT TO SAY] You have already decided what this reply is for. "
             "Say it in your own natural spoken voice; do not add other content.",
             f"- purpose: {gist.intent}"]
    for p in gist.points:
        lines.append(f"- {p.role}: {p.text}")
    lines.append(f"- tone: {tone_words(gist.stance)}")
    if gist.must_include:
        lines.append("- the reply MUST mention: " + ", ".join(gist.must_include))
    if gist.must_avoid:
        lines.append("- the reply must NOT say: " + ", ".join(f"'{a}'" for a in gist.must_avoid))
    if gist.follow_up:
        lines.append(f"- then, briefly: {gist.follow_up}")
    return "\n".join(lines)


def verify(text: str, gist: Gist, embed_fn: Optional[Callable[[str], Any]] = None,
           min_cos: float = GIST_MIN_COS) -> Tuple[bool, List[str]]:
    """Did the reply carry the gist? Returns (ok, problems)."""
    problems: List[str] = []
    t = (text or "").strip()
    if not t:
        return False, ["the reply was empty"]
    if gist is None or gist.intent == "converse":
        return True, []

    low = t.lower()
    for fact in gist.must_include:
        if not contains_fact(t, fact):
            problems.append(f"it left out '{fact}'")
    for bad in gist.must_avoid:
        if bad.lower() in low:
            problems.append(f"it said '{bad}', which misstates what happened")

    if embed_fn is not None and gist.vector is not None:
        try:
            v = embed_fn(t)
            g = np.asarray(gist.vector, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            if v.shape == g.shape:
                cos = float(np.dot(v, g) / max(float(np.linalg.norm(v) * np.linalg.norm(g)), 1e-9))
                if cos < min_cos:
                    problems.append("it drifted away from what was meant to be said")
        except Exception:
            pass
    return (not problems), problems
