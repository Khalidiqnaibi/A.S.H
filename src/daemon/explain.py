"""
src/daemon/explain.py

"Why did you say that?"

Every time ASH says something, the ambient loop notes why, as a small record
(what kind of remark it was, and the facts behind it). Asked "why did you say
that?", "why are you asking?" or "why did you do that?", ASH answers from
that record in plain words, by rule:

    an answer      what was asked, what ASH read it as, which tool it used,
                   and whether it came from a quick reflex or the model
    a question     what it did not know and how it noticed (mentioned three
                   times, two answers that clash, a routine seen on five days)
    a remark       what made the moment seem right (a pause after forty
                   minutes of work, the first time at the computer today)
    an action      that the user asked for it

The full decision trace (`ashctl.py why`) stays for debugging; this is the
answer a person would give.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_ASKED = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:but\s+|so\s+)?(?:why|how come)\s+(?:(?:did|do|are|would|were)\s+)?you\s+"
    r"(?:say(?:ing)?|said|ask(?:ing|ed)?|do(?:ing)?|did|tell(?:ing)? me|told me|bring(?:ing)? (?:that|it) up|"
    r"brought (?:that|it) up|mention(?:ing|ed)?|suggest(?:ing|ed)?|open(?:ing|ed)?|want(?:ed)? to know)\b"
    r"(?:\s+(?:that|this|it|me that|so))?\s*\??\s*$"
    r"|^\s*(?:why|how come)\s*\??\s*$"
    r"|^\s*what made you (?:say|ask|do) (?:that|this)\s*\??\s*$", re.I)


def asked_why(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and bool(_ASKED.match(t))


_PATHWAY = {"fast": "a quick reflex, without the language model",
            "reflex": "a quick reflex, without the language model",
            "slow": "thinking it through with the language model",
            "escalated": "thinking it through with the language model, after a quick answer looked wrong",
            "offer": "my own rules, without the language model"}

_WONDER_WHY = {
    "person": "you'd mentioned {subject}{times} and I didn't know who that is",
    "role": "you mentioned {subject} and I don't know their name",
    "between": "you mentioned {subject} together{times} and I didn't know how they know each other",
    "term": "you used \"{subject}\"{times} and I didn't know what it means",
    "fact": "you mentioned {subject} and I don't have it",
    "ambiguous": "I know more than one {subject} and wasn't sure which you meant",
    "intent": "I couldn't tell what you meant earlier",
    "activity": "you keep coming back to {subject} and I don't know what it's for",
    "conflict": "two things you told me about {subject} don't match",
    "stale": "it's been a long while since you told me about {subject}",
    "routine": "you've opened those apps in that order on several different days",
}

_MOMENT_WHY = {
    "joke": "you'd just finished a stretch of work and things seemed to be going fine, "
            "so it seemed a good moment for something light",
    "mood_check": "I hadn't checked in on how you were doing today",
    "system_assessment": "something had gone wrong inside me since I last mentioned it",
    "follow_up": "you'd told me about it before it happened, and it's over now",
    "research": "you'd asked me to look into it, and I'd finished",
}


_SOURCE = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:but|so|and|ok(?:ay)?)\s+)?"
    r"(?:where\s+(?:did|do|does)\s+(?:you|u)\s+(?:get|got|find|learn|know)\s+(?:that|this|it|those|these|the)?(?:\s*(?:info(?:rmation)?|answer|stuff|from))?"
    r"|how\s+(?:do|did)\s+(?:you|u)\s+know\s+(?:that|this|it)"
    r"|what(?:'s|\s+is|\s+was)\s+(?:your|the)\s+(?:source|sources)(?:\s+for\s+(?:that|this|it))?"
    r"|(?:where|what)\s+(?:is|was)\s+(?:that|this|it)\s+from"
    r"|who\s+told\s+you\s+(?:that|this)"
    r"|(?:are|is)\s+(?:you|that|this|it)\s+sure(?:\s+about\s+that)?)"
    r"(?:\s+(?:from|info(?:rmation)?|that|this|it))*\s*\??\s*$", re.I)


def asks_source(text: str) -> bool:
    """"Where did you get that?": asked about the last answer's source."""
    t = (text or "").strip()
    return bool(t) and bool(_SOURCE.match(t))


def source_reply(why: Optional[Dict[str, Any]]) -> str:
    """Where the last thing said really came from -- the tool, the model's own
    knowledge, or ASH's own rules -- and never the window that was in front."""
    if not why:
        return "I haven't said anything yet that I could trace."
    kind = why.get("kind", "")
    if kind == "answer":
        tool = str(why.get("tool") or "").replace("_", " ")
        how = str(why.get("pathway", ""))
        if tool:
            return (f"From the {tool} tool, which ran when you asked. "
                    + ("It's a quick reflex, not the language model." if how in ("fast", "reflex") else
                       "The language model then put it into words."))
        if how == "reflex":
            return "That was a repeat of what I'd just said, not a new look-up."
        return ("That was my language model answering from what it was trained on -- I didn't look it up or take "
                "it from your files or memory. It can be out of date or wrong, so check it if it matters.")
    if kind == "answered":
        return "That came from what you'd just told me, not from the language model."
    if kind in ("asked", "watch", "stopped", "correction"):
        return "That was my own rules acting on what you asked -- the language model wasn't involved."
    if kind == "facts":
        return "That was the first paragraph of Wikipedia's page on it, fetched when you asked."
    if kind == "notes":
        return "That came from what you asked me to remember, kept on this computer -- not from the language model."
    return "That was something I noticed myself, from your activity on the computer -- not from the language model."


def explain(why: Optional[Dict[str, Any]]) -> str:
    """One or two sentences saying why ASH said or did its last thing."""
    if not why:
        return "I haven't said anything yet that I can explain."
    kind = why.get("kind", "")
    if kind == "answer":
        q = str(why.get("query", "")).strip()
        intent = str(why.get("intent") or "")
        tool = str(why.get("tool") or "")
        how = _PATHWAY.get(str(why.get("pathway", "")), "")
        if intent and intent != "conversation":
            s = f"You asked \"{q}\", and I read it as {intent.replace('_', ' ')}"
            s += f", so I used the {tool.replace('_', ' ')} tool." if tool else "."
        else:
            s = f"You said \"{q}\", and I took it as conversation, so I just replied."
        return s + (f" The reply came from {how}." if how else "")
    if kind == "wonder":
        w = why.get("wonder") or {}
        n = int(w.get("mentions", 1) or 1)
        times = f" {n} times" if n > 2 else (" twice" if n == 2 else "")
        reason = _WONDER_WHY.get(w.get("kind", ""), "there was something I didn't know").format(
            subject=w.get("subject", "it"), times=times)
        when = (" You'd paused, so it seemed a good moment to ask." if why.get("free")
                else " It seemed important enough to ask at a pause.")
        return f"I asked because {reason}.{when}"
    if kind == "moment":
        r = _MOMENT_WHY.get(str(why.get("moment", "")), "you'd paused and it seemed a good moment")
        return f"I said that because {r}."
    if kind == "check_in":
        m = why.get("minutes")
        return (f"You'd been working for about {int(m)} minutes without a break, so I said something."
                if m else "You'd been working for a while without a break, so I said something.")
    if kind == "briefing":
        b = why.get("briefing", "")
        return ("It's the first time you've been at the computer this morning, so I gave you the morning "
                "briefing." if b == "morning" else
                "It's the evening and you'd paused, so I gave you a summary of the day." if b == "evening"
                else "You asked me for a briefing.")
    if kind == "watch":
        return f"You asked me to tell you when {why.get('what', 'that happened')}."
    if kind == "notes":
        return "You asked me to remember that, or asked about something you'd told me to remember."
    if kind == "asked":
        return f"You asked me to: \"{why.get('query', '')}\"."
    if kind == "answered":
        return "You'd just answered my question, so I said what I'd taken from it."
    if kind == "correction":
        return "You told me my last answer was for the wrong thing, so I asked what you meant."
    if kind == "observation":
        reasons = [r for r in (why.get("reasons") or []) if r]
        s = f"I noticed {why.get('noticed', 'something')} and it seemed worth mentioning"
        return s + (f" ({'; '.join(reasons[:3])})." if reasons else ".")
    return "I'm not sure I can explain that one."


def route_label(meta: Dict[str, Any], text: str = "", tool: str = "") -> Dict[str, str]:
    """How a line reached the screen, for the console's "why" card: the
    route (rules / tool / model / noticed / wikipedia) and one plain sentence.

    From what the reply's own record already says -- no new signal."""
    pathway = str(meta.get("pathway") or "")
    outcome = str(meta.get("outcome") or "")
    att = meta.get("attention") or {}
    kind = str(att.get("kind") or "") if isinstance(att, dict) else ""
    t = (text or "").rstrip()
    tool = str(tool or "").replace("_", " ")
    if meta.get("degraded"):
        return {"route": "notice", "detail": "The language model wasn't available, so this is a fixed line."}
    if t.endswith("(Wikipedia)"):
        return {"route": "wikipedia", "detail": "The first lines of Wikipedia's page, fetched when you asked. Only the topic was sent."}
    if not meta.get("addressed"):
        return {"route": "noticed", "detail": "ASH said this on its own, from what it saw you doing -- you didn't ask."}
    if pathway == "slow" and tool:
        return {"route": "tool", "detail": f"The {tool} tool ran, and the language model put it into words."}
    if pathway == "slow":
        return {"route": "model", "detail": "The language model answered from what it was trained on. It didn't look this up, so check it if it matters."}
    if pathway in ("fast", "reflex"):
        return {"route": "reflex", "detail": f"A quick habit{' using the ' + tool + ' tool' if tool else ''}, no language model."}
    if outcome or pathway in ("offer", "rules", ""):
        return {"route": "rules", "detail": "ASH's own rules acted on what you said -- the language model wasn't involved."}
    return {"route": "rules", "detail": "ASH's own rules."}
