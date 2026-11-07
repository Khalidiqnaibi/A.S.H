"""
src/daemon/honest.py

Keeping ASH's answers true about itself.

In one session the model, left to itself:

  * listed "manage your calendar, send and receive emails, make phone calls"
    when asked what it can do (it can do none of the last two)
  * said "Alright, I can use a web browser to search for pictures of cats",
    printed "[web_browser open new tab]" as if it had run a tool, and told
    the user to "scroll through the results"
  * ended an answer with "Facts: { "intent": "clarification", ... }" -- its own
    briefing, read aloud
  * produced "Phone: +212 XXX-XXXXXX, Email: khalid.iqnaibi@email.com" for
    someone, from nothing

So, by rule and before the model gets a chance:

    "what can you do?"            the list of what is really switched on here
    "show me cats online",        what ASH cannot do (browse the web, send
    "send an email to Sara",      email or messages, make calls, buy things),
    "call Omar", "order a pizza"  said plainly, with what it *can* do instead

and after the model has answered, `sanitize` takes out the things it should
never have said: a pasted "Facts:" block, fake tool calls in square brackets,
and any e-mail address, phone number or link that is nowhere in what ASH was
actually given.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_A = r"^\s*(?:(?:hey|hi|ok(?:ay)?)\s+)?(?:ash[,\s]+)?(?:(?:please|can\s+you|could\s+you)[,\s]+)*"
_Z = r"(?:[,\s]+(?:ash|please|now|for\s+me|right\s+now|for\s+now|today))*\s*[.!?]*\s*$"

_CAPS = re.compile(_A + r"(?:what\s+(?:can|could|do)\s+you\s+(?:do|help\s+(?:me\s+)?with|know\s+how\s+to\s+do)"
                   r"|what\s+are\s+(?:all\s+)?(?:your|the)\s+(?:capabilities|abilities|features|skills|commands)"
                   r"|what\s+can\s+i\s+(?:ask|say\s+to|tell)\s+you|what\s+(?:are\s+)?you\s+(?:good|able)\s+(?:at|to\s+do)"
                   r"|help(?:\s+me)?|list\s+(?:your\s+)?(?:commands|features))" + _Z, re.I)

_WEB = re.compile(_A + r"(?:(?:show|find|get|fetch|search|look\s+up|look\s+for|pull\s+up|bring\s+up)\b[^.?!]{0,60}"
                  r"\b(?:online|on\s+the\s+(?:web|internet|net)|on\s+google|on\s+youtube|in\s+google|from\s+the\s+(?:web|internet))\b"
                  r"|(?:search|browse|surf|google)\s+(?:the\s+)?(?:web|internet|net)\b"
                  r"|google\s+(?!it\b)\S+|(?:open|go\s+to)\s+(?:a\s+)?(?:new\s+)?(?:browser\s+)?tab"
                  r"|(?:show|find)\s+me\s+(?:some\s+)?(?:pictures|photos|images|videos|memes)\s+of\b)", re.I)
_MSG = re.compile(_A + r"(?:send\s+(?:an?\s+|the\s+|my\s+)?(?:e-?mail|mail|message|text|sms|whatsapp|dm)\b"
                  r"|(?:e-?mail|text|message|whatsapp|dm)\s+(?!me\b)[A-Z]?[a-z]+\s+(?:that|about|to|saying|and)\b"
                  r"|reply\s+to\s+(?:the\s+|that\s+|his\s+|her\s+)?(?:e-?mail|message|text))", re.I)
_CALL = re.compile(_A + r"(?:make\s+(?:a\s+)?(?:phone\s+)?call\b|(?:phone|dial|ring)\s+\S+"
                   r"|call\s+(?:my\s+)?(?:mom|mum|dad|wife|husband|brother|sister|boss|friend|home|the\s+\w+|\+?\d[\d\s-]{5,}))", re.I)
_BUY = re.compile(_A + r"(?:(?:order|buy|purchase|book|reserve)\s+(?:me\s+)?(?:an?\s+|some\s+|the\s+)?\S+"
                  r"(?:\s+\S+){0,5}?\s+(?:online|from\s+\w+|on\s+(?:amazon|ebay|talabat|uber)|for\s+me|delivered)\b"
                  r"|order\s+(?:me\s+)?(?:a\s+|some\s+)?(?:pizza|food|dinner|lunch|coffee|taxi|uber|ride)\b)", re.I)

_LIMITS = (
    (_WEB, "I can't browse the web or show you things from it -- I don't have a browser tool. I can open Chrome for "
           "you if you say “open Chrome”, and I can look up a term in your own files."),
    (_MSG, "I can only read your email, not send it, and I can't send messages or texts to anyone."),
    (_CALL, "I can't make phone calls."),
    (_BUY, "I can't buy things or place orders."),
)


def limit_reply(text: str) -> Optional[str]:
    """What ASH can't do, if `text` asks for it; else None."""
    t = (text or "").strip()
    if not t or len(t) > 140:
        return None
    for rx, say in _LIMITS:
        if rx.match(t):
            return say
    return None


def asks_capabilities(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= 80 and bool(_CAPS.match(t))


def capabilities(rt) -> str:
    """What is really on, here, said in a sentence or two."""
    can: List[str] = []

    def on(name: str) -> Any:
        return getattr(rt, name, None)

    if on("reminders") is not None:
        can.append("set reminders, timers and alarms")
    if on("watcher") is not None:
        can.append("watch a download, the battery or a program and tell you when it's done")
    if on("desktop") is not None:
        can.append("open, switch to and close your apps")
    if on("media") is not None:
        can.append("control the music")
    h = on("smarthome")
    if h is not None and getattr(getattr(h, "cfg", None), "url", ""):
        can.append("turn your lights and plugs on and off")
    w = on("weather")
    if w is not None and getattr(w, "configured", False):
        can.append("tell you the weather")
    c = on("calendar")
    if c is not None and (getattr(getattr(c, "cfg", None), "files", ()) or getattr(getattr(c, "cfg", None), "urls", ())):
        can.append("tell you what's on your calendar")
    i = on("inbox")
    if i is not None and getattr(i, "configured", False):
        can.append("tell you who emailed you")
    if on("tasks") is not None:
        can.append("do several things in a row once you approve a plan")
    if on("play") is not None:
        can.append("play games and tell stories")
    if on("research") is not None:
        can.append("look things up in the background")
    can.append("calculate, convert units and read files")
    controls = ("turn the microphone, camera or screen capture on and off, go private, go quiet, change my face, "
                "and tell you how the computer is doing")
    head = "I can " + ", ".join(can[:-1]) + (" and " if len(can) > 1 else "") + can[-1] + ". "
    return (head + "I can also " + controls + ". And I can talk, with my language model -- which I'd check before "
            "relying on it. I can't browse the web, send email or messages, or make calls.")


# -------------------------------------------------------------------- after the model has answered
_FACTS = re.compile(r"\n+\s*(?:Facts|Briefing|Context|Gist)\s*:\s*[{\[].*\Z", re.S)
_TRAIL_JSON = re.compile(r"\n+\s*\{\s*\"(?:intent|tool_used|action_class|blocked|facts)\".*\Z", re.S)
# A tool name has an underscore ("web_browser", "search_engine"); ordinary
# bracketed words ("[the docs]") are left alone.
_FAKE_CALL = re.compile(r"^\s*\[[a-z][a-z0-9]*_[a-z0-9_]+(?:[ :][^\]\n]{0,120})?\]\s*$", re.I)
_INLINE_CALL = re.compile(r"\[[a-z][a-z0-9]*_[a-z0-9_]+(?:[ :][^\]\n]{0,120})?\]", re.I)
_SPAN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"                      # an e-mail address
                   r"|\+\d[\d\s().-]{6,}\d|\+\d+[\s\d-]*X{3,}[\s\dX-]*"      # +country number, or a placeholder
                   r"|(?<![\w.])\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w])"   # 555-123-4567, 079 123 4567
                   r"|https?://\S+", re.I)                                  # a link
_SENT = re.compile(r"(?<=[.!?])\s+")


def _squash(s: str) -> str:
    return re.sub(r"[\s().-]+", "", (s or "").lower())


def sanitize(reply: str, known: Any = "") -> str:
    """The reply without what the model should never have said. `known` is
    everything ASH was actually given (the user's words, what memory
    returned), or a function that returns it -- only asked if it's needed."""
    t = reply or ""
    t = _FACTS.sub("", t)
    t = _TRAIL_JSON.sub("", t)
    t = "\n".join(line for line in t.split("\n") if not _FAKE_CALL.match(line))
    t = _INLINE_CALL.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return ""
    known_flat: Optional[str] = None
    out = []
    dropped = False
    for line in t.split("\n"):
        keep = []
        for sent in _SENT.split(line):
            bad = False
            for m in _SPAN.finditer(sent):
                span = m.group(0).strip(" .,;:)")
                if re.search(r"x{3,}", span, re.I):
                    bad = True                      # a placeholder is never a real number
                    continue
                if known_flat is None:
                    known_flat = _squash(known() if callable(known) else known)
                if _squash(span) not in known_flat:
                    bad = True                      # an address, number or link from nowhere
            if bad:
                dropped = True
            else:
                keep.append(sent)
        if keep:
            out.append(" ".join(keep))
    text = "\n".join(out).strip()
    if dropped and not text:
        return "I don't have that detail."
    return text
