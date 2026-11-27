"""
src/daemon/dial.py

Two small things people ask of something that lives with them, by rule:

    the dial   TARS's humour setting, kept to what ASH may change: how OFTEN it volunteers. "Be less chatty" / "be chattier" / "back to normal" set the
               hourly budget of remarks (quiet 2, normal 6, chatty 9); "fewer jokes" / "no more jokes" / "more jokes" set how often a joke comes. The
               personality itself -- direct, dry, a bit of BMO -- never changes; answers are never shorter or longer because of it.
    care       Baymax's "on a scale of one to ten": "I have a headache", "my eyes hurt", "I'm exhausted", "I feel sick" -> how long they have been at
               the screen, water, a break, and an offer to keep quiet. Never a diagnosis; words that sound like an emergency get "call your local
               emergency number".
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|ok(?:ay)?))*\s*[?.!]*\s*$"

LEVELS = {"quiet": (2, 1), "normal": (6, 3), "chatty": (9, 5)}            # (remarks an hour, offers an hour)
JOKES = {"none": None, "fewer": 12 * 3600.0, "normal": 6 * 3600.0, "more": 3 * 3600.0}

_CHAT = [("quiet", re.compile(_A + r"(?:be\s+)?(?:less\s+chatty|quieter|not\s+so\s+chatty)|(?:don'?t|stop)\s+(?:bring(?:ing)?\s+(?:things|stuff)\s+up|"
                              r"chatting)\s+so\s+(?:much|often)|(?:make\s+)?fewer\s+remarks|volunteer\s+less|speak\s+up\s+less|chattiness\s+(?:to\s+)?(?:low|quiet)" + _Z, re.I)),
         ("chatty", re.compile(_A + r"(?:be\s+)?(?:more\s+chatty|chattier|more\s+talkative)|chat\s+(?:with\s+me\s+)?more|bring\s+(?:things|stuff)\s+up\s+more|"
                               r"chattiness\s+(?:to\s+)?(?:high|up)" + _Z, re.I)),
         ("normal", re.compile(_A + r"(?:back\s+to\s+)?normal\s+(?:chattiness|chatter|amount\s+of\s+talking)|chattiness\s+(?:to\s+)?normal|"
                               r"talk\s+(?:as\s+much\s+)?as\s+usual" + _Z, re.I))]
_JOKE = [("none", re.compile(_A + r"(?:no\s+more\s+jokes|stop\s+(?:telling\s+|making\s+)?jokes|no\s+jokes(?:\s+please)?|enough\s+(?:with\s+the\s+)?jokes)" + _Z, re.I)),
         ("fewer", re.compile(_A + r"(?:fewer|less)\s+jokes|not\s+so\s+many\s+jokes|(?:tone\s+down|cut\s+down\s+on)\s+the\s+jokes" + _Z, re.I)),
         ("more", re.compile(_A + r"more\s+jokes|(?:tell|make)\s+(?:me\s+)?more\s+jokes|joke\s+more" + _Z, re.I)),
         ("normal", re.compile(_A + r"(?:you\s+can\s+(?:tell\s+)?jokes?\s+again|jokes\s+are\s+(?:fine|ok(?:ay)?)\s+again|normal\s+jokes)" + _Z, re.I))]

SAY_CHAT = {"quiet": "Okay -- I'll bring things up less: two remarks an hour at most. Answers stay the same.",
            "chatty": "Okay -- I'll bring up a bit more, up to nine remarks an hour. Still not while you're busy.",
            "normal": "Back to normal: up to six remarks an hour."}
SAY_JOKE = {"none": "Okay, no more jokes unless you ask for one.", "fewer": "Fewer jokes, then -- one every twelve hours at most.",
            "more": "More jokes it is -- one every three hours or so, when things are going fine.", "normal": "Jokes back to normal."}


def parse(text: str) -> Optional[Tuple[str, str]]:
    """('chat', level) or ('jokes', level) or None."""
    t = (text or "").strip()
    for level, rx in _CHAT:
        if rx.match(t):
            return "chat", level
    for level, rx in _JOKE:
        if rx.match(t):
            return "jokes", level
    return None


def apply(rt, kind: str, level: str) -> None:
    """Set it on the running parts: the gate's hourly caps, the free-moment joke's cooldown (or off)."""
    if kind == "chat":
        per_hour, offers = LEVELS[level]
        rt.gate.max_unsolicited_per_hour = per_hour
        rt.gate.max_offers_per_hour = offers
        return
    m = getattr(rt, "moments", None)
    joke = (getattr(m, "activities", None) or {}).get("joke") if m is not None else None
    if joke is None:
        return
    if not hasattr(joke, "_weight0"):
        joke._weight0 = joke.weight
    cd = JOKES[level]
    if cd is None:
        joke.weight = 0.0
    else:
        joke.weight = joke._weight0
        joke.cooldown_s = cd


# ------------------------------------------------------------------------------------------------------------------------------------ care
_HURT = re.compile(_A + r"(?:i\s+(?:have|'ve\s+got|got)\s+(?:a\s+|such\s+a\s+|this\s+)?(?P<head>(?:bad\s+|splitting\s+|terrible\s+)?headache|migraine)|"
                   r"my\s+(?P<eyes>eyes)\s+(?:hurt|are\s+(?:sore|tired|burning|dry))|my\s+(?P<back>back|neck|wrists?|shoulders?)\s+(?:hurts?|is\s+(?:killing\s+me|sore|stiff)|aches?)|"
                   r"i'?m\s+(?:so\s+|really\s+|super\s+)?(?P<tired>exhausted|wiped\s+out|drained|burnt\s+out|burned\s+out|dead\s+tired|knackered)|"
                   r"i\s+(?:feel|'m\s+feeling|am\s+feeling)\s+(?:a\s+bit\s+|really\s+|so\s+)?(?P<sick>sick|ill|unwell|nauseous|dizzy|rough))" + _Z, re.I)
_EMERGENCY = re.compile(r"\b(?:chest\s+pain|can'?t\s+breathe|cannot\s+breathe|heart\s+attack|stroke|passing\s+out|fainted|overdos|suicid|kill\s+myself|"
                        r"bleeding\s+(?:a\s+lot|badly|heavily))\b", re.I)


def care(text: str, at_it_s: float, hour: int) -> Optional[Tuple[str, int]]:
    """(what to say, minutes of quiet to offer -- 0 for none) for a line about how the body feels, or None."""
    t = (text or "").strip()
    if _EMERGENCY.search(t):
        return ("That sounds serious -- please call your local emergency number now, or get someone near you. I'm not able to help with that "
                "myself.", 0)
    m = _HURT.match(t)
    if not m:
        return None
    mins = int(at_it_s // 60)
    stretch = (f"You've been at the screen {mins // 60} hour{'s' if mins >= 120 else ''}" + (f" {mins % 60} minutes" if mins % 60 else "")
               if mins >= 60 else (f"You've been at it {mins} minutes" if mins >= 15 else ""))
    stretch = (stretch + " without a real break. ") if stretch else ""
    if m.group("head"):
        return (stretch + "Water and ten minutes away from the screen usually help. Want me to keep quiet for fifteen minutes?", 15)
    if m.group("eyes"):
        return (stretch + "Try twenty-twenty-twenty: every twenty minutes, look at something twenty feet away for twenty seconds. "
                "Want me to keep quiet for ten minutes while you rest them?", 10)
    if m.group("back"):
        return (stretch + "Stand up, roll your shoulders, walk for a couple of minutes. Want me to keep quiet for ten?", 10)
    if m.group("tired"):
        late = hour >= 23 or hour < 5
        return ((stretch + ("It's late -- sleep is the fix, and the work will keep. " if late else "A proper break would do more than pushing through. ")
                 + "Want me to keep quiet for half an hour?"), 30)
    return ("Rest beats pushing through, and if it gets worse, see a doctor. Want me to keep quiet for an hour?", 60)
