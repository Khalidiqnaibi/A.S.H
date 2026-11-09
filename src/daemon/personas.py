"""
src/daemon/personas.py

"Be a pirate for ten minutes."

A persona changes how ASH talks, not what it knows or can do. Said to ASH:

    be a pirate / talk like a robot / act like a wizard for 15 minutes
    pretend you're a detective / pretend to be a butler
    be yourself / drop the act / stop pretending / normal voice

While one is on, the model is told to word its answers in that voice (short,
still correct, never claiming a power ASH doesn't have), and every rule-made
answer -- timers, facts, commands -- is left as it is, since those are not the
model talking. It ends on its own after the time (ten minutes unless said,
at most an hour), and "be yourself" ends it at once. The face grins when it
starts. Built-in voices below; any "a/an/the <word or two>" is allowed too.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

DEFAULT_MIN = 10.0
MAX_MIN = 60.0

STYLES: Dict[str, Tuple[str, str]] = {
    # name: (how to talk, the line said when it starts)
    "pirate": ("a pirate: 'arr', 'matey', nautical words", "Arr, aye aye! Ask away, matey."),
    "robot": ("a stiff, literal robot: short beeping sentences, 'AFFIRMATIVE', 'COMPUTING'", "BEEP. ROBOT MODE ENGAGED."),
    "wizard": ("an old wizard: grand, mysterious, a little dramatic", "Behold! The wizard is in."),
    "detective": ("a noir detective: dry, suspicious, narrating the case", "The name's ASH. I'm on the case."),
    "butler": ("a polished British butler: formal, courteous, 'sir' or 'madam'", "At your service. How may I be of use?"),
    "cowboy": ("a cowboy: laid-back western drawl, 'partner', 'reckon'", "Howdy, partner. Reckon I'm ready."),
    "grandma": ("a warm grandmother: caring, 'dear', gently fussing", "Oh, hello dear. What can I do for you?"),
    "kid": ("a cheerful small child: simple words, excited, curious", "Yay, okay! What are we doing?"),
    "bmo": ("a sweet, silly little game console: playful, enthusiastic, refers to itself in the third person", "BMO is ready! Let's play!"),
}
ALIASES = {"pirate captain": "pirate", "buccaneer": "pirate", "droid": "robot", "jarvis": "butler", "child": "kid",
           "little kid": "kid", "gamer": "bmo", "cop": "detective", "sherlock": "detective"}

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?"
_FOR = (r"(?:\s+for\s+(?:(?:an?|one)\s+(?P<u1>hour|minute)|(?P<n>\d{1,2})\s*(?P<u>hours?|hrs?|h|minutes?|mins?|m)))?")
_Z = r"\s*[.!]*\s*$"
_START = re.compile(_A + r"(?:(?:be|act\s+(?:like|as)|talk\s+like|speak\s+like|sound\s+like|pretend\s+(?:to\s+be|you'?re|you\s+are))"
                    r"\s+(?:(?:an?|the)\s+)?(?P<w>[a-z][a-z ]{1,24}?))" + _FOR + _Z, re.I)
_END = re.compile(_A + r"(?:be\s+yourself|drop\s+the\s+act|stop\s+(?:pretending|acting|the\s+act)|normal\s+voice|"
                  r"talk\s+normally|speak\s+normally|back\s+to\s+normal|end\s+(?:the\s+)?(?:act|persona))" + _Z, re.I)
# Words that follow "be" but are never a character.
_NOT = frozenset("""quiet silent still careful honest brief short shorter longer louder quieter nice kind good bad fast slow
    patient serious sure ready here there back right wrong yourself myself happy sad mad angry sleepy tired
    more less so very too bit little lot few better worse real great my your our their his her its""".split())


def parse(text: str, now: float):
    """("start", name, style, opener, until) | ("end",) | None."""
    t = (text or "").strip()
    if not t or len(t) > 80:
        return None
    if _END.match(t):
        return ("end",)
    m = _START.match(t)
    if not m:
        return None
    word = re.sub(r"\s+", " ", m.group("w").strip().lower())
    if word.split()[0] in _NOT or word in _NOT:
        return None
    name = ALIASES.get(word, word)
    if name in STYLES:
        style, opener = STYLES[name]
    else:
        if len(word.split()) > 2 or not re.search(r"(?:an?|the)\s", t, re.I):
            return None                       # a made-up character needs "a/an/the"
        style, opener = f"{name}, in a way people expect of one", f"Okay -- {name} mode on."
    if m.group("u1"):
        mins = 60.0 if m.group("u1").lower() == "hour" else 1.0
    elif m.group("n"):
        mins = float(m.group("n")) * (60.0 if m.group("u").lower().startswith("h") else 1.0)
    else:
        mins = DEFAULT_MIN
    mins = max(1.0, min(MAX_MIN, mins))
    return ("start", name, style, opener, now + mins * 60.0)


def context_line(name: str, style: str) -> str:
    return (f"[PRETEND PLAY: the user asked you to talk like {style}. Word your answer in that voice, in one "
            f"or two short sentences, but keep it true: do not claim abilities you lack, do not invent facts, "
            f"and still answer what was asked.]")
