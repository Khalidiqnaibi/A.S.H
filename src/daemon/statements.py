"""
src/daemon/statements.py

"I'm just building my custom hardware chip." / "I've bought a 3D printer and I worked on it for a little bit.": the user telling ASH what they
are doing. Not a question, not a command -- nothing is being asked of it.

Both went to the language model, 60 seconds each, and came back as paragraphs of encouragement. What a person wants from a statement is to
have been heard, and for it to be remembered: a short acknowledgement, and the line kept as what they said (the model never kept it as
anything more than a turn of conversation). That is all this does, and only for the plain case: a first-person statement about doing, making,
buying or finishing something, with no question in it and nothing asked of ASH.

Deliberately narrow, and the daemon runs it last: feelings ("I'm tired" -- someone to answer, not to note), requests ("I need you to..."),
questions, commands and anything that is about ASH itself are left alone, to the rules that know them or to the model.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|well|oh|yeah|ok(?:ay)?|actually|btw|anyway|um|uh|just)[,\s]+)*"
_FIRST = re.compile(_LEAD + r"(?:i(?:'m|\s+am|'ve|\s+have|'ll|\s+will|\s+just|\s+was|\s+am\s+going\s+to|'m\s+going\s+to|\s+plan\s+to|\s+decided\s+to)?|"
                    r"we(?:'re|\s+are|'ve|\s+have|\s+just)?|my\s+\w+(?:\s+\w+)?\s+(?:is|was|just))\b", re.I)
# What makes it a statement about doing something, not chat: a doing word.
_DOING = re.compile(
    r"\b(?:build(?:ing)?|built|mak(?:e|ing)|made|work(?:ing|ed)\s+on|learn(?:ing|ed)|read(?:ing)?|writ(?:e|ing|ten)|wrote|design(?:ing|ed)|"
    r"test(?:ing|ed)|stud(?:y|ying|ied)|develop(?:ing|ed)|print(?:ing|ed)|set(?:ting)?\s+up|bought|ordered|got|finish(?:ed|ing)|start(?:ed|ing)|"
    r"fix(?:ed|ing)|install(?:ed|ing)|creat(?:ed|ing)|cod(?:e|ed|ing)|try(?:ing)?|tried|decided|going\s+to|plan(?:ning)?|will\s+(?:hook|connect|try|build|"
    r"make|get|buy|use)|'ll\s+(?:hook|connect|try|build|make|get|buy|use))\b", re.I)
# Left to someone else: how the person feels (answered, not noted), a request, a thing asked of ASH.
_FEELING = re.compile(r"\b(?:tired|bored|sad|angry|mad|happy|stressed|sick|hungry|sleepy|anxious|nervous|scared|worried|feeling|depressed|lonely|"
                      r"upset|excited|exhausted|annoyed|frustrated|overwhelmed|ill|hurt|cry(?:ing)?|dying|hate|love|miss)\b", re.I)
_ASKING = re.compile(r"\?|\b(?:i\s+(?:need|want|wish|would\s+like|'d\s+like)\s+(?:you|to\s+know)|can\s+you|could\s+you|would\s+you|please|"
                     r"remind\s+me|remember\s+(?:that|this)|tell\s+me|show\s+me|help\s+me|let\s+me\s+know|do\s+you|are\s+you|what|why|how|where|when|who)\b", re.I)
_YOU = re.compile(r"\byou(?:r|rs|rself)?\b", re.I)
_PROMISE = re.compile(r"\bi(?:'ll|\s+will|'m\s+going\s+to|\s+might|\s+plan\s+to)\b", re.I)

_OBJ = re.compile(
    r"\b(?:building|making|designing|working\s+on|learning|reading|writing|studying|developing|printing|setting\s+up|testing|fixing|installing|"
    r"creating|bought|ordered|built|made|started|finished|got)\s+(?:a|an|the|my|our|some|this|that|new)?\s*([a-z0-9][^.,!;]*?)"
    r"(?:\s+(?:and|but|with|for|right|today|now|so|which|that|because|since|using|in|on|at|it|then|yet|again|this|tonight|yesterday|earlier|last|all|ash)\b|[.,!;]|$)", re.I)

_LINES: List[str] = ["Nice.", "Got it.", "Good to know.", "Sounds good.", "Noted."]
_WITH_OBJ: List[str] = ["Nice -- {o}.", "Got it -- {o}.", "Good to know -- {o}.", "Sounds good, {o}."]


class Statements:
    def __init__(self, chooser=None, clock: Optional[Callable[[], float]] = None):
        self.chooser = chooser            # a Chooser (chooser.py): "a statement about what the user is doing" by meaning, as well as by rule
        self.clock = clock
        self.stats = {"heard": 0, "by_meaning": 0}
        self._n = 0

    def is_statement(self, text: str) -> bool:
        t = (text or "").strip()
        words = t.split()
        if not 4 <= len(words) <= 32 or len(t) > 220:
            return False
        if _ASKING.search(t) or _FEELING.search(t):
            return False
        if _YOU.search(t) and not _PROMISE.search(t):
            return False                    # about ASH, or said to it: not for this
        by_rule = bool(_FIRST.match(t)) and bool(_DOING.search(t))
        if by_rule:
            if self.chooser is not None:
                self.chooser.learn(t, "statement")
            return True
        if self.chooser is not None and 4 <= len(words) <= 24:
            if self.chooser.classify(t).label == "statement":
                self.stats["by_meaning"] += 1
                return True
        return False

    def hear(self, text: str) -> Optional[str]:
        """The acknowledgement, or None when this is not a plain statement."""
        if not self.is_statement(text):
            return None
        self.stats["heard"] += 1
        self._n += 1
        m = _OBJ.search(text)
        obj = ""
        if m:
            obj = " ".join(m.group(1).split()[:6]).strip(" -")
            obj = re.sub(r"\s+(?:ash|tonight|today|now)$", "", obj, flags=re.I)
        if obj and len(obj) >= 3:
            return _WITH_OBJ[self._n % len(_WITH_OBJ)].format(o=obj)
        return _LINES[self._n % len(_LINES)]


# What a Chooser learns "is this the user telling ASH what they are doing?" from.
SEEDS = {
    "statement": ["i'm building a custom chip", "i bought a 3d printer", "i've been working on my thesis all week", "i'm learning rust at the moment",
                  "i finished the report this morning", "we're setting up a new server", "i started a new project today",
                  "i made pasta for dinner", "i'm writing a paper about vision models", "i got a new keyboard yesterday",
                  "i decided to switch to linux", "i'm going to try a different approach", "i fixed the bug in the parser",
                  "i installed the new driver and it works", "i ordered a new monitor"],
    "none": ["what time is it", "can you help me with this", "i need you to open chrome", "i want to know what the weather is", "i'm tired",
             "i'm so bored today", "i love you ash", "i hate this", "what are you doing", "why is the computer slow", "play some music",
             "remind me to call mom", "i think you are wrong", "i don't understand what you said", "tell me a joke", "who is ada lovelace",
             "you are doing great", "thanks for that", "yes", "no", "i'm not sure what you mean", "that's not what i meant", "i feel sick",
             "i'm nervous about the interview", "do you remember my name", "i want to play a game", "how are you", "open the browser"],
}
