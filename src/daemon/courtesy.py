"""
src/daemon/courtesy.py

"Thanks", "will do thanks", "later", "good night", "kk": answered at once, by rule, with a short line -- no language model.

Today these went to the model: "Later" took 27 seconds and came back as "You just switched back to Discord... I told you to stand up
after an hour and a half" (ASH had said nothing of the kind), and "will do thanks" took 23 seconds and came back as a telling-off.
A person who says goodbye does not want a thought; they want "later". It is the last rule in the chain, so anything that is waiting
for an answer (an offer, a question ASH asked, a game) has the line first: this only takes what nothing else wanted.

Deliberately narrow: only short lines made of nothing but courtesy words. "Thanks, now what's the weather" is not one.
Not "yes", "ok", "sure" or "alright": those answer a question, and which one is not for this module to guess.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

_PUNCT = re.compile(r"[^\w\s']")

_THANKS_WORDS = frozenset("thanks thank thx ty tysm cheers appreciated appreciate".split())
_THANKS_EXTRA = frozenset("""you a lot so much very many it man bro mate dude ash buddy ok okay alright cool great nice awesome
    perfect sweet yeah yep will do sure thing bet got that again for everything the help""".split())
_ACK_LINES = frozenset({"k", "kk", "kay", "okie", "okey", "got it", "gotcha", "noted", "makes sense", "i see", "cool", "nice", "aight",
                        "will do", "on it", "sure thing", "no worries", "np", "all good", "fair enough", "understood"})
# "No thanks." / "nope" / "never mind" / "I'm good" / "ok" with NOTHING asked (the last thing ASH said was not a question): it went to the model for 62 s.
# Only then -- after a question these answer it, and which way is not for this module to guess.
_DECLINE = frozenset({"no thanks", "no thank you", "no thanks ash", "nope", "nah", "no", "no no", "not now", "not right now", "not today", "never mind", "nevermind",
                      "nvm", "im good", "i'm good", "i am good", "no need", "its fine", "it's fine", "thats fine", "that's fine", "forget it", "its okay", "it's okay",
                      "no its fine", "no it's fine", "no worries", "all good", "never mind that", "dont worry about it", "don't worry about it", "skip it", "leave it"})
_BARE_YES = frozenset({"ok", "okay", "alright", "all right", "sure", "yeah", "yep", "yes", "fine", "right", "ok ok", "okay okay", "yup", "mhm", "uh huh"})
_BYE = re.compile(r"^(?:ok(?:ay)?\s+|alright\s+|well\s+|cool\s+|ash\s+)*(?:later|laters|see\s+(?:you|ya)(?:\s+(?:later|soon|tomorrow))?|"
                  r"cya|bye(?:\s+bye)?|goodbye|talk\s+(?:to\s+you\s+)?later|ttyl|gotta\s+go|i(?:'m|\s+am)\s+(?:off|out|heading\s+out|"
                  r"going)|heading\s+out|catch\s+you\s+later|peace|peace\s+out|take\s+care)(?:\s+(?:ash|man|bro|mate))*$")
_NIGHT = re.compile(r"^(?:ok(?:ay)?\s+)?(?:good\s*night|goodnight|gn|night(?:\s+night)?|nighty\s+night|sleep\s+well)(?:\s+(?:ash|man|bro))*$")
_BRB = re.compile(r"^(?:brb|be\s+right\s+back|back\s+in\s+a\s+(?:bit|sec|minute|moment)|one\s+sec(?:ond)?|give\s+me\s+a\s+(?:sec|minute|moment))$")

# A complaint about ASH itself, said short ("You are so dumb today.", "you're hallucinating, boy", "stop making things up"). It went to the
# model, which answered an insult with a paragraph. Owned in a line, and asked what was wrong: the daemon adds what it knows went wrong.
_COMPLAINT = re.compile(
    r"^(?:ok(?:ay)?\s+|wow\s+|man\s+|dude\s+|bro\s+|ash\s+|seriously\s+|honestly\s+)*(?:you(?:'re|\s+are|\s+r)|ur|u\s+r|this\s+is)\s+"
    r"(?:so\s+|really\s+|just\s+|being\s+|too\s+|such\s+a\s+|completely\s+|totally\s+|very\s+)*"
    r"(?:dumb|stupid|useless|hallucinating|lying|terrible|awful|slow|bad|annoying|broken|garbage|trash|incompetent|worthless|"
    r"an\s+idiot|dumber|useless\s+today|not\s+(?:working|helping|listening|making\s+sense)|making\s+(?:things|stuff)\s+up|"
    r"making\s+no\s+sense|so\s+wrong|wrong\s+again)(?:\s+(?:today|again|now|right\s+now|lately|ash|boy|man|bro|mate|dude))*$"
    r"|^(?:you\s+suck|stop\s+(?:lying|hallucinating|making\s+(?:things|stuff)\s+up|being\s+(?:dumb|stupid))|you\s+keep\s+(?:lying|"
    r"hallucinating|making\s+(?:things|stuff)\s+up|getting\s+it\s+wrong)|you\s+never\s+(?:listen|get\s+it\s+right)|"
    r"what\s+is\s+wrong\s+with\s+you|you\s+make\s+no\s+sense)(?:\s+(?:today|again|now|ash|boy|man|bro|mate|dude))*$")

_LINES: Dict[str, List[str]] = {
    "complaint": ["Fair. Tell me what I got wrong and I'll fix it.", "That's on me. What did I get wrong?", "Sorry. Tell me what was off and I'll try again.",
                  "You're right to be annoyed. What should I have done?"],
    "thanks": ["Anytime.", "You got it.", "No problem.", "Happy to."],
    "bye": ["Later.", "See you.", "Catch you later.", "Alright, later."],
    "night": ["Night. Sleep well.", "Good night.", "Night."],
    "brb": ["I'll be here.", "Take your time.", "Sure."],
    "ack": ["Okay.", "Got it.", "Mm-hm."],
    "decline": ["Okay.", "No problem.", "Alright.", "Sure."],
}


def _norm(text: str) -> str:
    t = _PUNCT.sub(" ", str(text or "").lower().replace("’", "'"))
    t = re.sub(r"^\s*(?:hey\s+)?ash[\s,]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


# What a Chooser learns these from (the rules above catch the phrasings somebody thought of; these catch what they mean, and the
# rules' own catches are added to them as they happen). "none" is what must stay a sentence for the rest of ASH -- the near misses first.
SEEDS: Dict[str, List[str]] = {
    "thanks": ["thanks", "thank you", "thanks a lot", "thank you so much", "cheers", "much appreciated", "appreciate it", "thx", "ty",
               "thanks man", "will do thanks", "ok thanks", "awesome thanks", "great thank you", "i appreciate that", "that helps thanks",
               "perfect thanks"],
    "bye": ["later", "see you", "see you later", "bye", "goodbye", "cya", "talk later", "talk to you later", "gotta go", "i'm heading out",
            "i'm off", "catch you later", "see you tomorrow", "peace out", "take care", "i have to go now", "bye for now", "i'm leaving now"],
    "night": ["good night", "goodnight", "gn", "night", "sleep well", "i'm going to bed", "heading to bed", "time for bed", "off to sleep",
              "nighty night"],
    "brb": ["brb", "be right back", "back in a minute", "one sec", "give me a second", "hold on a moment", "wait a sec", "just a minute",
            "gimme a minute", "hang on"],
    "ack": ["k", "kk", "okay cool", "got it", "gotcha", "noted", "makes sense", "i see", "understood", "fair enough", "no worries",
            "sounds good", "alright then", "cool", "nice", "makes sense thanks"],
    "complaint": ["you are so dumb today", "you're useless", "you are hallucinating", "you're making things up", "stop making things up",
                  "you're being stupid", "you suck", "you are not listening", "you make no sense", "what is wrong with you", "you keep getting it wrong",
                  "you never listen", "you're so slow today", "you're terrible", "stop hallucinating", "you're not helping", "you're lying",
                  "you are broken today", "you are so wrong again", "that was useless"],
    "none": ["what is the weather today", "play some music", "who is ada lovelace", "tell me a joke", "open the browser", "what time is it",
             "remind me to call mom at five", "what do you know about me", "yes", "no", "ok", "sure", "yeah", "maybe", "why is my computer slow",
             "can you help me with something", "i need to write an email", "search for pizza places", "how do i fix this error",
             "what are you doing", "i'm tired", "i'm bored", "how are you", "tell me about yourself", "what's your name",
             "do you remember my friend", "turn the volume up", "set a timer for ten minutes", "what is two plus two", "show me my notes",
             "i want to play a game", "what should i eat tonight", "that's wrong", "you are wrong about that", "stop", "wait what", "hello",
             "hi there", "good morning", "i love you", "that's funny", "lol", "hmm", "really", "why", "what happened",
             "remember that i parked on level three", "open spotify", "i think so", "not really", "i don't know",
             "good night's sleep tips", "tips for sleeping better", "what's the plan for tonight", "i want to go home",
             "bye the way what time is it", "thanks but that's not what i asked", "no thanks", "yes please", "later today remind me to call dad",
             "goodbye cruel world is a song right", "i need a minute to think about what to ask", "what is my cpu usage", "what do you remember",
             "who is lieth", "my name is khalid", "can you play something upbeat", "how are you feeling", "remember to buy milk"],
}


class Courtesy:
    def __init__(self, clock: Optional[Callable[[], float]] = None, chooser=None):
        self.clock = clock
        self.chooser = chooser                      # a Chooser (chooser.py): what the rules do not list, by meaning
        self._recent: Dict[str, Deque[str]] = {k: deque(maxlen=2) for k in _LINES}
        self.stats = {"answered": 0, "by_meaning": 0}
        self.last_kind = ""                         # what the last line was taken for ("thanks", "complaint", ...)

    def _by_rule(self, t: str, words: List[str]) -> Optional[str]:
        if _COMPLAINT.match(t):
            return "complaint"
        if _NIGHT.match(t):
            return "night"
        if _BRB.match(t):
            return "brb"
        if _BYE.match(t):
            return "bye"
        if any(w in _THANKS_WORDS for w in words) and all(w in _THANKS_WORDS or w in _THANKS_EXTRA for w in words):
            return "thanks"
        if t in _ACK_LINES:
            return "ack"
        return None

    def kind(self, text: str, open_q: bool = True) -> Optional[str]:
        """What a short line is. `open_q` (the default, the safe side): ASH may have just asked something, so a bare no / ok is its answer, not courtesy."""
        t = _norm(text)
        words = t.split()
        if not words or len(words) > 6 or len(t) > 40:
            return None
        if not open_q:
            if t in _DECLINE or t.replace("'", "") in _DECLINE:
                return "decline"
            if t in _BARE_YES:
                return "ack"
        k = self._by_rule(t, words)
        if k is not None:
            if self.chooser is not None:
                self.chooser.learn(text, k)         # what the rules caught teaches the chooser too
            return k
        if self.chooser is not None and len(words) <= 5 and "?" not in str(text):
            v = self.chooser.classify(text)
            if v.label in _LINES:
                self.stats["by_meaning"] += 1
                return v.label
        return None

    def hear(self, text: str, now: Optional[float] = None, open_q: bool = True) -> Optional[str]:
        k = self.kind(text, open_q)
        self.last_kind = k or ""
        if k is None:
            return None
        options = [x for x in _LINES[k] if x not in self._recent[k]] or _LINES[k]
        line = options[self.stats["answered"] % len(options)]
        if k == "night" and now is not None and 5 <= datetime.fromtimestamp(now).hour < 18:
            line = "Alright, later."                      # "night" at noon is a goodbye, not a bedtime
        self._recent[k].append(line)
        self.stats["answered"] += 1
        return line
