"""
src/senses/address.py

"Was that meant for me?"

The old answer was a wake word: a transcript counted as addressed only if it
contained the name, or opened with one of four polite forms. That is cheap and
it is wrong in the way that matters most -- nobody says the name twice in a
row, and almost nobody says it at all. "what's your status", "how are you",
"are you there", "what time is it", and every follow-up in an ongoing
conversation all scored zero and fell through to the volunteering path, where
a 0.55-salience event has to clear a 0.62 bar it cannot reach. The result:
heard, remembered, never answered. Being spoken to and ignored is the single
worst failure an assistant has, and the wake word was causing it constantly.

So this is evidence, not a keyword. Each signal below contributes to a
confidence in 0..1:

    name in vocative position     decisive on its own
    second person ("your", "you") the user is talking TO someone
    a question                    questions want answers
    an imperative verb            "open", "remind me", "turn it down"
    a polite request opener       "can you", "please"
    the conversation is open      ASH spoke seconds ago; this is the reply

and some of it subtracts:

    another addressee named          "hey Sam," is not for ASH
    the name used as a noun          "ash tray", "volcanic ash", "ashes"
    the name in a referential frame  "I told ash about it" (about, not to)

Two thresholds, not one
-----------------------
ADDRESSED (>= 0.60) behaves exactly as the wake word used to: an
unconditional override, answered regardless of drives, budget or refractory
period.

LIKELY (>= 0.32) is the band the wake word had no way to express -- probably
for ASH, not certainly. These do not override the attention gate; they enter
it with the politeness timer and the unsolicited budget waived and a score
floor set by the confidence, so a bare "what time is it" gets answered in a
quiet room and stays quiet in the middle of a busy one. See
src/daemon/attention.py, route 1b.

State
-----
`conversation` is a module-level singleton because the two existing callers
(MicrophoneSensor.is_addressed and the daemon's `simulate mic`) are both
static, and because there is only ever one conversation: the turn-taking
clock belongs to the room, not to a sensor instance. The daemon pokes it from
src/daemon/ambient.py when ASH speaks or the user takes a turn.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["AddressVerdict", "Conversation", "conversation", "score", "is_addressed",
           "ADDRESSED_AT", "LIKELY_AT"]

ADDRESSED_AT = 0.60
LIKELY_AT = 0.32

# How long after ASH speaks the next utterance is read as a reply to it.
FOLLOW_UP_S = 45.0

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# The name, including the initialism the user actually writes it as.
_NAME = r"(?:ash|a\.\s?s\.\s?h\.?)"

# Vocative: the name opening the sentence, closing it, or set off by commas.
# This is the "hey ash", "ash, stop", "what time is it, ash" shape.
_VOCATIVE = re.compile(
    r"^\s*(?:hey|hi|hello|ok|okay|yo|um|uh)?[\s,]*" + _NAME + r"\b[\s,.!?:;-]"
    r"|[\s,]" + _NAME + r"\s*[?.!]*\s*$",
    re.I,
)
_NAME_ANYWHERE = re.compile(r"\b" + _NAME + r"\b", re.I)

# The name as an ordinary noun. Vetoes the name signal entirely.
_NAME_AS_NOUN = re.compile(
    r"\bash(?:tray|trays|es|en|y)\b"
    r"|\bash\s+(?:tray|trays|wednesday|cloud|clouds|fall|heap|pile|blond|blonde|tree|wood)\b"
    r"|\b(?:volcanic|cigarette|cigar|wood|coal|fire|burnt|burned|mountain|the)\s+ash\b",
    re.I,
)
# The name as a thing talked ABOUT rather than talked TO.
_NAME_REFERENTIAL = re.compile(
    r"\b(?:about|with|told|telling|tells|and|than|like)\s+" + _NAME + r"\b"
    r"|\b" + _NAME + r"\s+(?:is|was|isn't|wasn't|does|did|said|says|thinks|keeps|"
    r"never|always)\b",
    re.I,
)

# Someone else is being spoken to by name.
_OTHER_ADDRESSEE = re.compile(
    r"^\s*(?:hey|hi|hello|yo|yeah)\s+(?!" + _NAME + r"\b)([a-z]+)\b\s*[,!]", re.I
)

_SECOND_PERSON = re.compile(
    r"\b(?:you|your|yours|yourself|you're|youre|you've|youve|you'll|youll|u)\b", re.I
)
_QUESTION = re.compile(
    r"\?\s*$"
    r"|^\s*(?:so|and|but|ok|okay|hey)?[\s,]*"
    r"(?:what|what's|whats|who|who's|whose|when|when's|where|where's|why|how|how's|hows|"
    r"which|is|are|am|was|were|do|does|did|can|could|would|will|won't|should|shall|have|"
    r"has|had|may|might|any|anything)\b",
    re.I,
)
_COMMAND = re.compile(
    r"^\s*(?:please\s+|just\s+|go\s+ahead\s+and\s+)*"
    r"(?:open|close|shut|start|stop|pause|resume|play|skip|mute|unmute|turn|set|setup|"
    r"remind|tell|show|give|find|search|look|read|write|send|call|text|email|add|remove|"
    r"delete|create|make|run|check|list|note|save|load|translate|calculate|convert|"
    r"summarize|summarise|repeat|say|speak|wake|sleep|lock|take|put|schedule|cancel|"
    r"snooze|remember|forget|explain|describe|compare|draw|fix|help)\b",
    re.I,
)
_POLITE_OPENER = re.compile(
    r"^\s*(?:hey\s+)?(?:can|could|would|will|do)\s+(?:you|u)\b"
    r"|^\s*(?:please|pls)\b"
    r"|^\s*i\s+(?:need|want|would\s+like)\s+(?:you\s+to|to\s+know)\b",
    re.I,
)

# Short turns that only make sense as a reply to something just said. Written
# as one-or-more tokens so "no thanks" and "ok cool" land here too.
_REPLY_TOKEN = (
    r"(?:yes|yeah|yep|yup|no|nope|nah|ok|okay|sure|right|exactly|correct|wrong|"
    r"thanks|thank\s+you|cheers|nice|cool|great|perfect|good|bad|stop|wait|hold\s+on|"
    r"go\s+on|carry\s+on|continue|again|louder|slower|never\s*mind|nevermind|forget\s+it|"
    r"please|then|the\s+(?:first|second|third|last)\s+one|that\s+one|both|neither|either)"
)
_REPLY_TOKENS = re.compile(
    r"^\s*" + _REPLY_TOKEN + r"\b(?:[\s,.!]+" + _REPLY_TOKEN + r"\b)*[\s.!,?]*$", re.I
)

# An imperative aimed at the speaker's own business -- "remind me", "show me
# my calendar". Nobody says this to the room.
_FIRST_PERSON_OBJECT = re.compile(r"\b(?:me|my|mine|us|our)\b", re.I)

# Backchannel: the noises a listener makes to show they are listening. They
# are a response TO speech, not a request, and the right reply is almost
# always none. "Mm-hmm" arriving a second after ASH spoke scored 0.34 --
# "mid-conversation" -- which cleared the 0.32 bar and got a full model
# answer: a recital of every window the user had switched between, to a
# sound. Unless ASH just asked something, these are not addressed to it.
_BACKCHANNEL = re.compile(
    r"^\s*(?:m+[\s-]*h?m+|uh[\s-]*huh|uh[\s-]*uh|hm+|huh|ah+|oh+|ok(?:ay)?|"
    r"yeah|yep|yup|yes|right|sure|cool|nice|alright|i\s+see|got\s+it|gotcha|"
    r"true|exactly|indeed)[\s.!,?]*$", re.I)

# Plain narration about oneself, with nobody else in it.
_FIRST_PERSON_ONLY = re.compile(r"^\s*(?:i|i'm|im|i've|ive|my|we|we're)\b", re.I)


@dataclass
class AddressVerdict:
    """What the evidence came to, and what it was."""

    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def addressed(self) -> bool:
        return self.confidence >= ADDRESSED_AT

    @property
    def likely(self) -> bool:
        return LIKELY_AT <= self.confidence < ADDRESSED_AT

    def __bool__(self) -> bool:
        return self.addressed

    def as_dict(self) -> dict:
        return {
            "addressed": self.addressed,
            "likely": self.likely,
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
        }


class Conversation:
    """The turn-taking clock: when either side last spoke.

    This is the whole of the continuity story. A human does not re-say your
    name every sentence; for a window after ASH speaks, what it hears is
    assumed to be the other half of the exchange unless the words argue
    otherwise.
    """

    def __init__(self, window_s: float = FOLLOW_UP_S):
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._ash_spoke: float = 0.0
        self._user_turn: float = 0.0
        self._asked: bool = False

    def note_ash_spoke(self, ts: Optional[float] = None, asked: bool = False) -> None:
        """`asked` is whether what ASH said invites an answer -- a question,
        or an offer. It is what separates "yeah" meaning yes from "yeah"
        meaning go on."""
        with self._lock:
            self._ash_spoke = ts or time.time()
            self._asked = bool(asked)

    def expects_reply(self, now: Optional[float] = None) -> bool:
        """ASH asked something and the exchange is still open."""
        with self._lock:
            asked = self._asked
        return asked and self.openness(now) > 0.02

    def note_user_turn(self, ts: Optional[float] = None) -> None:
        """The user addressed ASH. Holds the window open for their next
        sentence even before ASH has managed to answer."""
        with self._lock:
            self._user_turn = ts or time.time()

    def close(self) -> None:
        """The exchange is over (muted, paused, or told to stop)."""
        with self._lock:
            self._ash_spoke = self._user_turn = 0.0
            self._asked = False

    def openness(self, now: Optional[float] = None) -> float:
        """0..1: how live the exchange is right now. Decays smoothly -- a hard
        window makes ASH deaf one second and credulous the next."""
        now = now or time.time()
        with self._lock:
            last = max(self._ash_spoke, self._user_turn)
        if not last or now < last:
            return 0.0
        elapsed = now - last
        if elapsed > self.window_s * 2:
            return 0.0
        return math.exp(-elapsed / max(1.0, self.window_s / 2.0))


conversation = Conversation()


# ---------------------------------------------------------------------------
def score(text: str, now: Optional[float] = None,
          convo: Optional[Conversation] = None) -> AddressVerdict:
    """Weigh the evidence that `text` was said to ASH."""
    raw = (text or "").strip()
    if not raw:
        return AddressVerdict(0.0, ["empty"])

    convo = conversation if convo is None else convo
    reasons: List[str] = []
    conf = 0.0

    second = bool(_SECOND_PERSON.search(raw))
    question = bool(_QUESTION.search(raw))
    command = bool(_COMMAND.search(raw))
    polite = bool(_POLITE_OPENER.search(raw))
    vocative = bool(_VOCATIVE.search(raw))
    open_now = convo.openness(now)

    # ---- a noise made to show someone is listening -----------------------
    if _BACKCHANNEL.match(raw) and not vocative:
        if convo.expects_reply(now):
            conf = 0.45 + 0.20 * open_now
            return AddressVerdict(min(1.0, conf), ["answering what was just asked"])
        return AddressVerdict(0.0, ["just acknowledging"])

    # ---- the name ------------------------------------------------------
    if _NAME_ANYWHERE.search(raw) and not _NAME_AS_NOUN.search(raw):
        if vocative:
            conf += 0.95
            reasons.append("named")
        elif _NAME_REFERENTIAL.search(raw) and not (second or command):
            # "I told ash about it" -- said to someone else, about ASH.
            reasons.append("name used in the third person")
        else:
            conf += 0.45
            reasons.append("name mentioned")

    # ---- who is being spoken to ----------------------------------------
    if polite:
        conf += 0.55
        reasons.append("direct request")
    if second:
        conf += 0.25
        reasons.append("second person")
    if question:
        conf += 0.35
        reasons.append("a question")
    if second and question:
        # "what's your status", "how are you", "are you there" -- the pair is
        # worth more than its parts. A question with "you" in it is for
        # whoever is listening, and ASH is always listening.
        conf += 0.15
        reasons.append("asking you something")
    if command:
        conf += 0.20 if second else 0.45
        reasons.append("an instruction")
        if _FIRST_PERSON_OBJECT.search(raw):
            # "remind me", "show me my calendar" -- an order about the
            # speaker's own affairs has exactly one plausible recipient.
            conf += 0.25
            reasons.append("about their own business")

    # ---- the conversation already running -------------------------------
    if open_now > 0.02:
        if _REPLY_TOKENS.match(raw):
            conf += 0.45 * open_now
            reasons.append("answering what was just said")
        else:
            conf += 0.35 * open_now
            reasons.append("mid-conversation")

    # ---- evidence against ------------------------------------------------
    other = _OTHER_ADDRESSEE.search(raw)
    if other and not vocative:
        conf -= 0.60
        reasons.append("addressed to " + other.group(1).lower())
    if _FIRST_PERSON_ONLY.match(raw) and not (second or question or command or vocative):
        # "I'm exhausted", "my back hurts" -- said out loud, not to anyone.
        conf -= 0.20
        reasons.append("talking to themselves")

    conf = max(0.0, min(1.0, conf))
    return AddressVerdict(conf, reasons)


def is_addressed(text: str, now: Optional[float] = None) -> bool:
    """Back-compatible boolean: certainly spoken to."""
    return score(text, now).addressed
