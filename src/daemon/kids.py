"""
src/daemon/kids.py

Kids mode: ASH for a child -- simpler, kinder, nothing grown-up, and a time limit.

    kids mode on / kids mode for 30 minutes / a kid is using you          on (an hour unless a time is said)
    kids mode off                                                          off -- after the passphrase, if one is set
                                                                           (kids.passphrase); a guest's voice can't

While it is on:
  * grown-up things are refused kindly, by rule, before anything else sees them: email and texts, the calendar, files
    and notes, opening or closing apps, the smart home, guard mode, buying, deleting, privacy and the camera settings
  * the language model is told it is talking with a child (short, simple, kind, nothing scary or adult), and anything
    it says with a word from the not-for-kids list is replaced with "Let's talk about something else!"
  * jokes, songs, games, stories, facts and questions all still work
  * when the time is up ASH says so once, and then only "kids mode off" works
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|turn|switch)[,\s]+)*"
_ON = re.compile(_LEAD + r"(?:(?:turn\s+on\s+|start\s+|switch\s+(?:on|to)\s+)?(?:kids?|child|children'?s?)\s+mode(?:\s+on)?|"
                 r"(?:my\s+)?(?:kid|son|daughter|child|little\s+one)\s+(?:is\s+)?(?:using\s+you|wants\s+to\s+(?:talk|play))|"
                 r"(?:a|my)\s+kid\s+is\s+(?:here|using\s+you))(?:\s+for\s+(?P<n>\d{1,3})\s*(?P<u>min(?:ute)?s?|hours?|h))?\s*(?:please|now)?\s*[.!]*\s*$", re.I)
_OFF = re.compile(_LEAD + r"(?:(?:turn\s+off\s+|end\s+|stop\s+|exit\s+)(?:kids?|child)\s+mode|(?:kids?|child)\s+mode\s+off)\s*[.!]*\s*$", re.I)

GROWN_UP = re.compile(
    r"\b(?:e-?mails?|inbox|reply\s+to|text\s+\w+\s+saying|sms|send\s+(?:a\s+)?text|calendar|schedule|meetings?|"
    r"find\s+(?:the|my)\s+\w+|search\s+my\s+files|my\s+files|notes?\b|remember\s+that|where\s+did\s+i|"
    r"open\s+\w+|close\s+\w+|launch|shut\s*down|turn\s+(?:on|off)\s+the\s+(?:lights?|heating|tv|camera|mic)|lights?\s+(?:on|off)|"
    r"guard\s+mode|privacy|buy|order|purchase|pay|delete|password|credit\s+card|bank|phone|clipboard|"
    r"why\s+is\s+my\s+computer|what'?s\s+using\s+my)\b", re.I)
NOT_FOR_KIDS = re.compile(
    r"\b(?:kill(?:ed|ing|s)?|murder\w*|blood\w*|gun\w*|weapon\w*|die[sd]?|dead|death|drugs?|alcohol|beer|wine|drunk|"
    r"sex\w*|naked|nude|porn\w*|damn|hell|shit\w*|fuck\w*|bitch\w*|crap|stupid|idiot|hate|suicide|terror\w*|bomb\w*|war)\b", re.I)
CONTEXT = ("[KIDS MODE -- you are talking with a child. Use short, simple sentences and friendly words. Be kind, playful "
           "and encouraging. Nothing scary, violent, sad or grown-up; no weapons, death, drugs, romance or bad words. If "
           "asked about something like that, gently say it's a question for a grown-up and suggest something fun instead.]")
REFUSE = "That one's a grown-up thing, so not in kids mode. Want a joke, a song or a game instead?"
SAFE = "Let's talk about something else! Want to hear a joke?"


class KidsMode:
    def __init__(self, passphrase: str = "", default_minutes: float = 60.0, clock: Callable[[], float] = time.time):
        self.passphrase = (passphrase or "").strip().lower()
        self.default_minutes = default_minutes
        self.clock = clock
        self.active = False
        self.until = 0.0
        self.ended = False
        self.awaiting_pass = False
        self.stats = {"sessions": 0, "refused": 0, "filtered": 0}

    def parse(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if _ON.match(t):
            return "on"
        if _OFF.match(t):
            return "off"
        return None

    def start(self, text: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        m = _ON.match((text or "").strip())
        minutes = self.default_minutes
        if m and m.group("n"):
            n = float(m.group("n"))
            minutes = n * 60 if (m.group("u") or "").lower().startswith("h") else n
        self.active, self.ended, self.until = True, False, now + minutes * 60
        self.stats["sessions"] += 1
        return (f"Kids mode on for {int(minutes)} minutes! I can tell jokes, sing songs, play games and answer questions. "
                "What shall we do?")

    def stop_asked(self, guest: bool) -> str:
        if not self.active:
            return "Kids mode isn't on."
        if guest:
            return "Only a grown-up can turn kids mode off."
        if self.passphrase:
            self.awaiting_pass = True
            return "What's the passphrase?"
        self.active = self.awaiting_pass = False
        return "Kids mode off."

    def passphrase_heard(self, text: str, guest: bool) -> Optional[str]:
        """The answer to "What's the passphrase?"."""
        if not self.awaiting_pass:
            return None
        self.awaiting_pass = False
        said = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
        said = re.sub(r"^(?:hey\s+)?ash\s+", "", said)
        if not guest and said == re.sub(r"[^\w\s]", "", self.passphrase):
            self.active = False
            return "Kids mode off."
        return "That's not it -- kids mode stays on."

    def time_up(self, now: Optional[float] = None) -> bool:
        """True once, when the time runs out."""
        now = self.clock() if now is None else now
        if self.active and not self.ended and now >= self.until:
            self.ended = True
            return True
        return False

    def refuse(self, text: str) -> Optional[str]:
        """What to say instead, for a grown-up request -- or for anything once the time is up."""
        if not self.active:
            return None
        if self.ended:
            return "Kids mode time is up for now -- ask a grown-up."
        if GROWN_UP.search(text or ""):
            self.stats["refused"] += 1
            return REFUSE
        return None

    def filter(self, reply: str) -> str:
        if self.active and reply and NOT_FOR_KIDS.search(reply):
            self.stats["filtered"] += 1
            return SAFE
        return reply

    def status(self) -> Dict[str, Any]:
        left = max(0, int((self.until - self.clock()) // 60)) if self.active else 0
        return {"active": self.active, "minutes_left": left, "ended": self.ended, **self.stats}
