"""
src/daemon/storybook.py

A story that continues across evenings.

The evening story is told by the brain (telling stories is what a language
model is for). This keeps where it got to: the end of the last part, which
part that was, and when. The next evening's offer becomes "Want the next
part of last night's story?", and the brain is asked to carry on from the
words it ended on rather than start over. A story left for more than a week
is finished; the next one starts fresh. "Tell me a new story" starts fresh
too; "what happens next?" / "continue the story" asks for the next part
there and then.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.daemon.storybook")

DAY = 86400.0
KEEP_S = 7 * DAY
TAIL = 700

NEW_PROMPT = ("Tell me a short, gentle story -- about two minutes to read aloud. End it at a soft cliffhanger, "
              "so it can be continued another evening.")

_NEXT = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:can\s+you\s+)?(?:continue|go\s+on\s+with)\s+(?:the|our|that|last\s+night'?s)\s+story"
                   r"|what\s+happens\s+next(?:\s+in\s+the\s+story)?|(?:tell\s+me\s+)?the\s+next\s+part(?:\s+of\s+the\s+story)?)"
                   r"\s*[?.!]*\s*$", re.I)
_NEW = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:tell\s+me\s+|start\s+)?a\s+(?:new|different)\s+story\s*(?:please)?\s*[?.!]*\s*$",
                  re.I)


class StoryBook:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path = path
        self.clock = clock
        self.story: Optional[Dict[str, Any]] = None      # {"tail", "part", "t"}
        self.waiting = False                             # the next answer is a part of the story
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                s = json.load(fh).get("story")
            if s and s.get("tail"):
                self.story = {"tail": str(s["tail"])[-TAIL:], "part": int(s.get("part", 1)), "t": float(s.get("t", 0))}
        except Exception:
            logger.warning("Could not read %s", self.path, exc_info=True)

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"story": self.story}, fh)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("Could not save %s", self.path, exc_info=True)

    def ongoing(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.story is not None and now - self.story["t"] <= KEEP_S

    def offer_text(self, now: float) -> str:
        if self.ongoing(now):
            return "Want the next part of last night's story?" if now - self.story["t"] <= 1.5 * DAY \
                else "Want the next part of our story?"
        return "Want a short story before you wrap up for the night?"

    def prompt(self, now: Optional[float] = None, fresh: bool = False) -> str:
        """What to ask the brain for, and expect the story in the next answer."""
        now = self.clock() if now is None else now
        self.waiting = True
        if fresh or not self.ongoing(now):
            self.story = None
            return NEW_PROMPT
        s = self.story
        return (f"Continue the story you were telling me -- this is part {s['part'] + 1}, about two minutes to read "
                f"aloud, in the same voice, with the same characters. Don't retell what happened. End at a soft "
                f"cliffhanger. It left off here: “{s['tail'][-400:]}”")

    def told(self, text: str, now: Optional[float] = None):
        """The answer that followed a story prompt."""
        if not self.waiting or not (text or "").strip():
            return
        self.waiting = False
        now = self.clock() if now is None else now
        part = (self.story["part"] + 1) if self.ongoing(now) else 1
        self.story = {"tail": text.strip()[-TAIL:], "part": part, "t": now}
        self.save()

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """A request about the story: the prompt to hand the brain, or None."""
        t = (text or "").strip()
        if _NEW.match(t):
            return self.prompt(now, fresh=True)
        if _NEXT.match(t) and self.ongoing(now):
            return self.prompt(now)
        return None

    def status(self) -> Dict[str, Any]:
        return {"ongoing": self.ongoing(), "part": self.story["part"] if self.story else 0}
