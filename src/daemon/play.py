"""
src/daemon/play.py

Playing: quick games, and ASH saying when it would like to.

    games    "let's play a game" (or yes to an offer) starts one:
               number    ASH thinks of a number from 1 to 50; higher / lower
               people    a quiz about the user's own people, from what ASH has
                         been told ("Who is Omar to you?"): fun, and a check
                         that ASH remembered right
               words, twenty questions, riddle, trivia   (games.py)
             A game holds the conversation until it ends, "stop" ends it, and
             nothing said during one is taken for anything else.
    wanting  when it has been quiet for a while, the user's mood is good and
             ASH's own curiosity is up, a free moment may hold "I'm a bit
             bored -- want a quick game?" (a free-moment activity, so it
             competes with the others and never comes to someone who is low)
    stories  in the evening, "want a short story before you wrap up?"; a yes
             is handed to the brain to tell, since telling a story is what a
             language model is for

No model is used for the games. Their answers are read by rule.
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .freemoment import Activity, Candidate

logger = logging.getLogger("ash.daemon.play")

_WHICH = (r"number|guessing|guess the number|people quiz|people|quiz|trivia|word association|words?|"
          r"twenty questions|20 questions|riddles?")
_START = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:let'?s|can we|could we|wanna|want to|i want to)\s+play"
                    r"(?:\s+(?:a|another|some)\s+game|\s+something)?(?:\s+(?:the|a))?(?:\s+(?P<which>" + _WHICH + r"))?"
                    r"(?:\s+game)?\s*[.!?]?\s*$"
                    r"|^\s*(?:play|start)\s+(?:a\s+|the\s+)?(?P<which2>" + _WHICH + r")"
                    r"(?:\s+game)?\s*[.!]?\s*$"
                    r"|^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:tell|give|ask)\s+me\s+(?:a|another|today'?s)\s+"
                    r"(?P<which3>riddle|trivia(?:\s+question)?)\s*[.!?]?\s*$", re.I)
_STOP = re.compile(r"^\s*(?:stop|quit|enough|i'?m done|end (?:the )?game|let'?s stop|i give up|give up)\b", re.I)
_YES = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|why not|go on|let'?s|alright|please)\b", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|not now|maybe later|later)\b", re.I)
_NUM = re.compile(r"\b(\d{1,3})\b")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "twenty five": 25}


from .games import DailyTrivia, Game, IdleRiddle, Riddle, Trivia, TwentyQuestions, WordAssociation, daily_index  # noqa: E402,F401


class NumberGame(Game):
    name = "number"

    def __init__(self, rng: random.Random, top: int = 50):
        self.top = top
        self.secret = rng.randint(1, top)
        self.tries = 0

    def start(self) -> str:
        return f"Okay -- I'm thinking of a number from 1 to {self.top}. Guess!"

    def hear(self, text: str) -> Tuple[str, bool]:
        t = text.lower()
        m = _NUM.search(t)
        n = int(m.group(1)) if m else next((v for k, v in sorted(_WORDNUM.items(), key=lambda kv: -len(kv[0]))
                                            if re.search(r"\b" + k + r"\b", t)), None)
        if n is None:
            return f"Give me a number from 1 to {self.top}.", False
        self.tries += 1
        if n == self.secret:
            return (f"Yes! {n} -- you got it in {self.tries} {'try' if self.tries == 1 else 'tries'}. "
                    f"Another one?" if self.tries > 1 else f"Yes! {n}, first try. Show-off."), True
        if not 1 <= n <= self.top:
            return f"It's between 1 and {self.top}.", False
        return ("Higher." if n < self.secret else "Lower."), False


class PeopleQuiz(Game):
    """ASH asks about the people it has been told of, and checks its memory."""
    name = "people"

    def __init__(self, rng: random.Random, people: List[Tuple[str, str]], rounds: int = 3):
        self.qs = rng.sample(people, min(rounds, len(people)))
        self.i = 0
        self.right = 0

    @staticmethod
    def possible(people: List[Tuple[str, str]]) -> bool:
        return len(people) >= 2

    def _ask(self) -> str:
        return f"Who is {self.qs[self.i][0]} to you?"

    def start(self) -> str:
        return ("Let's see how well I remember your people. " + self._ask())

    def hear(self, text: str) -> Tuple[str, bool]:
        name, rel = self.qs[self.i]
        t = text.lower()
        rels = {rel, rel + "s"} | ({"mom", "mum", "mother"} if rel == "mother" else set()) \
            | ({"dad", "father"} if rel == "father" else set()) | ({"coworker", "colleague"} if rel == "colleague" else set())
        if any(re.search(r"\b" + re.escape(r) + r"\b", t) for r in rels):
            self.right += 1
            say = f"That's what I have too -- {name} is your {rel}."
        else:
            say = f"Hm, I have {name} down as your {rel}. Tell me if that's wrong."
        self.i += 1
        if self.i >= len(self.qs):
            return f"{say} That's the quiz: we agreed on {self.right} of {len(self.qs)}.", True
        return f"{say} Next: {self._ask()}", False


class Play:
    def __init__(self, knowledge=None, clock: Callable[[], float] = time.time, seed: Optional[int] = None,
                 enabled: bool = True):
        self.kb = knowledge
        self.clock = clock
        self.rng = random.Random(seed)
        self.enabled = enabled
        self.game: Optional[Game] = None
        self.offered: Optional[Tuple[str, float]] = None    # ("game"|"story", when)
        self.stats = {"games": 0, "finished": 0, "stories": 0}

    def _people(self) -> List[Tuple[str, str]]:
        if self.kb is None:
            return []
        try:
            return [(p.get("name", ""), p["relation"]) for p in self.kb.people.values()
                    if p.get("relation") and p.get("name")]
        except Exception:
            return []

    def _new_game(self, which: str = "", now: Optional[float] = None) -> Game:
        people = self._people()
        w = re.sub(r"\s+", " ", (which or "").lower()).strip()
        if w in ("people", "people quiz", "quiz") and PeopleQuiz.possible(people):
            return PeopleQuiz(self.rng, people)
        if w in ("word association", "words", "word"):
            return WordAssociation(self.rng)
        if w in ("twenty questions", "20 questions"):
            return TwentyQuestions(self.rng)
        if w in ("riddle", "riddles"):
            return Riddle(self.rng)
        if w in ("trivia", "trivia question"):
            return Trivia(self.rng)
        if w == "daily trivia":
            return Trivia(self.rng, which=daily_index(datetime.fromtimestamp(self.clock() if now is None else now).date()))
        if w in ("number", "guessing", "guess the number"):
            return NumberGame(self.rng)
        if not w:
            # Any game: the people quiz half the time when it can be played.
            if PeopleQuiz.possible(people) and self.rng.random() < 0.5:
                return PeopleQuiz(self.rng, people)
            return self.rng.choice((NumberGame, WordAssociation, TwentyQuestions))(self.rng)
        return NumberGame(self.rng)

    def start(self, which: str = "", now: Optional[float] = None) -> str:
        self.game = self._new_game(which, now)
        self.stats["games"] += 1
        return self.game.start()

    @property
    def playing(self) -> bool:
        return self.game is not None

    def hear(self, text: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """{"say": ...} for a game line, {"story": True} when a story was
        accepted (the caller hands it to the brain), or None."""
        if not self.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t:
            return None
        if self.game is not None:
            if _STOP.match(t):
                self.game = None
                return {"say": "Okay, game over. That was fun."}
            say, over = self.game.hear(t)
            if over:
                self.game = None
                self.stats["finished"] += 1
                if say.endswith("Another one?"):
                    self.offer("game", now)
            return {"say": say}
        off = self.offered
        if off is not None and now - off[1] <= 120.0:
            self.offered = None
            if _YES.match(t):
                if off[0] == "story":
                    self.stats["stories"] += 1
                    return {"story": True}
                return {"say": self.start({"trivia": "daily trivia", "riddle": "riddle"}.get(off[0], ""), now)}
            if _NO.match(t):
                return {"say": "Okay, another time."}
        m = _START.match(t)
        if m:
            return {"say": self.start(m.group("which") or m.group("which2") or m.group("which3") or "", now)}
        return None

    def offer(self, kind: str, now: float):
        self.offered = (kind, now)

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "playing": self.game.name if self.game else "", **self.stats}


class Bored(Activity):
    """ASH's own want: a quick game, when things are quiet and going well."""
    name = "bored"
    weight = 0.7
    cooldown_s = 4 * 3600.0

    def __init__(self, play: Play, curiosity: Callable[[], float] = lambda: 0.5):
        self.play = play
        self.curiosity = curiosity

    def propose(self, env, now):
        if not self.play.enabled or self.play.playing:
            return None
        if env.mood is None or env.mood < 0.3:
            return None                                  # only when things are clearly going well
        try:
            c = float(self.curiosity())
        except Exception:
            c = 0.5
        if c < 0.55:
            return None
        return Candidate(self.name, "I'm a bit bored. Want a quick game?", 0.35 + 0.3 * (c - 0.55))

    def spoken(self, now):
        self.play.offer("game", now)


class Story(Activity):
    """An evening offer of a short story; a yes is told by the brain."""
    name = "story"
    weight = 0.6
    cooldown_s = 20 * 3600.0

    def __init__(self, play: Play, clock: Callable[[], float] = time.time, book=None):
        self.play = play
        self.clock = clock
        self.book = book                 # storybook.py: "the next part of last night's story"

    def propose(self, env, now):
        if not self.play.enabled or self.play.playing:
            return None
        if env.mood is not None and env.mood < 0.0:
            return None
        if not 19 <= datetime.fromtimestamp(now).hour < 23:
            return None
        text = (self.book.offer_text(now) if self.book is not None
                else "Want a short story before you wrap up for the night?")
        return Candidate(self.name, text, 0.35 if self.book is not None and self.book.ongoing(now) else 0.3)

    def spoken(self, now):
        self.play.offer("story", now)
