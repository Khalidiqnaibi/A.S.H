"""
src/daemon/games.py

More games for play.py, all read by rule -- no model:

    word association   ASH says a word, you say the first word it brings to
                       mind, ASH answers yours; six rounds, then the chain
    twenty questions   ASH thinks of a thing; yes/no questions about size,
                       alive, eating, where it lives...; "is it a cat?" to
                       guess; twenty questions at most
    riddle             one riddle; "hint" for a hint, "give up" for the answer
    trivia             a question with one short answer; also one a day,
                       offered at a free moment (DailyTrivia)

Each is a Game: start() says the opening line, hear(text) -> (what to say,
whether it is over).
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from .freemoment import Activity, Candidate


class Game:
    name = "game"

    def start(self) -> str:
        raise NotImplementedError

    def hear(self, text: str) -> Tuple[str, bool]:
        raise NotImplementedError


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


_GIVE_UP = re.compile(r"\b(?:i\s+give\s+up|give\s+up|tell\s+me|what\s+is\s+it|i\s+don'?t\s+know|no\s+idea|reveal)\b", re.I)


# ---------------------------------------------------------------- word association
ASSOC: Dict[str, List[str]] = {
    "sun": ["beach", "summer", "light"], "beach": ["sand", "waves", "sun"], "summer": ["holiday", "heat", "ice cream"],
    "ice": ["cold", "winter", "skating"], "cold": ["snow", "winter", "coffee"], "snow": ["mountain", "winter", "white"],
    "winter": ["snow", "coat", "fire"], "fire": ["smoke", "camp", "red"], "red": ["apple", "rose", "fire"],
    "apple": ["tree", "pie", "phone"], "tree": ["forest", "leaf", "bird"], "forest": ["wolf", "path", "tree"],
    "wolf": ["moon", "howl", "pack"], "moon": ["night", "star", "rocket"], "night": ["dream", "stars", "owl"],
    "star": ["sky", "night", "film"], "sky": ["blue", "cloud", "bird"], "blue": ["ocean", "sky", "jeans"],
    "ocean": ["whale", "wave", "salt"], "whale": ["ocean", "song", "giant"], "rain": ["umbrella", "cloud", "puddle"],
    "cloud": ["rain", "sky", "storage"], "coffee": ["morning", "cup", "beans"], "morning": ["coffee", "alarm", "sunrise"],
    "tea": ["cup", "biscuit", "green"], "book": ["library", "story", "page"], "story": ["book", "night", "dragon"],
    "dragon": ["fire", "castle", "knight"], "castle": ["king", "tower", "sand"], "king": ["crown", "queen", "chess"],
    "chess": ["king", "board", "move"], "music": ["song", "guitar", "dance"], "song": ["music", "bird", "radio"],
    "guitar": ["strings", "rock", "song"], "dance": ["music", "party", "shoes"], "party": ["cake", "music", "friends"],
    "cake": ["birthday", "candle", "chocolate"], "birthday": ["cake", "present", "party"],
    "chocolate": ["cake", "sweet", "brown"], "dog": ["bone", "walk", "cat"], "cat": ["mouse", "dog", "purr"],
    "mouse": ["cheese", "cat", "computer"], "computer": ["keyboard", "code", "screen"], "code": ["bug", "computer", "secret"],
    "bug": ["code", "insect", "fix"], "phone": ["call", "screen", "apple"], "car": ["road", "wheel", "drive"],
    "road": ["trip", "car", "map"], "trip": ["suitcase", "road", "holiday"], "holiday": ["beach", "trip", "rest"],
    "city": ["lights", "traffic", "tower"], "school": ["teacher", "book", "bell"], "work": ["office", "coffee", "desk"],
    "food": ["pizza", "kitchen", "hungry"], "pizza": ["cheese", "italy", "slice"], "cheese": ["mouse", "pizza", "milk"],
    "milk": ["cow", "cookies", "white"], "cow": ["milk", "farm", "grass"], "farm": ["cow", "tractor", "barn"],
    "garden": ["flower", "bee", "soil"], "flower": ["bee", "rose", "spring"], "bee": ["honey", "flower", "sting"],
    "honey": ["bee", "bear", "sweet"], "bear": ["honey", "forest", "teddy"], "game": ["play", "win", "board"],
    "robot": ["metal", "future", "beep"], "future": ["robot", "time", "space"], "space": ["rocket", "stars", "moon"],
    "rocket": ["space", "launch", "fast"], "time": ["clock", "future", "watch"], "clock": ["time", "tick", "alarm"],
    "water": ["river", "ocean", "drink"], "river": ["bridge", "fish", "water"], "fish": ["river", "ocean", "chips"],
    "bridge": ["river", "city", "cards"], "mountain": ["snow", "climb", "view"], "light": ["lamp", "sun", "dark"],
    "dark": ["night", "shadow", "light"], "dream": ["sleep", "night", "wish"], "sleep": ["bed", "dream", "night"],
    "bed": ["pillow", "sleep", "blanket"], "friend": ["laugh", "party", "trust"], "home": ["family", "door", "warm"],
}
SEEDS = ("sun", "coffee", "dragon", "music", "ocean", "robot", "garden", "space", "book", "winter")


class WordAssociation(Game):
    name = "words"

    def __init__(self, rng: random.Random, rounds: int = 6):
        self.rng = rng
        self.rounds = rounds
        self.word = rng.choice(SEEDS)
        self.chain = [self.word]
        self.used: Set[str] = {self.word}

    def start(self) -> str:
        return f"Word association: say the first word that comes to mind. Mine is “{self.word}”."

    def _answer(self, w: str) -> str:
        options = [x for x in ASSOC.get(w, []) if x not in self.used]
        if not options:
            options = [x for x in ASSOC if x not in self.used] or list(ASSOC)
        return self.rng.choice(options)

    def hear(self, text: str) -> Tuple[str, bool]:
        ws = [w for w in _words(text) if w not in ("um", "uh", "the", "a", "an", "hmm", "like")]
        if not ws:
            return "Just one word -- whatever pops up.", False
        theirs = ws[-1] if len(ws) <= 3 else ws[0]
        self.chain.append(theirs)
        self.used.add(theirs)
        if len(self.chain) >= self.rounds * 2:
            return f"{theirs.capitalize()}! Great chain: " + " → ".join(self.chain) + ".", True
        mine = self._answer(theirs)
        self.chain.append(mine)
        self.used.add(mine)
        return f"{mine.capitalize()}.", False


# ---------------------------------------------------------------- twenty questions
# Each thing: its names, and what is true of it. A question asks about one of
# these features; any feature not listed is false.
THINGS: List[Tuple[Tuple[str, ...], Set[str]]] = [
    (("cat", "kitten"), {"alive", "animal", "legs", "fur", "pet", "house", "small"}),
    (("dog", "puppy"), {"alive", "animal", "legs", "fur", "pet", "house", "small"}),
    (("elephant",), {"alive", "animal", "legs", "big", "wild", "grey"}),
    (("penguin",), {"alive", "animal", "legs", "bird", "water", "cold", "wild"}),
    (("goldfish", "fish"), {"alive", "animal", "water", "pet", "house", "small"}),
    (("eagle",), {"alive", "animal", "bird", "fly", "legs", "wild"}),
    (("bee",), {"alive", "animal", "insect", "fly", "small", "legs", "wild"}),
    (("apple",), {"alive", "plant", "eat", "small", "red", "kitchen", "house"}),
    (("banana",), {"alive", "plant", "eat", "small", "yellow", "kitchen", "house"}),
    (("oak tree", "tree", "oak"), {"alive", "plant", "big", "wood", "outside"}),
    (("sunflower", "flower"), {"alive", "plant", "yellow", "outside", "small"}),
    (("pizza",), {"eat", "kitchen", "house", "small", "hot"}),
    (("chair",), {"house", "furniture", "legs", "wood"}),
    (("bed",), {"house", "furniture", "big", "legs"}),
    (("refrigerator", "fridge"), {"house", "kitchen", "electric", "big", "metal", "cold"}),
    (("phone", "smartphone", "mobile phone"), {"house", "electric", "small", "screen", "carry"}),
    (("laptop", "computer"), {"house", "electric", "screen", "carry", "metal"}),
    (("television", "tv"), {"house", "electric", "screen", "big"}),
    (("guitar",), {"house", "music", "wood", "carry"}),
    (("car",), {"metal", "big", "wheels", "outside", "electric", "transport"}),
    (("bicycle", "bike"), {"metal", "wheels", "outside", "transport"}),
    (("aeroplane", "airplane", "plane"), {"metal", "big", "fly", "transport", "wings"}),
    (("boat", "ship"), {"water", "transport", "big", "outside"}),
    (("book",), {"house", "paper", "small", "carry"}),
    (("pencil", "pen"), {"small", "carry", "wood", "house"}),
    (("umbrella",), {"carry", "outside", "small"}),
    (("moon",), {"big", "sky", "outside", "round"}),
    (("sun",), {"big", "sky", "outside", "hot", "round", "yellow"}),
    (("snowman",), {"cold", "outside", "white"}),
    (("clock",), {"house", "round", "electric"}),
    (("key",), {"small", "metal", "carry", "house"}),
    (("ball", "football"), {"round", "small", "outside", "toy"}),
]

# question words -> the feature asked about
FEATURES: List[Tuple[str, str]] = [
    (r"\balive|living|grow", "alive"), (r"\banimal|creature|mammal", "animal"), (r"\bplant|vegetable|fruit", "plant"),
    (r"\bbird\b", "bird"), (r"\binsect|bug\b", "insect"),
    (r"\bfly|flies|flying|wings?\b", "fly"), (r"\bwater|swim|sea|ocean|river", "water"),
    (r"\beat|edible|food|taste", "eat"), (r"\bkitchen|cook", "kitchen"),
    (r"\bhouse|home|indoors|inside", "house"), (r"\boutside|outdoors", "outside"),
    (r"\bbig|large|huge|bigger|larger", "big"), (r"\bsmall|little|tiny|smaller|fit in", "small"),
    (r"\blegs?\b|walk", "legs"), (r"\bfur|furry|hair", "fur"), (r"\bpet\b", "pet"), (r"\bwild\b", "wild"),
    (r"\belectric|electronic|plug|battery|power", "electric"), (r"\bscreen|display", "screen"),
    (r"\bmetal|metallic|steel", "metal"), (r"\bwood|wooden", "wood"), (r"\bpaper\b", "paper"),
    (r"\bwheels?\b", "wheels"), (r"\btransport|travel|ride|drive|vehicle", "transport"),
    (r"\bfurniture|sit on|sleep on", "furniture"), (r"\bmusic|instrument|play music", "music"),
    (r"\bcarry|pocket|hold it|portable", "carry"), (r"\bsky|space\b", "sky"), (r"\bhot|warm", "hot"),
    (r"\bcold|freez|ice", "cold"), (r"\bround|circle|sphere", "round"), (r"\btoy|play with", "toy"),
    (r"\bred\b", "red"), (r"\byellow\b", "yellow"), (r"\bwhite\b", "white"), (r"\bgr[ae]y\b", "grey"),
]
_GUESS = re.compile(r"^\s*(?:is\s+it|it'?s|could\s+it\s+be|maybe|i\s+think\s+it'?s|my\s+guess\s+is|guess)\s+"
                    r"(?:an?\s+|the\s+)?(?P<x>[a-z][a-z ]{1,24}?)\s*[?.!]*\s*$", re.I)


class TwentyQuestions(Game):
    name = "twenty"

    def __init__(self, rng: random.Random, limit: int = 20):
        self.names, self.feats = rng.choice(THINGS)
        self.limit = limit
        self.asked = 0

    def start(self) -> str:
        return ("Twenty questions: I'm thinking of something. Ask me yes-or-no questions -- is it alive, "
                "is it bigger than a breadbox, can you eat it -- or guess with “is it a ...?”")

    def hear(self, text: str) -> Tuple[str, bool]:
        t = (text or "").strip()
        if _GIVE_UP.search(t):
            return f"It was a {self.names[0]}! Another time.", True
        m = _GUESS.match(t)
        feature = self._feature(t)
        if m and (feature is None or m.group("x").lower().strip() in self._all_names()):
            self.asked += 1
            guess = m.group("x").lower().strip()
            if guess in self.names:
                return f"Yes! It's a {self.names[0]} -- you got it in {self.asked}.", True
            if guess in self._all_names():
                return self._after(f"No, it's not a {guess}.")
        if feature is None:
            return ("I can't tell from that one. Ask about size, alive, eating, where it lives, legs, "
                    "electric, metal or wood, flying, water -- or guess."), False
        self.asked += 1
        yes = feature in self.feats
        return self._after("Yes." if yes else "No.")

    def _after(self, said: str) -> Tuple[str, bool]:
        if self.asked >= self.limit:
            return f"{said} That's twenty -- it was a {self.names[0]}!", True
        left = self.limit - self.asked
        return (said + (f" {left} left." if left <= 5 else "")), False

    @staticmethod
    def _feature(text: str) -> Optional[str]:
        t = (text or "").lower()
        for pat, f in FEATURES:
            if re.search(pat, t):
                return f
        return None

    @staticmethod
    def _all_names() -> Set[str]:
        return {n for names, _ in THINGS for n in names}


# ---------------------------------------------------------------- riddles
# (the riddle, (the answer as said, then the words that count as it), a hint)
RIDDLES: List[Tuple[str, Tuple[str, ...], str]] = [
    ("What has keys but can't open locks?", ("a piano", "piano", "keyboard"), "You play it."),
    ("What gets wetter the more it dries?", ("a towel", "towel"), "You use it after a shower."),
    ("What has hands but can't clap?", ("a clock", "clock", "watch"), "It tells you something all day."),
    ("What has a neck but no head?", ("a bottle", "bottle"), "You might pour from it."),
    ("What can you catch but not throw?", ("a cold", "cold"), "You'd rather not catch it."),
    ("What has many teeth but can't bite?", ("a comb", "comb"), "It goes through your hair."),
    ("What goes up but never comes down?", ("your age", "age"), "Everyone has one, and it only grows."),
    ("What has one eye but can't see?", ("a needle", "needle"), "It's used for sewing."),
    ("What runs but never walks, and has a bed but never sleeps?", ("a river", "river"), "It flows to the sea."),
    ("The more of them you take, the more you leave behind. What are they?", ("footsteps", "footsteps", "steps"),
     "You take them when you walk."),
    ("What belongs to you, but others use it more than you do?", ("your name", "name"), "People call you by it."),
    ("What can travel around the world while staying in a corner?", ("a stamp", "stamp"), "It goes on a letter."),
    ("What has a head and a tail but no body?", ("a coin", "coin"), "You flip it."),
    ("What is full of holes but still holds water?", ("a sponge", "sponge"), "It's in the kitchen sink."),
    ("What can't talk but will reply when spoken to?", ("an echo", "echo"), "Try shouting in a canyon."),
]


class Riddle(Game):
    name = "riddle"

    def __init__(self, rng: random.Random, which: Optional[int] = None):
        self.q, self.answers, self.hint = RIDDLES[which if which is not None else rng.randrange(len(RIDDLES))]
        self.tries = 0
        self.hinted = False

    def start(self) -> str:
        return "Here's a riddle: " + self.q

    def hear(self, text: str) -> Tuple[str, bool]:
        t = (text or "").lower()
        it = self.answers[0]                 # said with its article: "a piano", "footsteps"
        if any(re.search(r"\b" + re.escape(a) + r"s?\b", t) for a in self.answers[1:]):
            return (f"Yes! It's {it}." if self.tries == 0 else f"Got it -- {it}!"), True
        if re.search(r"\bhint\b|\bclue\b", t):
            self.hinted = True
            return "Hint: " + self.hint, False
        if _GIVE_UP.search(t):
            return f"It's {it}.", True
        self.tries += 1
        if self.tries >= 3:
            return f"Not quite -- it's {it}.", True
        return ("Not quite. Want a hint?" if not self.hinted else "Not quite -- try again."), False


# ---------------------------------------------------------------- trivia
TRIVIA: List[Tuple[str, Tuple[str, ...]]] = [
    ("What is the largest planet in our solar system?", ("jupiter",)),
    ("How many legs does a spider have?", ("8", "eight")),
    ("What gas do plants take in from the air?", ("carbon dioxide", "co2")),
    ("What is the chemical symbol for gold?", ("au",)),
    ("Which ocean is the largest?", ("pacific",)),
    ("How many continents are there?", ("7", "seven")),
    ("What is the fastest land animal?", ("cheetah",)),
    ("What is the boiling point of water at sea level, in Celsius?", ("100", "a hundred", "one hundred")),
    ("Which planet is known as the red planet?", ("mars",)),
    ("What is the hardest natural substance?", ("diamond",)),
    ("How many sides does a hexagon have?", ("6", "six")),
    ("What is the smallest prime number?", ("2", "two")),
    ("Which language has the most native speakers?", ("mandarin", "chinese")),
    ("What do bees make?", ("honey",)),
    ("What is the capital of Japan?", ("tokyo",)),
    ("How many minutes are in a day?", ("1440",)),
    ("What is frozen water called?", ("ice",)),
    ("Which instrument has 88 keys?", ("piano",)),
    ("What is the largest mammal?", ("blue whale", "whale")),
    ("In computing, what does CPU stand for?", ("central processing unit",)),
    ("How many bits are in a byte?", ("8", "eight")),
    ("What is the closest star to Earth?", ("sun", "the sun")),
    ("Which element does 'O' stand for on the periodic table?", ("oxygen",)),
    ("How many hours are in a week?", ("168",)),
]


class Trivia(Game):
    name = "trivia"

    def __init__(self, rng: random.Random, which: Optional[int] = None):
        self.i = which if which is not None else rng.randrange(len(TRIVIA))
        self.q, self.answers = TRIVIA[self.i]
        self.tries = 0

    def start(self) -> str:
        return "Trivia: " + self.q

    def hear(self, text: str) -> Tuple[str, bool]:
        t = " ".join(_words(text)) + " " + " ".join(re.findall(r"\d+", text or ""))
        if any(re.search(r"(?:^|\s)" + re.escape(a) + r"(?:\s|$)", t) for a in self.answers):
            return f"Correct! {self.answers[0].capitalize()}.", True
        if _GIVE_UP.search(text or ""):
            return f"It's {self.answers[0]}.", True
        self.tries += 1
        if self.tries >= 2:
            return f"Not quite -- it's {self.answers[0]}.", True
        return "Not quite. One more try?", False


def daily_index(day: date) -> int:
    """The same trivia question all day, a different one each day."""
    return day.toordinal() % len(TRIVIA)


class DailyTrivia(Activity):
    """Once a day, at a free moment: today's trivia question."""
    name = "daily_trivia"
    weight = 0.5
    cooldown_s = 20 * 3600.0

    def __init__(self, play):
        self.play = play
        self.asked_on: Optional[str] = None

    def propose(self, env, now):
        if not self.play.enabled or self.play.playing:
            return None
        if env.mood is not None and env.mood < 0.0:
            return None
        today = date.fromtimestamp(now)
        if self.asked_on == today.isoformat():
            return None
        return Candidate(self.name, "Want today's trivia question?", 0.3)

    def spoken(self, now):
        self.asked_on = date.fromtimestamp(now).isoformat()
        self.play.offer("trivia", now)


class IdleRiddle(Activity):
    """A riddle, now and then, when it has been quiet and things are fine."""
    name = "riddle"
    weight = 0.4
    cooldown_s = 6 * 3600.0

    def __init__(self, play):
        self.play = play

    def propose(self, env, now):
        if not self.play.enabled or self.play.playing:
            return None
        if env.mood is None or env.mood < 0.2:
            return None
        return Candidate(self.name, "I've got a riddle for you, if you want one.", 0.25)

    def spoken(self, now):
        self.play.offer("riddle", now)
