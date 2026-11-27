"""
src/daemon/pocket.py

BMO's side, by rule -- small things a friend in the room does, from what is really there:

    pocket games   "flip a coin", "roll a die" / "roll 2d6", "pick a number between 1 and 50", "rock paper scissors" (then "rock"), "magic 8 ball,
                   will I pass?" -- instant, and fair (the coin is a coin).
    what to play   "what should I play?" -- the games installed here and when each was last played: "You haven't touched Elden Ring in two weeks."
    ASH's own day  "what did you do while I was away?" / "what have you been up to?" -- only what really happened: it slept and sorted memories,
                   wondered about something, saved questions, watched the machine.
    hype           "hype me up" / "I need motivation" -- facts, never flattery: hours this week, a streak, a thing that went well.
    a quiz         "quiz me on NoSQL" -- fill-in-the-blank questions from the person's own notes and Samsung Notes lines shaped "X is Y"; "hint",
                   "I give up", "next", "stop".
    the music      "play my coding music" / "play something I like" -- what they listen to while coding, or like most, opened in YouTube.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"

_COIN = re.compile(_A + r"(?:flip|toss)\s+(?:a\s+|the\s+)?coin|heads\s+or\s+tails" + _Z, re.I)
_DICE = re.compile(_A + r"roll\s+(?:a\s+|the\s+|some\s+)?(?:(?P<n>\d|one|two|three|four|five|six)\s*)?(?:d(?P<s>\d{1,3})|dice|die|d6)"
                   r"(?:\s+(?:with\s+)?(?P<s2>\d{1,3})\s+sides)?" + _Z, re.I)
_DND = re.compile(_A + r"roll\s+(?P<n>\d{1,2})?d(?P<s>\d{1,3})" + _Z, re.I)
_PICK = re.compile(_A + r"(?:pick|choose|give\s+me|think\s+of)\s+a\s+(?:random\s+)?number(?:\s+(?:between|from)\s+(?P<a>-?\d+)\s+(?:and|to)\s+(?P<b>-?\d+))?" + _Z, re.I)
_RPS = re.compile(_A + r"(?:let'?s\s+)?(?:play\s+)?rock[,\s]+paper[,\s]+(?:and\s+)?scissors?" + _Z, re.I)
_RPS_MOVE = re.compile(r"^\W*(?P<m>rock|paper|scissors?)\W*$", re.I)
_EIGHT = re.compile(_A + r"(?:(?:ask\s+(?:the\s+)?)?magic\s+(?:8|eight)[\s-]?ball|(?:ask\s+)?(?:the\s+)?(?:8|eight)[\s-]?ball)[,:\s]*(?P<q>.*?)\s*[?.!]*\s*$", re.I)
_WORDN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_EIGHT_SAYS = ("It is certain.", "Without a doubt.", "Yes, definitely.", "Most likely.", "Outlook good.", "Signs point to yes.",
               "Reply hazy, try again.", "Ask again later.", "Better not tell you now.", "Cannot predict now.",
               "Don't count on it.", "My reply is no.", "My sources say no.", "Very doubtful.")


class Pocket:
    def __init__(self, rng: Optional[random.Random] = None, clock=time.time):
        self.rng = rng or random.SystemRandom()
        self.clock = clock
        self.rps_at: Optional[float] = None

    def hear(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if self.rps_at is not None and self.clock() - self.rps_at <= 60.0:
            m = _RPS_MOVE.match(t)
            if m:
                self.rps_at = None
                return self._rps(m.group("m").lower().rstrip("s") if m.group("m").lower().startswith("scissor") else m.group("m").lower())
        if _COIN.match(t):
            return self.rng.choice(("Heads.", "Tails."))
        m = _DND.match(t) or _DICE.match(t)
        if m:
            gd = m.groupdict()
            n = gd.get("n")
            n = int(n) if (n or "").isdigit() else _WORDN.get((n or "").lower(), 1)
            sides = int(gd.get("s") or gd.get("s2") or 6)
            if not (1 <= n <= 10 and 2 <= sides <= 100):
                return "I'll roll up to ten dice, two to a hundred sides."
            rolls = [self.rng.randint(1, sides) for _ in range(n)]
            return (f"{rolls[0]}." if n == 1 else ", ".join(str(r) for r in rolls) + f" -- {sum(rolls)} in all.")
        m = _PICK.match(t)
        if m:
            a, b = int(m.group("a") or 1), int(m.group("b") or 100)
            a, b = min(a, b), max(a, b)
            return f"{self.rng.randint(a, b)}."
        if _RPS.match(t):
            self.rps_at = self.clock()
            return "Rock, paper, scissors -- go! Say yours."
        m = _EIGHT.match(t)
        if m and (m.group("q") or "").strip():
            return self.rng.choice(_EIGHT_SAYS)
        if m:
            return "Ask it something -- “magic 8 ball, will it rain tomorrow?”"
        return None

    def _rps(self, you: str) -> str:
        you = "scissors" if you.startswith("scissor") else you
        me = self.rng.choice(("rock", "paper", "scissors"))
        beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        if me == you:
            return f"{me.capitalize()}. A draw."
        return f"{me.capitalize()}. " + ("You win." if beats[you] == me else "I win.")


# ----------------------------------------------------------------------------------------------------------------------------- what to play
_WHAT_PLAY = re.compile(_A + r"(?:what|which)\s+(?:game\s+)?should\s+i\s+play(?:\s+(?:tonight|today|now|next))?|(?:suggest|recommend|pick)\s+(?:me\s+)?a\s+game"
                        r"(?:\s+(?:to\s+play|for\s+tonight))?|what\s+haven'?t\s+i\s+played\s+(?:in\s+a\s+while|lately)" + _Z, re.I)


def what_to_play(games: List[str], last_played: Dict[str, float], now: float, rng: Optional[random.Random] = None) -> str:
    """From the games installed and when each was last seen played: the one left longest, said with how long."""
    if not games:
        return "I don't see any games installed here -- Steam and Epic are what I look in."
    rng = rng or random.Random()
    played = [(g, last_played[g]) for g in games if g in last_played]
    never = [g for g in games if g not in last_played]
    if played:
        g, t = min(played, key=lambda x: x[1])
        days = int((now - t) // 86400)
        if days >= 7:
            gap = f"{days // 7} week{'s' if days >= 14 else ''}" if days < 60 else f"{days // 30} months"
            return f"You haven't touched {g} in {gap}." + (f" Or something new: {rng.choice(never)}." if never else "")
    if never:
        pick = rng.choice(never)
        return f"{pick} -- I haven't seen you play it yet."
    g, _t = min(played, key=lambda x: x[1])
    return f"{g} -- it's the one you've played least lately."


# ------------------------------------------------------------------------------------------------------------------------------ ASH's day
_MY_DAY = re.compile(_A + r"(?:what\s+(?:did\s+you\s+do|have\s+you\s+been\s+(?:up\s+to|doing))(?:\s+(?:while\s+i\s+was\s+(?:away|gone|out)|today|lately|all\s+day))?"
                     r"|what\s+were\s+you\s+doing(?:\s+while\s+i\s+was\s+(?:away|gone))?|how\s+was\s+your\s+day|what'?s\s+new\s+with\s+you)" + _Z, re.I)


def my_day(stats: Dict[str, Any], wonders: int, last_sleep: Optional[Dict[str, Any]], mem_high: bool) -> str:
    """Only what really happened -- read from the runtime's own counters."""
    bits = []
    if last_sleep:
        n = int(last_sleep.get("trained") or 0)
        bits.append("I slept" + (f" and sorted {n} memories" if n else " and tidied my memory"))
    asked = int(stats.get("wonders_asked", 0) or 0)
    if asked:
        bits.append(f"asked you {asked} question{'s' if asked != 1 else ''}")
    if wonders:
        bits.append(f"saved {wonders} more for a good moment")
    obs = int(stats.get("observed", 0) or 0)
    if obs:
        bits.append(f"noticed {obs} things on the computer and kept the ones worth keeping")
    if mem_high:
        bits.append("watched the memory sit above 90%")
    if not bits:
        return "Not much -- I've only just started."
    return "I " + (", ".join(bits[:-1]) + " and " + bits[-1] if len(bits) > 1 else bits[0]).removeprefix("I ") + "."


# ------------------------------------------------------------------------------------------------------------------------------------ hype
_HYPE = re.compile(_A + r"(?:hype\s+me\s+up|motivate\s+me|i\s+need\s+(?:some\s+)?(?:motivation|a\s+pep\s+talk|encouragement)|give\s+me\s+a\s+pep\s+talk|"
                   r"cheer\s+me\s+on|pump\s+me\s+up)" + _Z, re.I)


def hype(week: List[Tuple[str, float]], streak: Optional[Tuple[str, int]], good: Optional[str]) -> str:
    """Facts that are true, put forward -- never flattery."""
    bits = []
    if week:
        n, s = week[0]
        h = s / 3600.0
        bits.append(f"{h:.0f} hour{'s' if round(h) != 1 else ''} on {n} this week" if h >= 1 else f"{int(s // 60)} minutes on {n} this week")
    if streak and streak[1] >= 2:
        bits.append(f"{streak[1]} days in a row on {streak[0]}")
    if good:
        bits.append(good)
    if not bits:
        return "I don't have your wins on record yet -- tell me how things go and I'll keep them. For now: start with the smallest piece."
    return "Here's what's true: " + "; ".join(bits) + ". Keep that going."


# ------------------------------------------------------------------------------------------------------------------------------------ quiz
_QUIZ = re.compile(_A + r"(?:quiz|test)\s+me\s+(?:on|about)\s+(?:my\s+)?(?P<s>[\w.'&+ -]{2,40}?)(?:\s+notes?)?" + _Z, re.I)
_DEF = re.compile(r"^\s*(?P<x>[A-Za-z][\w .'&+()-]{1,40}?)\s+(?:is|are|means|stands\s+for|refers\s+to)\s+(?P<y>[^.\n]{4,140})", re.I)


def quiz_subject(text: str) -> Optional[str]:
    m = _QUIZ.match((text or "").strip())
    if not m:
        return None
    s = re.sub(r"\s+notes?$", "", m.group("s").strip(), flags=re.I)
    return None if re.fullmatch(r"(?:trivia|something|anything|it|that)", s, re.I) else s


def cards_from(texts: List[str], limit: int = 12) -> List[Tuple[str, str]]:
    """(term, what it is) from lines shaped "X is Y" -- the person's own words, nothing invented."""
    out, seen = [], set()
    for t in texts:
        for line in re.split(r"[\n•]+|(?<=[.!?])\s+", t or ""):
            m = _DEF.match(line.strip(" -*\t"))
            if not m:
                continue
            x, y = m.group("x").strip(), m.group("y").strip().rstrip(",;:")
            if len(x.split()) > 5 or x.lower() in ("it", "this", "that", "there", "he", "she", "they", "which") or x.lower() in seen:
                continue
            seen.add(x.lower())
            out.append((x, y))
            if len(out) >= limit:
                return out
    return out


class Quiz:
    def __init__(self, cards: List[Tuple[str, str]], subject: str, rng: Optional[random.Random] = None):
        self.cards = list(cards)
        (rng or random.Random()).shuffle(self.cards)
        self.subject, self.i, self.right, self.hints = subject, 0, 0, 0
        self.done = False

    def ask(self) -> str:
        x, _y = self.cards[self.i]
        return f"What is {x}?"

    def _good(self, said: str, want: str) -> bool:
        w = {x for x in re.findall(r"[a-z0-9]{3,}", want.lower()) if x not in ("the", "and", "that", "with", "for", "are", "which")}
        s = set(re.findall(r"[a-z0-9]{3,}", said.lower()))
        return bool(w) and len(w & s) >= max(1, round(len(w) * 0.4))

    def hear(self, text: str) -> str:
        t = (text or "").strip().lower()
        x, y = self.cards[self.i]
        if re.match(r"^\W*(?:stop|quit|enough|end\s+(?:the\s+)?quiz|that'?s\s+enough)\W*$", t):
            self.done = True
            return f"Stopped. {self.right} of {self.i} right."
        if re.match(r"^\W*(?:hint|give\s+me\s+a\s+hint|clue)\W*$", t):
            self.hints += 1
            words = y.split()
            return "It starts: “" + " ".join(words[:max(1, len(words) // 3)]) + "...”"
        if re.match(r"^\W*(?:i\s+give\s+up|pass|skip|no\s+idea|i\s+don'?t\s+know|next)\W*$", t):
            line = f"{x} is {y}."
        elif self._good(t, y):
            self.right += 1
            line = "Right."
        else:
            line = f"Not quite -- {x} is {y}."
        self.i += 1
        if self.i >= len(self.cards):
            self.done = True
            return f"{line} That's all {len(self.cards)}: {self.right} right."
        return f"{line} Next: {self.ask()}"


# ------------------------------------------------------------------------------------------------------------------------------------ music
_MUSIC = re.compile(_A + r"(?:play|put\s+on)\s+(?:my\s+|some\s+)?(?:(?P<occ>coding|studying|working|reading|gaming|workout|gym|writing)\s+(?:music|playlist|songs?|tunes)|"
                    r"(?:music|songs?|something)\s+(?:for|while)\s+(?P<occ2>coding|studying|working|reading|gaming|writing)|something\s+i\s+like|"
                    r"(?:my\s+)?favou?rite\s+(?:music|song))" + _Z, re.I)
_OCC = {"studying": "studying", "working": "working", "coding": "coding", "reading": "reading", "gaming": "gaming", "writing": "writing",
        "workout": "exercising", "gym": "exercising"}


def music_ask(text: str) -> Optional[str]:
    """'coding' for "play my coding music", '' for "play something I like", None when it is not this."""
    m = _MUSIC.match((text or "").strip())
    if not m:
        return None
    occ = m.group("occ") or m.group("occ2") or ""
    return _OCC.get(occ.lower(), occ.lower())


def pick_music(items: Dict[str, Dict[str, Any]], occasion: str, occasion_of) -> Optional[str]:
    """The item liked most for the occasion (or at all), from the taste profile -- songs and music, not shows."""
    rows = [e for e in items.values() if float(e.get("score", 0)) > 0 and e.get("kind", "") not in ("movie", "activity", "show", "food")]
    if occasion:
        rows = [e for e in rows if occasion_of(e) == occasion]
    if not rows:
        return None
    return max(rows, key=lambda e: (e.get("kind") == "music", float(e.get("score", 0)), float(e.get("last", 0))))["item"]
