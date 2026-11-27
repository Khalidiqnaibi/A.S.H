"""
src/daemon/meaning.py

Reading what was MEANT, so a request does not have to be worded the way its author expected.

Every feature that ASH answers by rule used to start with a regex written for the phrasings its author thought of. Measured on 28 Nov with 51 natural rewordings
of 17 features, those regexes recognised 12 (23%), and the fallback ("nearest known command", by shared words) one more. The sentence encoder ASH already loads,
given ONE example sentence per feature, put 41 of the 51 on the right feature (80%), with ordinary sentences kept out by a "none" class.

So: the encoder decides WHICH request it is (chooser.py already holds the examples, the none class, the margin and learning from corrections), a small builder turns the
sentence into the canonical phrase that feature's rule already understands, filling the slots (a number, a length of time, a day, a time, a title) with tolerant extractors,
and the existing rule then does the work -- the same path "Did you mean...?" and the nearest-command fallback already use. The regex handlers stay; they stop being the router.

    classify(chooser, text)  ->  Route(label, phrase, score, margin, runner_up)  or None when it is not clear
    SEEDS / NONE / BUILD     the catalogue: examples, ordinary sentences, and how each label becomes its canonical phrase
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .numwords import digits

# ---------------------------------------------------------------------------------------------------------------------------- slots
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS = {"morning": 9, "noon": 12, "afternoon": 15, "evening": 19, "tonight": 20, "night": 20}


def number(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\b", digits((text or "").lower()))
    return int(m.group(1)) if m else None


def duration(text: str) -> Optional[str]:
    """'for a couple of hours' / 'the next two hours' / 'about 45 minutes' / 'an hour' / 'half an hour' -> '2 hours' / '45 minutes' / '1 hour' / '30 minutes'."""
    t = digits((text or "").lower())
    if re.search(r"half\s+an?\s+hour", t):
        return "30 minutes"
    m = re.search(r"(?:a\s+)?couple\s+of\s+(hour|minute)s?", t)
    if m:
        return f"2 {m.group(1)}s"
    m = re.search(r"(?:a\s+)?few\s+(hour|minute)s?", t)
    if m:
        return f"3 {m.group(1)}s"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)s?\b", t)
    if m:
        n = m.group(1)
        return f"{n} {'hour' if m.group(2).startswith('h') else 'minute'}{'' if n == '1' else 's'}"
    if re.search(r"\b(?:an|one|the\s+next|next)\s+hour\b", t) or re.search(r"\ban?\s+hour\b", t):
        return "1 hour"
    return None


def day(text: str) -> Optional[str]:
    t = (text or "").lower()
    for d in _DAYS:
        if re.search(r"\b" + d + r"\b", t):
            return d
    if re.search(r"\btomorrow\b", t):
        return "tomorrow"
    if re.search(r"\b(?:today|tonight|this\s+evening)\b", t):
        return "today"
    return None


def clock(text: str) -> Optional[str]:
    """'at 8' / 'around eight' / 'eight in the evening' / '9:30 am' -> '8 pm' / '9:30 am' (a part of the day settles am/pm)."""
    t = digits((text or "").lower())
    m = re.search(r"(?:\bat\b|\baround\b|\babout\b|\bfor\b|\bby\b)?\s*\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?!\d)", t)
    cand = [x for x in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d%])", t)]
    best = None
    for x in cand:
        before = t[max(0, x.start() - 8):x.start()]
        if x.group(3) or x.group(2) or re.search(r"(?:at|around|about|by)\s*$", before):
            best = x
            break
    if best is None:
        return None
    h, mi, ap = int(best.group(1)), best.group(2), (best.group(3) or "").replace(".", "")
    if h > 12 and not ap:
        return f"{h}:{mi or '00'}"
    if not ap:
        ap = "pm" if re.search(r"evening|night|tonight|dinner|supper|afternoon", t) else "am" if re.search(r"morning|breakfast", t) else ""
    return f"{h}{':' + mi if mi else ''}{' ' + ap if ap else ''}"


def strip_lead(text: str) -> str:
    t = re.sub(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?", "", (text or "").strip(), flags=re.I)
    return t.strip(" .?!")


# ---------------------------------------------------------------------------------------------------------------------------- builders
def _fixed(phrase: str) -> Callable[[str], Optional[str]]:
    return lambda text: phrase


def _volume(text: str) -> Optional[str]:
    n = number(text)
    return f"volume to {n}" if n is not None and 0 <= n <= 100 else ("what's the volume" if re.search(r"\b(?:what|how\s+loud)\b", text.lower()) else None)


def _awake(text: str) -> Optional[str]:
    if re.search(r"\b(?:let|allow)\b.*\bsleep\b|\bsleep\s+again\b", text.lower()):
        return "let it sleep"
    d = duration(text)
    return f"keep my pc awake for {d}" if d else "keep my pc awake for 3 hours"


def _quiet(text: str) -> Optional[str]:
    d = duration(text)
    if d:
        return f"quiet for {d}"
    c = clock(text)
    if c and re.search(r"\b(?:until|till|til)\b", text.lower()):
        return f"quiet until {c}"
    return None


def _event(text: str) -> Optional[str]:
    d, c = day(text), clock(text)
    if not d or not c:
        return None
    t = strip_lead(text)
    t = re.sub(r"\b(?:to|in|on|into)\s+my\s+(?:calendar|diary|schedule|agenda)\b", " ", t, flags=re.I)
    t = re.sub(r"\b(?:put|add|schedule|book|create|enter|note)\s+(?:it|that|this)?\s*(?:in|down)?\b", " ", t, flags=re.I)
    t = re.sub(r"^\s*i(?:'ve|\s+have)\s+got\s+|^\s*i\s+have\s+|^\s*i'm\s+(?:having|meeting)\s+", " ", t, flags=re.I)
    t = re.sub(r"\b(?:this|on|next|for|at|around|about|by)\b", " ", t, flags=re.I)
    t = re.sub(r"\b(?:" + "|".join(_DAYS) + r"|tomorrow|today|tonight|morning|evening|night|afternoon)\b", " ", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b", " ", digits(t))
    t = re.sub(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|o'?clock)\b", " ", t, flags=re.I)
    t = re.sub(r"[,.;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
    if not t or len(t.split()) > 8:
        return None
    return f"put {t} on {d} at {c}"


def _search(text: str) -> Optional[str]:
    t = strip_lead(text)
    t = re.sub(r"^(?:i\s+want\s+to\s+(?:see|find|look\s+up)|find\s+me|show\s+me|get\s+me|search\s+for|look\s+for)\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:google|look\s+up|search)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(?:online|on\s+google|on\s+the\s+web|on\s+the\s+internet)$", "", t, flags=re.I)
    t = t.strip(" .?!")
    return f"look up {t}" if t and len(t.split()) <= 8 else None


def _note(text: str) -> Optional[str]:
    t = strip_lead(text)
    t = re.sub(r"^(?:make\s+a\s+note\s+that|keep\s+in\s+mind\s+(?:that\s+)?|don'?t\s+forget\s+(?:that\s+)?|note\s+(?:down\s+)?(?:that\s+)?|remember\s+(?:that\s+)?)", "", t, flags=re.I)
    t = re.sub(r"[,]?\s*(?:don'?t\s+forget|remember\s+that|keep\s+that\s+in\s+mind)$", "", t, flags=re.I).strip(" ,.")
    return f"remember that {t}" if len(t.split()) >= 2 else None


# ---------------------------------------------------------------------------------------------------------------------------- the catalogue
# label -> examples. Written for MEANING, in different words each, not as the regexes' phrasings; "none" holds ordinary sentences that must not be read as any of them.
SEEDS: Dict[str, List[str]] = {
    "memory": ["what's using my memory", "why is my ram so full", "which program is eating all my memory", "what's hogging the memory", "is my memory almost full"],
    "slow": ["why is my computer slow", "my laptop feels sluggish", "what's making my pc lag", "everything is running slowly right now"],
    "volume": ["volume to 30", "set the volume to forty percent", "make the sound level fifty", "put the volume at twenty", "turn the sound down to 10 percent"],
    "internet": ["is the internet down", "my wifi isn't working", "why can't I get online", "am I connected to the network", "the connection seems dead"],
    "awake": ["keep my pc awake for 2 hours", "don't let the laptop sleep for a couple of hours", "stop the computer from going to sleep for the next hour", "I need it to stay on while I'm away"],
    "calendar": ["put dinner with Omar on Saturday at 8", "add a meeting to my calendar tomorrow at 3", "schedule lunch with the team on Friday at noon", "I have a dentist appointment on Monday at 9, put it in my calendar"],
    "downloads": ["tidy my downloads", "my downloads folder is a mess, sort it out", "clean up the files in downloads", "organise what I've downloaded"],
    "plan": ["plan my day", "what should I do first today", "help me decide what to work on this morning", "what's on my plate today", "where should I start today"],
    "reflect": ["let's reflect on my day", "how did my day go", "did I get anything done today", "what went well today and what didn't"],
    "yesterday": ["what did I do yesterday", "what was I working on yesterday", "what had I been up to last night", "remind me what I did the day before"],
    "update": ["what's your latest update", "what's new with you", "did you change anything recently", "what have you added lately"],
    "away": ["what happened while I was away", "did I miss anything", "anything happen while I was gone", "catch me up"],
    "mic_off": ["turn off the mic", "stop listening for a bit", "mute your microphone", "I don't want you to hear me right now"],
    "mic_on": ["turn on the mic", "start listening again", "unmute your microphone"],
    "quiet": ["quiet for an hour", "leave me alone for the next hour", "don't talk to me for thirty minutes", "give me an hour of silence", "be quiet until three"],
    "smile": ["smile", "give me a big grin", "show me a happy face", "look cheerful"],
    "surprised": ["look surprised", "show me a shocked face", "act amazed"],
    "sad": ["make a sad face", "look sad", "show me you're upset"],
    "wink": ["wink", "give me a wink", "wink at me"],
    "search": ["look up cat pics", "find me pictures of cats online", "google how to boil an egg", "I want to see photos of the moon", "search for the best laptop deals"],
    "note": ["remember that I parked on level 3", "I parked on level three, don't forget", "make a note that the car is on three", "keep in mind my passport is in the top drawer"],
    "skills": ["what have I taught you", "what things can you do that I showed you", "list the skills I taught you", "which lessons did I give you"],
    "joke": ["tell me a joke", "got anything funny", "make me laugh", "say something amusing"],
    "capabilities": ["what can you do", "what are you able to do", "list your abilities", "what are all your features"],
}
NONE: List[str] = [
    "what's the capital of France", "tell me about the roman empire", "how do transformers work in machine learning", "I think the game was great last night",
    "my brother is coming over tomorrow", "can you explain recursion to me", "that movie was too long", "I'm tired today", "who won the match yesterday",
    "what is a black hole", "the weather in my city is nice", "how do I learn python quickly", "write me a poem about the sea", "I like jazz music a lot",
    "do you think AI will replace programmers", "what does ephemeral mean", "she said she would call me back", "the meeting went really well",
    "I need to buy milk and eggs", "my sister lives in Amman", "this keyboard is so loud", "it was a long day at work", "the sound in that film was amazing",
]
BUILD: Dict[str, Callable[[str], Optional[str]]] = {
    "memory": _fixed("what's using my memory"), "slow": _fixed("why is my computer slow"), "volume": _volume, "internet": _fixed("is the internet down"),
    "awake": _awake, "calendar": _event, "downloads": _fixed("tidy my downloads"), "plan": _fixed("plan my day"), "reflect": _fixed("let's reflect on my day"),
    "yesterday": _fixed("what did I do yesterday"), "update": _fixed("what's your latest update"), "away": _fixed("what happened while I was away"),
    "mic_off": _fixed("turn off the mic"), "mic_on": _fixed("turn on the mic"), "quiet": _quiet, "smile": _fixed("smile"), "surprised": _fixed("look surprised"),
    "sad": _fixed("make a sad face"), "wink": _fixed("wink"), "search": _search, "note": _note, "skills": _fixed("what have I taught you"), "joke": _fixed("tell me a joke"),
    "capabilities": _fixed("what can you do"),
}
# Things that change something are confirmed when the match is not certain; these only look or say.
EFFECT = {"awake", "calendar", "downloads", "mic_off", "mic_on", "quiet", "volume", "note"}
# Features about the same stretch of time are easily confused: close scores between them are asked about, not guessed.
SIBLINGS = [{"plan", "reflect", "yesterday", "away"}, {"memory", "slow"}, {"mic_off", "quiet"}, {"smile", "surprised", "sad", "wink"}]


@dataclass
class Route:
    label: str
    phrase: str
    score: float
    margin: float
    runner_up: Optional[str] = None
    runner_phrase: Optional[str] = None
    sure: bool = True


def make_seeds() -> Dict[str, List[str]]:
    s = {k: list(v) for k, v in SEEDS.items()}
    s["none"] = list(NONE)
    return s


def classify(chooser, text: str) -> Optional[Route]:
    """The feature `text` means, as the canonical phrase its rule understands; None when it is not clear (the caller falls through as before)."""
    if chooser is None or not getattr(chooser, "ready", False):
        return None
    v = chooser.classify(text)
    if not v.label or v.label == "none" or v.label not in BUILD:
        # Close between two features about the same thing: say so and let the caller ask which.
        ranked = sorted(((k, s) for k, s in v.scores.items() if k != "none"), key=lambda kv: -kv[1])
        if len(ranked) >= 2 and ranked[0][1] >= chooser.accept + 0.1 and ranked[0][1] - v.scores.get("none", 0.0) >= 0.03:
            a, b = ranked[0][0], ranked[1][0]
            if any(a in g and b in g for g in SIBLINGS) and ranked[0][1] - ranked[1][1] < chooser.margin and a in BUILD and b in BUILD:
                pa, pb = BUILD[a](text), BUILD[b](text)
                if pa and pb:
                    return Route(a, pa, ranked[0][1], ranked[0][1] - ranked[1][1], b, pb, sure=False)
        return None
    phrase = BUILD[v.label](text)
    if not phrase:
        return None
    ranked = sorted(((k, s) for k, s in v.scores.items() if k not in ("none", v.label)), key=lambda kv: -kv[1])
    runner = ranked[0][0] if ranked else None
    sure = True
    if runner is not None and any(v.label in g and runner in g for g in SIBLINGS) and v.margin < 2 * chooser.margin:
        sure = False
    if v.label in EFFECT and v.score < chooser.accept + 0.08:
        sure = False
    if not sure and v.score < chooser.accept + 0.1:
        return None                                  # a weak match is not worth interrupting for: an ordinary line goes on as it always did
    return Route(v.label, phrase, v.score, v.margin, runner, BUILD[runner](text) if runner and runner in BUILD else None, sure)
