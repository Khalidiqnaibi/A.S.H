"""
src/daemon/yesno.py

Yes and no, said the way people say them.

Every question ASH asks that wants a yes -- "Shut ASH down completely?", "Close Word?", "Go ahead with that plan?",
"Want a game?" -- used to carry its own short list of accepted words: yes, yeah, yep, sure, okay... so "ye", "yea", "yeh",
"yess", "mhm", "k", "alright then", "do it", "please", "confirm", "absolutely" or "yes, shut it down" was not an answer, and
the question was lost (and the line went to the model, which explained at length what 'ye' might mean).

One reading, used by all of them:

    is_yes(text)   a plain agreement
    is_no(text)    a plain refusal
    is_yes(text, strong=True)   only a clear one -- for things that cannot be taken back (sending a mail or a text): yes,
                   yeah, yep, sure, do it, go ahead, send it. Not "ok", "mhm", "fine", which can mean "I heard you".

A line that agrees and then says something else ("yes but not now") is neither. Nothing here calls a model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_STRONG_YES = {"yes", "yeah", "yea", "yep", "yup", "yeh", "ye", "y", "ya", "yah", "yas", "aye", "sure", "definitely", "absolutely",
               "certainly", "affirmative", "confirm", "confirmed", "proceed", "please", "doit", "goahead", "sendit", "gofor",
               "ofcourse", "yesplease", "surething", "byallmeans"}
_WEAK_YES = {"ok", "okay", "k", "kk", "okey", "okie", "alright", "aight", "fine", "right", "correct", "mhm", "mmhm", "uhhuh",
             "uhuh", "sounds", "whynot", "gofor", "goon", "thatsfine", "yesok", "okey", "okie"}
_NO_WORDS = {"no", "n", "nope", "nah", "nay", "negative", "cancel", "stop", "dont", "wait", "nevermind", "never", "leave",
             "forget", "abort", "hold", "not", "nothanks", "nono", "noo"}
_BUT = re.compile(r"\b(?:but|except|unless|however|though|actually|wait|not\s+(?:now|yet|that)|later|instead)\b", re.I)
_FILLER = {"then", "please", "thanks", "thank", "you", "do", "it", "go", "ahead", "and", "ash", "hey", "just", "now", "yes",
           "that", "one", "sounds", "good", "great", "close", "shut", "down", "send", "run", "start", "open", "turn", "off", "on",
           "sure", "okay", "ok", "course", "of", "why", "not", "for", "on", "i", "do", "think", "so", "is", "fine", "the"}


def _norm(text: str) -> str:
    t = str(text or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                      # yesssss -> yess
    return re.sub(r"\s+", " ", t).strip()


def _squash(word: str) -> str:
    return re.sub(r"(.)\1+", r"\1", word)


def _is(word: str, targets) -> Optional[str]:
    """The word, or the word with doubled letters undone (yess, yeahh, okayy, noo, suree)."""
    if word in targets:
        return word
    sq = _squash(word)
    if sq in targets:
        return sq
    return {_squash(t): t for t in targets}.get(sq)


def _tokens(text: str):
    t = _norm(text)
    if not t or len(t.split()) > 8:
        return []
    t = re.sub(r"^(?:hey )?ash ", "", t)
    t = re.sub(r" ash$", "", t)
    t = re.sub(r"\bgo ahead\b", "goahead", t)
    t = re.sub(r"\bdo it\b", "doit", t)
    t = re.sub(r"\bsend it\b", "sendit", t)
    t = re.sub(r"\bof course\b", "ofcourse", t)
    t = re.sub(r"\bwhy not\b", "whynot", t)
    t = re.sub(r"\bsure thing\b", "surething", t)
    t = re.sub(r"\buh ?huh\b", "uhhuh", t)
    t = re.sub(r"\bm+ ?hm+\b", "mhm", t)
    t = re.sub(r"\bnever mind\b", "nevermind", t)
    t = re.sub(r"\bno thanks?\b", "nothanks", t)
    t = re.sub(r"\bdon'?t\b", "dont", t)
    return t.split()


def is_yes(text: str, strong: bool = False) -> bool:
    toks = _tokens(text)
    if not toks or _BUT.search(_norm(text)):
        return False
    if any(t in _NO_WORDS for t in toks):
        return False
    first = toks[0]
    word = _is(first, _STRONG_YES)
    weak = False
    if word is None:
        word = _is(first, _WEAK_YES)
        weak = word is not None
    if word is None:
        return False
    if strong and weak:
        return False
    rest = toks[1:]
    # whatever follows must be small talk about the same thing ("yes, shut it down", "okay do it"), not a new sentence
    return all(r in _FILLER or _is(r, _STRONG_YES | _WEAK_YES) for r in rest) or len(rest) <= 1


def is_no(text: str) -> bool:
    toks = _tokens(text)
    if not toks:
        return False
    first = toks[0]
    if first in ("not",):
        return len(toks) <= 3 and (len(toks) == 1 or toks[1] in ("now", "yet", "really", "that", "this"))
    word = _is(first, _NO_WORDS)
    if word is None:
        return False
    if word in ("never", "leave", "forget", "hold", "wait", "stop"):
        return len(toks) <= 3
    return len(toks) <= 4 or word in ("no", "nope", "nah", "n")


class _M:
    """A drop-in for a compiled pattern's .match()."""

    def __init__(self, fn, rx=None):
        self.fn, self.rx = fn, rx

    def match(self, text: Any, *a):
        t = str(text or "")
        if self.rx is not None:
            m = self.rx.match(t)
            if m:
                return m
        return _Hit(t) if self.fn(t) else None

    def search(self, text: Any, *a):
        return self.match(text)


class _Hit:
    def __init__(self, t: str):
        self.t = t

    def group(self, *a):
        return self.t

    def __bool__(self):
        return True


YES = _M(is_yes)
NO = _M(is_no)


def yes_or(rx) -> _M:
    """A pattern with plain yes added: what it matched before still matches, and so does the rest of what is yes."""
    return _M(is_yes, rx)


def no_or(rx) -> _M:
    return _M(is_no, rx)
