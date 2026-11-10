"""
src/daemon/taste.py

What people like, and what they can't stand -- the start of a profile.

ASH knew who people were ("Omar is your cousin") and nothing about them. Said
aloud, "I love Interstellar", "I can't stand horror films", "Ali likes jazz",
"I listen to lo-fi when I'm coding" went to the model and were gone with the
conversation. This keeps them, the way the rest of what ASH is told is kept:

    * in Knowledge (`kb.tastes`), so it is saved, undone ("undo that"),
      forgotten ("forget Interstellar", "forget Ali") and mirrored into entity
      memory by the code that already does all of that for people and terms;
    * learned from the same utterances wondering reads, by the same rules
      about who is speaking (a guest's voice never writes the profile);
    * answered by rule when asked ("what do I like?", "what does Ali hate?"),
      never made up by the model;
    * handed to the model, as background, only when the question is about
      choosing something -- what to watch, listen to, eat.

Behaviour is a second, weaker source: what the now-playing sensor
(senses/nowplaying.py) sees played to the end, replayed or skipped moves a
score in small steps (`watch()`), capped below "love" and never overriding what
the person said. Nothing is learned from the pixels of the screen.

A taste is {item, kind, score, n, by, when}. `score` runs from -1 (hates) to 1
(loves). Saying it again strengthens it; saying the opposite replaces it,
because people change their minds and the latest word is the truer one.
`when` counts what was going on -- "sad", "coding" -- so "Ali likes this when
he's sad" is kept as exactly that.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .wondering import NOT_PERSON, ROLE_CANON, norm

logger = logging.getLogger("ash.daemon.taste")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


@dataclass
class TasteConfig:
    enabled: bool = True
    learn_from_others: bool = True      # "Ali likes jazz" -- on the user's word
    mirror: bool = True                 # short likes/dislikes on the entity
    in_answers: bool = True             # background for choosing something
    from_playing: bool = True           # what is played, finished or skipped (senses/nowplaying.py)
    listen_s: float = 90.0              # a song played this long counts as listened to
    watch_s: float = 600.0              # a video or episode watched this long counts as watched
    max_items_per_subject: int = 200
    like_at: float = 0.30               # |score| below this is not worth saying
    strong_at: float = 0.75             # "love" and "hate", not "like"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TasteConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


# ----------------------------------------------------------------------
# Reading a sentence
# ----------------------------------------------------------------------
_ROLES = "|".join(sorted(ROLE_CANON, key=len, reverse=True))
_LEAD = (r"^\W*(?:(?:and|but|so|well|honestly|actually|yeah|yes|ok|okay|oh|tbh|also|man|dude|like|"
         r"i mean|you know)\b[,\s]*)*")
_ME = r"(?P<me>i|i'?m|i am|i've|ive|i have|we|we're|we all)"
_NAME = r"(?-i:(?P<name>[A-Z][a-z]{1,19}(?:\s[A-Z][a-z]{1,19})?))"
_HE = r"(?P<pro>he|she)"
_SUBJ = rf"(?:{_ME}|(?:my\s+(?:{_ROLES})\s+)?{_NAME}(?:'s|’s)?|{_HE}(?:'s|’s)?)"
_ADV = (r"(?:(?:really|absolutely|totally|just|honestly|actually|still|always|truly|genuinely|literally|"
        r"completely|seriously|so|super|kinda|kind of|sort of|a bit|a little|somewhat|usually|often|"
        r"quite|pretty|definitely)\s+)*")
_BE = r"(?:(?:am|is|are|was|were|be)\s+)?"
_END = r"(?=\s*(?:[.!?;]|$|\s+(?:because|but|though|although|since|cause|cuz|when|while|whenever|if|so)\b))"

# (regex of the verb phrase, strength). Strengths are what the words claim.
_POS_STRONG = (r"(?:loves?|loved|adores?|adored|obsessed\s+with|can'?t\s+get\s+enough\s+of|"
               r"hooked\s+on|(?:a\s+)?(?:huge|big|massive)\s+fan\s+of|crazy\s+about|"
               r"in\s+love\s+with|addicted\s+to)")
_POS = r"(?:likes?|liked|enjoys?|enjoyed|into|(?:a\s+)?fan\s+of|digs?|a\s+fan\s+of)"
_NEG = (r"(?:(?:do|does|did)\s*(?:n'?t|not)\s+(?:really\s+)?(?:like|enjoy|care\s+for)|dislikes?|disliked|"
        r"(?:not|n'?t)\s+(?:a\s+)?(?:big\s+)?fan\s+of|not\s+(?:really\s+)?into|not\s+a\s+fan\s+of|"
        r"never\s+(?:really\s+)?liked)")
_NEG_STRONG = (r"(?:hates?|hated|can'?t\s+stand|cannot\s+stand|couldn'?t\s+stand|detests?|detested|"
               r"loathes?|loathed|despises?|despised|(?:so\s+)?sick\s+of)")

_VERBS: List[Tuple[str, float]] = [(_NEG_STRONG, -0.9), (_NEG, -0.55), (_POS_STRONG, 0.9), (_POS, 0.55)]
_MAIN = [(re.compile(_LEAD + _SUBJ + r"\s*" + _BE + _ADV + r"(?P<v>" + v + r")\s+(?P<obj>.+?)" + _END, re.I), s)
         for v, s in _VERBS]

# "my favourite movie is Interstellar"
_FAVOURITE = re.compile(
    _LEAD + r"(?:(?:my)|" + _NAME + r"(?:'s|’s)|(?P<pro>his|her))\s+(?:all[- ]time\s+|current\s+)?"
    r"(?:favou?rite|fave)\s+(?P<k>[a-z][a-z \-]{1,24}?)\s+(?:is|was|are|has been)\s+(?P<obj>.+?)" + _END, re.I)

# "Interstellar is amazing" / "the new Dune was terrible": a taste about a named thing.
_GOOD = (r"(?:amazing|awesome|incredible|brilliant|fantastic|excellent|great|so\s+good|really\s+good|the\s+best|"
         r"a\s+masterpiece|perfect|fire|a\s+banger|a\s+bop|underrated|addictive|beautiful|a\s+classic)")
_BAD = (r"(?:awful|terrible|horrible|boring|trash|garbage|the\s+worst|so\s+bad|really\s+bad|overrated|"
        r"a\s+waste\s+of\s+time|unwatchable|unplayable|annoying|mid|disappointing|a\s+mess|cringe)")
_OPINION = [(re.compile(_LEAD + r"(?P<obj>(?-i:[A-Z][\w'\-:&]*(?:\s+[A-Za-z][\w'\-:&]*){0,3}))\s+(?:is|was|are|were)\s+"
                        + r"(?:(?:really|so|absolutely|totally|just|honestly|actually|pretty|super|incredibly)\s+)?"
                        + v + r"\s*(?:[.!?;]|$|\s+(?:but|though|because|cause)\b)", re.I), s)
            for v, s in ((_BAD, -0.6), (_GOOD, 0.6))]

# What was done, said as an aside: "I skipped X", "I binged X".
_ACT_NEG = re.compile(
    _LEAD + _ME + r"\s+(?:just\s+|ended\s+up\s+)?(?:skipped|couldn'?t\s+finish|could\s+not\s+finish|gave\s+up\s+on|"
    r"stopped\s+watching|quit\s+watching|turned\s+off|walked\s+out\s+of|dropped|bailed\s+on|"
    r"didn'?t\s+finish|never\s+finished)\s+(?P<obj>.+?)" + _END, re.I)
_ACT_POS = re.compile(
    _LEAD + _ME + r"\s+(?:just\s+|totally\s+|ended\s+up\s+)?(?:binged|binge[- ]?watched|rewatched|re-watched|"
    r"rewatching|replayed|replaying)\s+(?P<obj>.+?)" + _END, re.I)
# "I listen to lo-fi while I'm coding", "Ali watches Jaws when he's sad"
_HABIT = re.compile(
    _LEAD + _SUBJ + r"\s*" + _ADV + r"(?P<v>listens?\s+to|listening\s+to|watch(?:es|ing)?|plays?|playing|"
    r"puts?\s+on|put\s+on|reads?|reading|eats?|eating|orders?)\s+(?P<obj>.+?)\s+"
    r"(?:when|while|whenever|every\s+time)\b", re.I)

# What was going on, from a short whitelist: free text here is how "when he's
# sad" turns into "when the build fails and my coffee went cold".
_MOODS = frozenset("""sad happy stressed tired bored angry anxious lonely excited nervous sick upset down low
    relaxed calm focused motivated busy sleepy hungry depressed down heartbroken""".split())
_ACTIVITIES = frozenset("""coding working studying driving cooking running cleaning gaming reading sleeping
    writing exercising training walking commuting relaxing designing programming debugging""".split())
_WHEN = re.compile(r"\b(?:when|while|whenever|every\s+time)\s+(?:(?:i|he|she|they|we)(?:'?s|'?re|'?m|\s+(?:am|is|are))?\s+)?"
                   r"(?:feeling\s+|getting\s+|being\s+)?(?P<c>[a-z]{3,14})\b", re.I)

_QUESTION = re.compile(r"^\W*(?:do|does|did|would|could|should|is|are|was|were|will|can|what|why|how|who|where|"
                       r"which|when|if|maybe|suppose|imagine|wish|what if)\b", re.I)
_HEDGE = re.compile(r"\b(?:if|wish|supposedly|apparently|they say|people say|he said|she said|"
                    r"would you|do you|did you|are you)\b", re.I)

# Words that cannot be what someone likes.
_NOT_ITEM = frozenset("""it that this these those them him her you me us myself yourself himself herself itself
    everything anything something nothing someone somebody anyone everyone people stuff things thing one ones
    ash a.s.h when how what why who where which that i i'm im ive he she they we being doing having getting
    waking going trying to be then there here so very much lot more less less""".split())
_ARTICLES = re.compile(r"^(?:the|a|an|my|his|her|their|our|your|some|all|any|every|these|those|to|just)\s+", re.I)
_TRAIL = re.compile(
    r"\s*(?:a\s+lot|so\s+much|very\s+much|too|lately|these\s+days|nowadays|right\s+now|already|though|honestly|"
    r"to\s+be\s+honest|as\s+well|also|at\s+all|ever|anymore|any\s+more|now|again|forever|"
    r"more\s+than\s+anything|of\s+all\s+time|(?:last|this|over\s+the)\s+(?:night|week|weekend|month|year|summer|winter)|"
    r"yesterday|today|tonight|this\s+morning|on\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day)\s*$", re.I)
_FUSED = frozenset({"rock and roll", "salt and vinegar", "tom and jerry", "mac and cheese", "fish and chips",
                    "pride and prejudice", "black and white", "drum and bass", "r and b", "rhythm and blues",
                    "bread and butter", "law and order", "beauty and the beast", "lord of the rings and",
                    "lock stock and two smoking barrels", "peanut butter and jelly", "salt and pepper",
                    "rick and morty", "ben and jerry", "simon and garfunkel", "hall and oates"})

_KIND_WORDS: List[Tuple[str, "re.Pattern"]] = [(k, re.compile(p, re.I)) for k, p in (
    ("music", r"\b(?:songs?|albums?|tracks?|bands?|artists?|singers?|rappers?|music|playlists?|genres?|lo-?fi|"
              r"jazz|rock|metal|rap|hip[- ]?hop|pop|classical|techno|edm|k-?pop|listen(?:s|ed|ing)?|album)\b"),
    ("movie", r"\b(?:films?|movies?|cinema|documentar(?:y|ies)|horror|thrillers?|rom-?coms?)\b"),
    ("show", r"\b(?:shows?|series|anime|cartoons?|episodes?|seasons?|sitcoms?|netflix|manga)\b"),
    ("game", r"\b(?:games?|gaming|played|playing|fps|rpg|console|steam)\b"),
    ("food", r"\b(?:food|eat(?:s|ing)?|ate|dish(?:es)?|pizza|burgers?|pasta|sushi|cuisine|restaurants?|coffee|tea|"
             r"chocolate|spicy|drink(?:s|ing)?|cook(?:s|ing)?|snacks?|dessert|breakfast|dinner|lunch)\b"),
    ("book", r"\b(?:books?|novels?|authors?|reads?|reading|comics?|poems?|poetry)\b"),
    ("app", r"\b(?:apps?|software|editor|ide|browser|tool|extension|theme)\b"),
    ("watch", r"\b(?:watch(?:es|ed|ing)?)\b"),
)]
_GERUND = re.compile(r"^(?:[a-z]+ing)\b", re.I)


def _title_item(item: str) -> str:
    """Keep the way it was said, but tidy a shouted or all-lower phrase."""
    item = item.strip(" \t\"'“”‘’.,!?;:-")
    if item.isupper() and len(item) > 4:
        return item.title()
    return item


def _split_items(obj: str) -> List[str]:
    t = obj.strip()
    if not t:
        return []
    if t.lower() in _FUSED:
        return [t]
    pieces = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s+&\s+", t)
    # "rock and roll" style: if a fused phrase is in there, keep it whole.
    low = t.lower()
    for f in _FUSED:
        if f in low:
            return [t]
    return [p for p in pieces if p.strip()]


_GENERIC_NOUNS = frozenset("""movie film show song game book album episode series track one thing place app
    time day week night guy girl person idea way part bit lot kind sort type""".split())


def _clean_item(raw: str) -> str:
    """The thing liked, or "" if what follows the verb is not a thing."""
    t = re.sub(r"\s+", " ", (raw or "").strip())
    t = t.strip(" \t\"'“”‘’.,!?;:-")
    if re.match(r"^your\b", t, re.I):
        return ""                                   # said to ASH, not about the world
    keep_the = re.match(r"^The\s+(?-i:[A-Z])", t) is not None      # "The Boys", not "the movie"
    deictic = re.match(r"^(?:the|that|this|these|those)\s+(\w+)$", t, re.I)
    if deictic and deictic.group(1).lower() in _GENERIC_NOUNS:
        return ""                                   # "the movie" is not a movie
    for _ in range(3):
        t2 = _TRAIL.sub("", t).strip(" ,.")
        if not (keep_the and t2.lower().startswith("the ")):
            t2 = _ARTICLES.sub("", t2).strip()
        if t2 == t:
            break
        t = t2
    if not t or len(t) > 44 or len(t.split()) > 5 or not re.search(r"[A-Za-z]{2}", t):
        return ""
    first = t.lower().split()[0]
    if t.lower() in _NOT_ITEM or first in {"when", "how", "what", "why", "who", "where", "which", "that", "if", "i",
                                           "i'm", "im", "i've", "ive", "it", "you", "being", "having", "waking"}:
        return ""
    if re.search(r"\b(?:i|i'm|i've|you|we|he|she|they)\s+(?:am|are|is|was|were|have|had|do|did|can|will|would)\b",
                 t, re.I):
        return ""
    return _title_item(t)


def _kind_of(item: str, sentence: str, verb: str = "") -> str:
    low_item, low_sent = item.lower(), sentence.lower()
    for text in (low_item, low_sent):
        for kind, rx in _KIND_WORDS:
            if rx.search(text):
                if kind == "watch":
                    return "movie" if "movie" in low_sent or "film" in low_sent else "show" if re.search(
                        r"\b(?:show|series|episode|season|anime)\b", low_sent) else "watch"
                return kind
    if _GERUND.match(low_item) or low_item.startswith("to "):
        return "activity"
    return "other"


def _intensity(sentence: str, base: float) -> float:
    """Nudge a strength by how it was said, keeping its sign and its band."""
    s = sentence.lower()
    sign = 1.0 if base > 0 else -1.0
    mag = abs(base)
    if re.search(r"\b(?:kind of|sort of|kinda|a bit|a little|somewhat|meh|not really|pretty)\b", s):
        mag = max(0.3, mag - 0.25)
    elif re.search(r"\b(?:really|absolutely|totally|so much|a lot|literally|completely|seriously|super|"
                   r"incredibly|truly|honestly|always)\b", s):
        mag = min(0.95, mag + 0.15)
    return round(sign * mag, 2)


@dataclass
class Found:
    """One taste read out of one sentence."""
    subject: str            # "" is the speaker; otherwise a name, or "he"/"she"
    item: str
    kind: str
    strength: float
    when: str = ""
    by: str = "said"        # said | opinion | habit | action
    pronoun: str = ""       # he / she, resolved by the caller


def _when_of(sentence: str) -> str:
    for m in _WHEN.finditer(sentence):
        c = m.group("c").lower()
        if c in _MOODS or c in _ACTIVITIES:
            return c
    return ""


def _is_speaker(m: "re.Match") -> bool:
    return bool(m.groupdict().get("me"))


def read(sentence: str) -> List[Found]:
    """Every taste this one sentence states. Pure: no memory, no names known."""
    s = (sentence or "").strip()
    if not s or len(s) > 280 or "?" in s or _QUESTION.match(s) or _HEDGE.search(s):
        return []
    # "I love Interstellar and I hate Tenet" is two statements.
    clauses = re.split(r"\s*;\s*|,?\s+(?:and|but)\s+(?=(?:i|i'm|i've|he|she)\b)", s, flags=re.I)
    if len(clauses) > 1:
        found: List[Found] = []
        for c in clauses:
            found.extend(_read_clause(c.strip()))
        return found
    return _read_clause(s)


def _read_clause(s: str) -> List[Found]:
    out: List[Found] = []

    def add(m, obj, strength, by="said", verb=""):
        who = "" if _is_speaker(m) else (m.groupdict().get("name") or m.groupdict().get("pro") or "")
        if who.lower() in NOT_PERSON or (who and who.split()[0].lower() in NOT_PERSON):
            return
        when = _when_of(s)
        for raw in _split_items(obj):
            item = _clean_item(raw)
            if item:
                kind = "activity" if raw.strip().lower().startswith("to ") else _kind_of(item, s, verb)
                out.append(Found(subject=who if who and who.lower() not in ("he", "she") else "",
                                 item=item, kind=kind,
                                 strength=_intensity(s, strength), when=when, by=by,
                                 pronoun=who.lower() if who.lower() in ("he", "she") else ""))

    # Favourite: the clearest thing anyone says about taste.
    m = _FAVOURITE.search(s)
    if m:
        kindword = (m.group("k") or "").lower()
        who = (m.groupdict().get("name") or "")
        pro = (m.groupdict().get("pro") or "").lower()
        for raw in _split_items(m.group("obj")):
            item = _clean_item(raw)
            if item:
                kind = _kind_of(item, kindword + " " + s)
                out.append(Found(subject=who, item=item, kind=kind, strength=0.95,
                                 by="said", pronoun={"his": "he", "her": "she"}.get(pro, "")))
        if out:
            return out

    # A habit with its occasion: the kind of thing a profile is for.
    m = _HABIT.search(s)
    if m and _when_of(s):
        add(m, m.group("obj"), 0.5, by="habit", verb=m.group("v"))
        if out:
            for f in out:
                f.kind = _kind_of(f.item, s, m.group("v")) if f.kind == "other" else f.kind
            return out

    for rx, strength in _MAIN:
        m = rx.search(s)
        if m:
            # "I don't like X" must not also match as the "like" it contains.
            add(m, m.group("obj"), strength, verb=m.group("v"))
            if out:
                return out
    for rx, strength in ((_ACT_NEG, -0.5), (_ACT_POS, 0.6)):
        m = rx.search(s)
        if m:
            add(m, m.group("obj"), strength, by="action")
            if out:
                return out
    for rx, strength in _OPINION:
        m = rx.search(s)
        if m:
            item = _clean_item(m.group("obj"))
            if item and item.lower() not in NOT_PERSON and not item.lower().startswith(
                    ("the weather", "this ", "that ", "it ")):
                out.append(Found(subject="", item=item, kind=_kind_of(item, s),
                                 strength=_intensity(s, strength), by="opinion"))
                return out
    return out


def mirror(kb, keys: Iterable[str], user: str = "", like_at: float = 0.30) -> None:
    """Write each subject's strongest likes and dislikes onto their entity.

    Shared with forget.py: forgetting "Interstellar" has to take it off the
    entity too, or entity memory keeps saying what the profile no longer does.
    """
    if getattr(kb, "entities", None) is None:
        return
    for key in set(keys):
        with kb._lock:
            subj = dict(kb.tastes.get(key) or {})
            rows = [dict(v) for v in (subj.get("items") or {}).values()]
        name, is_user = subj.get("name", key), bool(subj.get("user"))
        if is_user and (not user and name.strip().lower() in ("", "you", "user")):
            continue                      # an unnamed user would become an entity called "you"
        rows.sort(key=lambda r: (-abs(float(r.get("score", 0))), -float(r.get("last", 0))))
        likes = ", ".join((f"{r['item']} ({_while(occasion_of(r))})" if occasion_of(r) else r["item"])
                          for r in rows if float(r.get("score", 0)) >= like_at)[:200]
        dis = ", ".join(r["item"] for r in [r for r in rows if float(r.get("score", 0)) <= -like_at][:4])[:160]
        if not subj and not likes and not dis:
            continue
        try:
            kb._mirror("person", name, {"likes": likes, "dislikes": dis})
        except Exception:
            logger.debug("Could not mirror tastes into entity memory", exc_info=True)



# ----------------------------------------------------------------------
# What is kept
# ----------------------------------------------------------------------
_LOVE_WORDS = {1: ("loves", "likes"), -1: ("hates", "dislikes")}


def _band(score: float, strong: float = 0.75) -> str:
    return "loves" if score >= strong else "likes" if score > 0 else "hates" if score <= -strong else "dislikes"


LINK_AT = 3          # times seen with the same activity before it is called a habit
LINK_SHARE = 0.5     # ...and at least this share of the plays


def occasion_of(entry: Dict[str, Any]) -> str:
    """What this is liked "when" or "while": what the person said, else what has
    been seen often enough. One sighting is a coincidence; three, most of the
    plays, is "listens to this while coding"."""
    said = entry.get("when") or {}
    if said:
        return max(said, key=lambda k: said[k])
    seen = entry.get("seen_when") or {}
    if seen:
        top = max(seen, key=lambda k: seen[k])
        if seen[top] >= LINK_AT and seen[top] >= LINK_SHARE * max(1, int(entry.get("n", 1))):
            return top
    return ""


def _while(act: str) -> str:
    return f"while {act}" if act.endswith("ing") else f"when feeling {act}"


class Taste:
    """The profile: what each person likes and dislikes, in `kb.tastes`."""

    def __init__(self, kb, user: str = "", cfg: Optional[TasteConfig] = None,
                 clock: Callable[[], float] = time.time, scrub: Optional[Callable[[str], str]] = None):
        self.kb = kb
        self.user = (user or "").strip()
        self.cfg = cfg or TasteConfig()
        self.clock = clock
        self.scrub = scrub
        self.stats = {"heard": 0, "learned": 0, "ignored": 0}
        self.last: List[Tuple[str, str]] = []           # (subject key, item key) of the last learning
        self.last_undo: Optional[Callable[[], None]] = None   # reverses the last watch() that said something

    # -- who ---------------------------------------------------------------
    @property
    def user_key(self) -> str:
        return norm(self.user) or "user"

    def _subject(self, f: Found, people: Iterable[str]) -> Optional[Tuple[str, str, bool]]:
        """(key, display name, is the user) for a taste, or None if unplaceable."""
        if f.pronoun:
            last = [p for p in people if p]
            if not last:
                return None                             # "he loves it" with no "he": say nothing
            name = last[-1]
            return norm(name), name, False
        if not f.subject:
            return self.user_key, self.user or "you", True
        if self.user and norm(f.subject) == norm(self.user):
            return self.user_key, self.user, True
        if not self.cfg.learn_from_others or self._not_a_person(f.subject):
            return None
        return norm(f.subject), f.subject, False

    def _not_a_person(self, name: str) -> bool:
        """Entity memory already knows "Netflix" and "Python" are not people,
        and a capitalised word before "likes" is otherwise anyone."""
        n = norm(name)
        try:
            if self.kb.person_seen(name):
                return False
            ents = getattr(self.kb, "entities", None)
            if ents is None:
                return False
            for e in ents.store.all():
                names = [norm(str(v)) for v in (e.primary_identifiers.get("name"),
                                                 (e.attributes or {}).get("canonical_name")) if v]
                if n in names and getattr(e, "entity_type", "") != "person":
                    return True
        except Exception:
            logger.debug("Could not check entity memory for a person", exc_info=True)
        return False

    # -- listening ----------------------------------------------------------
    def observe(self, text: str, source: str = "console", now: Optional[float] = None,
                people: Iterable[str] = (), guest: bool = False) -> List[str]:
        """Learn the tastes a message states. Returns what was learned, as
        sentences ("The user loves Interstellar."), like wondering does."""
        now = self.clock() if now is None else now
        if not self.cfg.enabled or guest or not text:
            return []
        people = list(people)
        learned: List[str] = []
        self.last = []
        for sent in re.split(r"(?<=[.!?])\s+|\n+", str(text)):
            self.stats["heard"] += 1
            for f in read(sent):
                who = self._subject(f, people)
                if who is None:
                    self.stats["ignored"] += 1
                    continue
                key, name, is_user = who
                if self.scrub is not None:
                    f.item = self.scrub(f.item).strip()
                    if not f.item or "[" in f.item:
                        continue
                line = self.record(key, name, is_user, f, now)
                if line:
                    learned.append(line)
        if learned:
            self._mirror(set(k for k, _ in self.last))
        return learned

    def record(self, key: str, name: str, is_user: bool, f: Found, now: float) -> Optional[str]:
        """Write one taste. Returns the sentence for it, or None if nothing changed."""
        ik = norm(f.item)
        if not ik:
            return None
        with self.kb._lock:
            subj = self.kb.tastes.setdefault(key, {"name": name, "user": bool(is_user), "items": {}})
            subj["name"] = name or subj.get("name", "")
            subj["user"] = bool(is_user) or bool(subj.get("user"))
            items = subj.setdefault("items", {})
            cur = items.get(ik)
            if cur is None:
                if len(items) >= self.cfg.max_items_per_subject:
                    worst = min(items, key=lambda k: (abs(items[k].get("score", 0)), items[k].get("last", 0)))
                    del items[worst]
                cur = items[ik] = {"item": f.item, "kind": f.kind, "score": f.strength, "n": 1,
                                   "first": now, "last": now, "by": f.by, "flips": 0, "when": {}}
                changed = True
            else:
                old = float(cur.get("score", 0.0))
                if old * f.strength < 0:                              # changed their mind
                    cur["score"], cur["flips"] = f.strength, int(cur.get("flips", 0)) + 1
                    changed = True
                else:
                    sign = 1.0 if f.strength > 0 else -1.0
                    new = sign * min(0.95, max(abs(old), abs(f.strength)) + 0.08)
                    changed = abs(new - old) > 1e-6 or cur.get("kind") == "other" and f.kind != "other"
                    cur["score"] = round(new, 3)
                cur["n"] = int(cur.get("n", 0)) + 1
                cur["last"] = now
                if cur.get("kind", "other") == "other" and f.kind != "other":
                    cur["kind"] = f.kind
                if f.by == "said" or cur.get("by") != "said":
                    cur["by"] = f.by if cur.get("by") != "said" else "said"
            if f.when:
                w = cur.setdefault("when", {})
                w[f.when] = int(w.get(f.when, 0)) + 1
            self.kb.save()
        self.stats["learned"] += 1
        self.last.append((key, ik))
        return self.sentence(name, is_user, cur)

    # -- what is played ------------------------------------------------------------
    def watch(self, meta: Dict[str, Any], now: Optional[float] = None, occasion: str = "") -> List[str]:
        """Learn from what the now-playing sensor reports has just finished.

        Behaviour is weaker evidence than being told, so it moves a score in
        small steps (a listen to the end +0.10, a skip -0.20), never past
        +/-0.60 ("likes", not "loves"), and never changes what the person said
        themselves. Three listens make a song something they like; two skips,
        something they don't. Returns a sentence only when an item first
        crosses into the profile, so a play is not an announcement.
        """
        now = self.clock() if now is None else now
        cfg = self.cfg
        if not (cfg.enabled and cfg.from_playing) or not isinstance(meta, dict) or not meta.get("finished"):
            return []
        kind, secs = str(meta.get("finished_kind") or ""), float(meta.get("finished_s") or 0)
        title, artist = str(meta.get("finished_title") or ""), str(meta.get("finished_artist") or "")
        skipped = bool(meta.get("finished_skip"))
        changed = meta.get("finished_how") == "changed"
        if kind == "music":
            delta = -0.20 if skipped else (0.10 if secs >= cfg.listen_s else 0.0)
        elif kind in ("show", "media"):
            delta = -0.15 if changed and secs < 120 else (0.10 if secs >= cfg.watch_s else 0.0)
        elif kind in ("video", "stream"):
            delta = 0.10 if secs >= cfg.watch_s else 0.0         # hopping between videos says nothing
        else:
            delta = 0.0
        if not delta or not title:
            return []
        by = "action" if delta < 0 else "habit"
        item = f"{title} by {artist}" if artist else title
        rows = [(item, "music" if kind == "music" else kind, delta)]
        if artist and delta > 0:
            rows.append((artist, "music", delta / 2))             # the artist gets half the credit, never the blame
        key = self.user_key
        sentences: List[str] = []
        undo: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        with self.kb._lock:
            subj = self.kb.tastes.setdefault(key, {"name": self.user or "you", "user": True, "items": {}})
            subj["user"] = True
            items = subj.setdefault("items", {})
            for name, k, d in rows:
                ik = norm(name)
                if not ik:
                    continue
                cur = items.get(ik)
                prev_entry = copy.deepcopy(cur) or {}
                undo.append((ik, copy.deepcopy(cur)))
                before = float(cur.get("score", 0.0)) if cur else 0.0
                if cur is None:
                    if len(items) >= cfg.max_items_per_subject:
                        worst = min(items, key=lambda x: (abs(items[x].get("score", 0)), items[x].get("last", 0)))
                        del items[worst]
                    cur = items[ik] = {"item": name, "kind": k, "score": 0.0, "n": 0, "first": now, "last": now,
                                       "by": by, "flips": 0, "when": {}}
                cur["n"] = int(cur.get("n", 0)) + 1
                cur["last"] = now
                if delta > 0 and occasion in _ACTIVITIES:            # heard through, and what they were doing
                    sw = cur.setdefault("seen_when", {})
                    sw[occasion] = int(sw.get(occasion, 0)) + 1
                if cur.get("by") not in ("said", "opinion"):         # their own word stands
                    cur["score"] = round(max(-0.6, min(0.6, before + d)), 3)
                    cur["by"] = by if abs(cur["score"]) >= abs(before) else cur.get("by", by)
                now_score = float(cur["score"])
                linked = bool(occasion_of(cur)) and not occasion_of(prev_entry)
                if abs(before) < cfg.like_at <= abs(now_score) or (linked and abs(now_score) >= cfg.like_at):
                    sentences.append(self.sentence(self.user or "you", True, cur))
            self.kb.save()
        self.stats["played"] = self.stats.get("played", 0) + 1

        def revert(undo=undo, key=key):
            with self.kb._lock:
                its = (self.kb.tastes.get(key) or {}).get("items", {})
                for ik, old in undo:
                    if old is None:
                        its.pop(ik, None)
                    else:
                        its[ik] = old
                self.kb.save()
            self._mirror({key})
        self.last_undo = revert if sentences else None
        if sentences:
            self._mirror({key})
        return sentences

    # -- saying ---------------------------------------------------------------
    @staticmethod
    def sentence(name: str, is_user: bool, entry: Dict[str, Any], strong: float = 0.75) -> str:
        who = "The user" if is_user else name
        verb = _band(float(entry.get("score", 0.0)), strong)
        occ = occasion_of(entry)
        when = f", {_while(occ)}" if occ else ""
        return f"{who} {verb} {entry.get('item', '')}{when}."

    def items(self, key: str, polarity: int = 0, kind: str = "", limit: int = 8,
              min_abs: Optional[float] = None) -> List[Dict[str, Any]]:
        """A subject's tastes, strongest and most recent first."""
        floor = self.cfg.like_at if min_abs is None else min_abs
        with self.kb._lock:
            subj = self.kb.tastes.get(key) or {}
            rows = [dict(v) for v in (subj.get("items") or {}).values()
                    if abs(float(v.get("score", 0))) >= floor
                    and (not polarity or (float(v["score"]) > 0) == (polarity > 0))
                    and (not kind or v.get("kind") == kind)]
        rows.sort(key=lambda r: (-abs(float(r.get("score", 0))), -float(r.get("last", 0))))
        return rows[:limit]

    def subjects(self) -> List[Tuple[str, str, bool]]:
        with self.kb._lock:
            return [(k, v.get("name", k), bool(v.get("user"))) for k, v in self.kb.tastes.items()
                    if v.get("items")]

    def has_any(self) -> bool:
        return any(True for _ in self.subjects())

    def find(self, name: str) -> Optional[str]:
        """The subject key for a name (or 'me'/'I'), if it has a profile."""
        n = norm(name)
        if n in ("me", "i", "my", "myself", "mine", "") or n == norm(self.user):
            return self.user_key
        with self.kb._lock:
            if n in self.kb.tastes:
                return n
            for k, v in self.kb.tastes.items():
                if norm(v.get("name", "")).split()[:1] == [n]:
                    return k
        return None

    def _list(self, rows: List[Dict[str, Any]]) -> str:
        names = [r["item"] for r in rows]
        if len(names) <= 1:
            return "".join(names)
        return ", ".join(names[:-1]) + " and " + names[-1]

    def describe(self, key: str, polarity: int = 0, kind: str = "", you: bool = True) -> str:
        """One or two sentences on what a subject likes or dislikes."""
        with self.kb._lock:
            subj = self.kb.tastes.get(key) or {}
            name = subj.get("name", key)
        who = "You" if you and subj.get("user") else name
        strong = self.cfg.strong_at
        parts = []
        if polarity >= 0:
            love = [r for r in self.items(key, 1, kind, 12) if r["score"] >= strong]
            like = [r for r in self.items(key, 1, kind, 12) if r["score"] < strong]
            if love:
                parts.append(f"{who} love{'' if who == 'You' else 's'} {self._list(love[:6])}")
            if like:
                parts.append(f"{who} like{'' if who == 'You' else 's'} {self._list(like[:6])}")
        if polarity <= 0:
            hate = [r for r in self.items(key, -1, kind, 12) if r["score"] <= -strong]
            dis = [r for r in self.items(key, -1, kind, 12) if r["score"] > -strong]
            if hate:
                parts.append(f"{who} hate{'' if who == 'You' else 's'} {self._list(hate[:6])}")
            if dis:
                parts.append(f"{who} dislike{'' if who == 'You' else 's'} {self._list(dis[:6])}")
        if not parts:
            return ""
        # "You love X, and like Y. You hate Z." -- the second half keeps its own subject.
        return ". ".join(parts) + "."

    # -- being asked ------------------------------------------------------------
    _ASK_KIND = (r"(?:(?P<kind>music|songs?|movies?|films?|shows?|series|games?|food|foods|books?|apps?|"
                 r"anime|bands?|artists?)\s+)?")
    _ASK = [
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?what\s+(?:kind\s+of\s+|sort\s+of\s+|type\s+of\s+)?"
                   + _ASK_KIND + r"do\s+i\s+(?P<v>like|love|enjoy|hate|dislike|can'?t\s+stand)"
                   r"(?:\s+(?:to\s+)?(?P<act>watch|listen\s+to|play|eat|read|drink))?\s*\??\W*$", re.I),
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?what\s+(?:are|were)\s+my\s+(?P<v>likes|dislikes|favou?rites|"
                   r"hates|tastes?|preferences)(?:\s+in\s+(?P<kind>\w+))?\s*\??\W*$", re.I),
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?what\s+do\s+you\s+(?:know|remember)\s+about\s+(?:my\s+)?"
                   r"(?P<v>tastes?|likes|dislikes|preferences|favou?rites|what\s+i\s+like)\s*\??\W*$", re.I),
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?what\s+(?:does|did)\s+(?P<who>[A-Za-z][A-Za-z'\-]{1,19})\s+"
                   + _ASK_KIND + r"(?P<v>like|love|enjoy|hate|dislike|can'?t\s+stand)"
                   r"(?:\s+(?:to\s+)?(?P<act>watch|listen\s+to|play|eat|read|drink))?\s*\??\W*$", re.I),
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?do\s+i\s+(?P<v>like|love|enjoy|hate|dislike)\s+(?P<obj>.{2,40}?)"
                   r"\s*\??\W*$", re.I),
        # "what do I listen to while I'm coding?" -- the link between a thing and what was going on
        re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?what\s+(?:do|did)\s+i\s+(?:usually\s+|normally\s+|"
                   r"typically\s+|often\s+|always\s+)?(?P<act>listen\s+to|watch|play|eat|read)\s+"
                   r"(?:when|while|whenever)\s+(?:i'?m\s+|i\s+am\s+|i\s+|feeling\s+)?(?P<occ>[a-z]{3,14})\s*\??\W*$",
                   re.I),
    ]
    _OCC_VERBS = {"code": "coding", "work": "working", "study": "studying", "game": "gaming", "read": "reading",
                  "write": "writing", "cook": "cooking", "drive": "driving", "design": "designing",
                  "run": "running", "clean": "cleaning", "train": "training", "exercise": "exercising"}
    _KIND_CANON = {"songs": "music", "song": "music", "music": "music", "bands": "music", "band": "music",
                   "artists": "music", "artist": "music", "movies": "movie", "movie": "movie", "films": "movie",
                   "film": "movie", "shows": "show", "show": "show", "series": "show", "anime": "show",
                   "games": "game", "game": "game", "food": "food", "foods": "food", "books": "book",
                   "book": "book", "apps": "app", "app": "app"}
    _ACT_KIND = {"watch": "", "listen to": "music", "play": "game", "eat": "food", "drink": "food", "read": "book"}

    def wants(self, text: str) -> Optional[Dict[str, str]]:
        """The question being asked about taste, or None."""
        t = (text or "").strip()
        for rx in self._ASK:
            m = rx.match(t)
            if not m:
                continue
            d = m.groupdict()
            if d.get("occ"):
                occ = self._OCC_VERBS.get(d["occ"].lower(), d["occ"].lower())
                if occ not in _ACTIVITIES and occ not in _MOODS:
                    continue                          # not something it keeps: leave the question to the model
                act = re.sub(r"\s+", " ", d["act"].lower())
                return {"who": "", "pol": "1", "kind": self._ACT_KIND.get(act, ""), "obj": "", "occ": occ}
            v = re.sub(r"\s+", " ", (d.get("v") or "").lower())
            pol = -1 if re.search(r"hate|dislike|stand", v) else 1 if re.search(r"like|love|enjoy|favou?rite", v) else 0
            if re.search(r"likes|tastes?|preferences|what i like", v):
                pol = 0
            if v == "dislikes":
                pol = -1
            kind = self._KIND_CANON.get((d.get("kind") or "").lower(), "")
            if not kind and d.get("act"):
                kind = self._ACT_KIND.get(re.sub(r"\s+", " ", d["act"].lower()), "")
            return {"who": d.get("who") or "", "pol": str(pol), "kind": kind, "obj": (d.get("obj") or "").strip()}
        return None

    def answer(self, text: str, people: Iterable[str] = ()) -> Optional[str]:
        q = self.wants(text)
        if q is None or not self.cfg.enabled:
            return None
        pol, kind = int(q["pol"]), q["kind"]
        if q.get("occ"):
            return self._during(q["occ"], kind)
        if q["obj"]:
            return self._do_i_like(q["obj"], pol)
        who = q["who"]
        key = self.find(who) if who else self.user_key
        you = not who or key == self.user_key
        if key is None:
            return (f"I don't know what {who} likes yet. Tell me -- say \"{who} likes jazz\" and I'll remember it.")
        said = self.describe(key, pol, kind, you=you)
        if said:
            return said
        what = {1: "like", -1: "dislike", 0: "like or dislike"}[pol]
        of = f" in {kind}" if kind else ""
        if you:
            return (f"You haven't told me what you {what}{of} yet. Say \"I love Interstellar\" or \"I can't stand "
                    f"horror films\" and I'll remember it.")
        return f"I don't have anything on what {who} {what}s{of} yet."

    def linked(self, occ: str, key: str = "", kind: str = "", limit: int = 6) -> List[Dict[str, Any]]:
        """What a subject likes that goes with an occasion, said or seen often enough."""
        key = key or self.user_key
        return [r for r in self.items(key, 1, kind, 40) if occasion_of(r) == occ][:limit]

    def _during(self, occ: str, kind: str) -> str:
        rows = self.linked(occ, kind=kind)
        what = {"music": "listen to", "show": "watch", "movie": "watch", "game": "play", "food": "eat",
                "book": "read"}.get(kind, "like")
        if not rows:
            return (f"I haven't noticed what you {what} {_while(occ)} yet. Tell me -- say \"I listen to lo-fi "
                    f"while I'm coding\" -- or I'll notice it as it happens.")
        return f"{_while(occ).capitalize()}, you {what} {self._list(rows)}."

    def _do_i_like(self, obj: str, pol: int) -> str:
        item = _clean_item(obj) or obj
        with self.kb._lock:
            entry = ((self.kb.tastes.get(self.user_key) or {}).get("items") or {}).get(norm(item))
            entry = dict(entry) if entry else None
        if not entry:
            return f"You haven't told me how you feel about {item}."
        verb = _band(float(entry.get("score", 0.0)), self.cfg.strong_at)[:-1]      # loves -> love
        return f"You {verb} {entry['item']}."

    # -- being used ------------------------------------------------------------
    _CHOOSING = re.compile(
        r"\b(?:recommend|suggest|suggestion|something\s+(?:to|fun|new|good|nice|cool)|what\s+should\s+i|"
        r"what\s+to\s+(?:watch|play|eat|listen|read|cook|have|order|get)|i'?m\s+bored|bored|in\s+the\s+mood|"
        r"mood\s+for|any\s+ideas|ideas\s+for|favou?rite|good\s+(?:movie|film|show|song|game|book|food)s?|"
        r"watch|listen|playlist|dinner|lunch|snack|order\s+(?:food|something))\b", re.I)

    def context(self, query: str, now: Optional[float] = None, people: Iterable[str] = ()) -> str:
        """Background for the model when the question is about choosing something.

        Empty for everything else. The profile is not a thing to bring up; it
        is a thing to draw on when someone asks what to watch.
        """
        if not (self.cfg.enabled and self.cfg.in_answers):
            return ""
        q = (query or "")
        names = [p for p in people if p]
        mentioned = [k for k, n, _u in self.subjects() if k != self.user_key and re.search(
            r"\b" + re.escape(n.split()[0]) + r"\b", q, re.I)]
        if not self._CHOOSING.search(q) and not mentioned:
            return ""
        kind = ""
        for kw, canon in self._KIND_CANON.items():
            if re.search(r"\b" + kw + r"\b", q, re.I):
                kind = canon
                break
        occ = ""
        mo = re.search(r"\b(?:when|while|whenever)\s+(?:i'?m\s+|i\s+am\s+|i\s+)?([a-z]{3,14})\b", q, re.I)
        if mo:
            occ = self._OCC_VERBS.get(mo.group(1).lower(), mo.group(1).lower())
            occ = occ if occ in _ACTIVITIES or occ in _MOODS else ""
        lines = []
        for key in [self.user_key] + mentioned[:2]:
            likes = self.items(key, 1, kind, 8)
            if key == self.user_key and occ:
                first = self.linked(occ, key, kind)
                likes = first + [r for r in likes if r not in first]
            dis = self.items(key, -1, kind, 5)
            if not likes and not dis and kind:
                likes, dis = self.items(key, 1, "", 6), self.items(key, -1, "", 4)
            if not likes and not dis:
                continue
            with self.kb._lock:
                name = (self.kb.tastes.get(key) or {}).get("name", key)
            who = "The user" if key == self.user_key else name
            bits = []
            if likes:
                bits.append("likes " + ", ".join(self._with_when(r) for r in likes))
            if dis:
                bits.append("dislikes " + ", ".join(r["item"] for r in dis))
            lines.append(f"{who} " + "; ".join(bits) + ".")
        if not lines:
            return ""
        return ("[Tastes, for choosing what to suggest -- background only. Pick from what they like; "
                "never read this list back or say you were given it:\n" + "\n".join(lines) + "]")

    @staticmethod
    def _with_when(r: Dict[str, Any]) -> str:
        occ = occasion_of(r)
        return f"{r['item']} ({_while(occ)})" if occ else r["item"]

    # -- entity memory -------------------------------------------------------------
    def _mirror(self, keys: Iterable[str]):
        """A short likes/dislikes pair on each person's entity, so the entity
        block the narrator already receives carries it."""
        if self.cfg.mirror:
            mirror(self.kb, keys, self.user, self.cfg.like_at)

    # -- managing -----------------------------------------------------------------
    def forget(self, name: str = "") -> int:
        """Forget one subject's tastes ("" is everyone's). Returns how many items went."""
        with self.kb._lock:
            if not name:
                n = sum(len(v.get("items") or {}) for v in self.kb.tastes.values())
                self.kb.tastes.clear()
            else:
                key = self.find(name)
                n = len((self.kb.tastes.get(key) or {}).get("items") or {}) if key else 0
                if key:
                    del self.kb.tastes[key]
            self.kb.save()
        return n

    def report(self) -> str:
        subs = self.subjects()
        if not subs:
            return "No tastes yet. Say \"I love Interstellar\" or \"Ali likes jazz\" and they are kept."
        lines = []
        for key, name, is_user in sorted(subs, key=lambda s: (not s[2], s[1])):
            said = self.describe(key, 0, "", you=False) or "(nothing strong yet)"
            lines.append(f"{name + ' (you)' if is_user else name}: {said}")
        return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        with self.kb._lock:
            people = len(self.kb.tastes)
            items = sum(len(v.get("items") or {}) for v in self.kb.tastes.values())
        return {"enabled": self.cfg.enabled, "people": people, "items": items, **self.stats}
