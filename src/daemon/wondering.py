"""
src/daemon/wondering.py

Wondering: noticing what ASH does not know, and asking.

Someone mentions "my brother" and ASH has never heard who that is. They talk
about Sara twice and nothing in memory says who she is. They use "ZXQ" as if
everyone knew. The old behaviour was to carry on regardless -- guessing, or
staying silent and learning nothing. A person in that position wonders, and
at a good moment asks.

Four parts, all rule-based (no model, no network, inspectable and testable):

    detectors     read what the user said and find a GAP: an unknown person, an
                  unresolved role ("my boss"), two people with no known
                  relation, an unknown term or acronym, a bare name that fits
                  two people, a missing fact (birthday), an unclear intent.
                  Statements that *close* a gap ("Sara is my sister") are
                  learned on the spot, with no question asked.
    queue         every gap is a Wonder. Deduplicated, prioritised, rate
                  capped, cooled down per subject, expired, and never
                  re-asked once answered or dismissed.
    asking        the one question that is next, only at a moment the quiet
                  rules allow: present but paused, not muted, not asleep, not
                  mid-conversation, nothing else awaiting an answer.
    answers       the next reply is read as the answer if it plausibly is one
                  (a name, a relation, a definition), stored in Knowledge so
                  ASH stops wondering, and a "not now" / "stop asking about
                  that" / "stop asking me things" is honoured. An unclear
                  answer earns one follow-up and then the question is let go.

What this module does NOT do: speak (the ambient loop delivers the question
through the attention gate like any other offer, so the hourly budget, quiet
hours and the learned bar all still apply) or decide whether a moment is
quiet (the caller describes the moment in an `Environment`; the rules that
read it are here and tested).

Privacy: only the user's own utterances (typed, or heard by the microphone)
are read; screen text is not, unless `sources` says so. A wonder keeps a
shortened, redacted snippet as evidence only while it is open and only if
`keep_evidence` is on; it is erased when the wonder closes. Logging names the
kind and id of a wonder, never the words. The state file lives under state/,
which is not committed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.wondering")

HOUR = 3600.0
DAY = 86400.0

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
# Every word that names a relationship, mapped to one canonical form.
ROLE_CANON: Dict[str, str] = {
    "mother": "mother", "mom": "mother", "mum": "mother", "mama": "mother", "mommy": "mother",
    "father": "father", "dad": "father", "papa": "father", "daddy": "father",
    "brother": "brother", "sister": "sister", "sibling": "sibling",
    "wife": "wife", "husband": "husband", "spouse": "spouse",
    "son": "son", "daughter": "daughter", "child": "child", "kid": "child",
    "uncle": "uncle", "aunt": "aunt", "auntie": "aunt", "cousin": "cousin",
    "grandmother": "grandmother", "grandma": "grandmother", "granny": "grandmother",
    "grandfather": "grandfather", "grandpa": "grandfather", "grandad": "grandfather",
    "nephew": "nephew", "niece": "niece",
    "friend": "friend", "bestie": "friend",
    "boyfriend": "boyfriend", "girlfriend": "girlfriend", "partner": "partner",
    "fiance": "fiance", "fiancee": "fiance", "fiancé": "fiance", "fiancée": "fiance",
    "boss": "boss", "manager": "manager", "supervisor": "manager",
    "colleague": "colleague", "coworker": "colleague", "teammate": "colleague",
    "teacher": "teacher", "professor": "teacher", "tutor": "teacher",
    "neighbor": "neighbor", "neighbour": "neighbor",
    "roommate": "roommate", "flatmate": "roommate", "housemate": "roommate",
    "doctor": "doctor", "landlord": "landlord", "landlady": "landlord",
    "classmate": "classmate", "mentor": "mentor", "client": "client",
    "employee": "employee", "assistant": "assistant",
}
_ROLE_ALT = "|".join(sorted(ROLE_CANON, key=len, reverse=True))

# Roles a person has one of, so "my boss" with no name is a real gap.
# "my friend" / "my colleague" are not asked about: there are many.
UNIQUE_ROLES = frozenset({
    "mother", "father", "brother", "sister", "wife", "husband", "spouse", "son",
    "daughter", "boss", "manager", "partner", "girlfriend", "boyfriend", "fiance",
    "roommate", "landlord", "uncle", "aunt", "grandmother", "grandfather", "mentor",
})

# Family ties do not change, so two different ones for the same person cannot
# both be true. Other relations can (a colleague who is also a friend, a
# boyfriend who becomes a fiance): a new one simply replaces the old.
FAMILY = frozenset({"mother", "father", "brother", "sister", "sibling", "son", "daughter", "child",
                    "uncle", "aunt", "cousin", "grandmother", "grandfather", "nephew", "niece"})
_GENERIC = {"sibling": {"brother", "sister"}, "child": {"son", "daughter"}}
# Roles one person fills at a time and that are handed on: a new boss replaces
# the old one. (A second brother does not replace the first.)
REPLACEABLE = frozenset({"boss", "manager", "landlord", "partner", "girlfriend", "boyfriend", "fiance",
                         "wife", "husband", "spouse", "mentor"})
# Relations that end or change with time, so worth checking after a while.
CHANGEABLE = frozenset({"boss", "manager", "colleague", "roommate", "landlord", "partner", "girlfriend",
                        "boyfriend", "fiance", "teacher", "mentor", "classmate", "client", "employee",
                        "neighbor", "doctor", "assistant"})


def relations_conflict(old: str, new: str) -> bool:
    """Two relations for one person that cannot both be true."""
    if not old or not new or old == new:
        return False
    if new in _GENERIC.get(old, ()) or old in _GENERIC.get(new, ()):
        return False
    return old in FAMILY or new in FAMILY


def definitions_differ(old: str, new: str) -> bool:
    """Two definitions that say different things, not the same thing twice."""
    a, b = set(norm(old).split()), set(norm(new).split())
    if not a or not b or norm(old) == norm(new):
        return False
    if norm(old) in norm(new) or norm(new) in norm(old):
        return False
    return len(a & b) / len(a | b) < 0.5


# The company a name keeps. Sara mentioned three times next to a meeting, a
# deadline and the team is probably a colleague; asking "Is Sara a colleague?"
# is easier to answer than "Who is Sara to you?". Only a guess: it is put as a
# yes/no question, and only a yes stores it.
CONTEXT_CUES: Dict[str, Tuple[str, str, "re.Pattern[str]"]] = {
    "work": ("colleague", "around work", re.compile(
        r"\b(?:meetings?|stand-?ups?|projects?|office|team|deadlines?|clients?|sprints?|reviews?|"
        r"pull requests?|PRs?|deploy(?:ment)?s?|reports?|presentations?|work|job|shift|slack|jira|"
        r"tickets?|code|release|manager|boss)\b", re.I)),
    "school": ("classmate", "around your classes", re.compile(
        r"\b(?:class(?:es)?|lectures?|homework|assignments?|exams?|course|professor|campus|semester|"
        r"labs?|university|uni|school|study(?:ing)?|thesis|tutorial)\b", re.I)),
    "family": ("family", "with your family", re.compile(
        r"\b(?:family|parents|mom|mum|dad|grandma|grandpa|relatives|eid|family dinner|at home|"
        r"wedding|cousins)\b", re.I)),
}
GUESS_MIN_CUES = 2


def context_of(sent: str) -> Dict[str, int]:
    """Which kinds of company a sentence keeps, by cue words."""
    out = {}
    for k, (_r, _w, rx) in CONTEXT_CUES.items():
        n = len(rx.findall(sent))
        if n:
            out[k] = n
    return out


# What a relation between two people can be said as, to one label.
BETWEEN_CANON: List[Tuple[str, str]] = [
    (r"\b(?:married|husband and wife|wife and husband|spouses|newlyweds?)\b", "married"),
    (r"\b(?:dating|a couple|couple|together|partners|engaged)\b", "partners"),
    (r"\b(?:siblings|brothers|sisters|brother and sister|twins)\b", "siblings"),
    (r"\b(?:cousins)\b", "cousins"),
    (r"\b(?:colleagues|coworkers|co-workers|teammates|work together|same team|same company)\b", "colleagues"),
    (r"\b(?:friends|best friends|buddies|pals)\b", "friends"),
    (r"\b(?:neighbou?rs|next door)\b", "neighbors"),
    (r"\b(?:classmates|schoolmates|same class|same school)\b", "classmates"),
    (r"\b(?:parent and child|mother and son|mother and daughter|father and son|"
     r"father and daughter|mom and|dad and|family|related|relatives)\b", "family"),
    (r"\b(?:roommates|flatmates|housemates)\b", "roommates"),
    (r"\b(?:strangers|don'?t know each other|not related|unrelated)\b", "unrelated"),
]

FACT_NOUNS: Dict[str, str] = {"birthday": "When is your birthday?",
                              "anniversary": "When is your anniversary?"}

# Capitalised words that are not somebody's name.
NOT_PERSON = frozenset("""
i i'm i'll i've i'd ill im ash a.s.h okay ok yes no yeah yep nope hi hey hello thanks thank please sorry
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
english arabic french german spanish italian chinese japanese russian turkish hebrew hindi urdu korean
python java javascript typescript rust golang ruby swift kotlin linux windows macos mac android ios ubuntu
chrome firefox edge safari opera google microsoft apple amazon facebook meta twitter youtube netflix spotify
github gitlab discord slack zoom teams skype whatsapp telegram instagram tiktok reddit wikipedia claude
gpt chatgpt openai anthropic gemini copilot siri alexa docker kubernetes git npm pip excel word powerpoint
outlook gmail notepad photoshop vscode code pycharm jupyter
america europe asia africa australia antarctica
paris london rome berlin madrid cairo amman dubai tokyo beijing delhi mumbai moscow istanbul
canada mexico brazil india china japan egypt jordan palestine israel iran iraq syria lebanon turkey
germany france spain italy russia england scotland wales ireland britain
christmas ramadan eid easter halloween thanksgiving
god sir madam mr mrs ms dr
the this that there then they them their these those what when where which while who whom whose why how
and but for nor yet so because although though however also again still just only even
today tomorrow yesterday tonight now soon later morning evening afternoon night
""".split())

_COMMON_ACRONYMS = frozenset("""
OK PC TV USB CPU GPU RAM ROM SSD HDD AI ML API URL HTML CSS SQL PDF USA UK UAE EU UN US ID FAQ ASAP FYI LOL OMG
DIY GPS SMS MMS WIFI LED LCD OLED VPN DNS HTTP HTTPS FTP SSH IP TCP UDP OS UI UX CEO CFO CTO HR PR IT BTW IMO IMHO
TBH TLDR AM PM AD BC NASA FBI CIA NATO WHO WHO'S NO YES THE AND FOR NOT BUT YOU ARE WAS HAS ALL CAN HOW WHY
JSON XML CSV YAML GUI CLI IDE SDK NPM PIP GIT AWS GCP USD EUR GBP JOD ATM PIN SIM MP3 MP4 JPG PNG GIF ZIP
""".split())

# Said before a name, so that what follows is somebody.
_STRONG_BEFORE = frozenset("""told tell tells ask asked asks call called calls text texted email emailed message
messaged met meet hi hey hello dear thanks named""".split())
_WEAK_BEFORE = frozenset("with see saw visit visited visiting and".split())
_PAIR_BEFORE = {("talk", "to"), ("talked", "to"), ("spoke", "to"), ("speak", "to"),
                ("talking", "to"), ("listen", "to"), ("listened", "to"), ("said", "to"),
                ("write", "to"), ("wrote", "to"), ("thank", "")}
_AFTER_VERBS = frozenset("""said says told tells called calls asked asks texted emailed messaged wants wanted
came comes coming arrives arrived lives lived visited visits mentioned thinks thought left leaves
phoned phones""".split())

# What a reply that is not an answer looks like.
_NON_ANSWER_START = re.compile(
    r"^\s*(?:what|when|where|why|how|who|which|whose|can|could|would|will|do|does|did|is|are|am|was|were|"
    r"should|shall|please|open|play|set|remind|search|turn|show|tell|find|look|check|start|stop|close|"
    r"launch|run|send|add|remove|delete|go|take|make|give|let|call|text|read|write|translate|calculate)\b",
    re.I)
_DONT_KNOW = re.compile(
    r"\b(?:don'?t know|do not know|no idea|not sure|unsure|nobody|no one|none|n/a|skip|never ?mind|"
    r"forget it|doesn'?t matter|does not matter|not important|whatever|can'?t say|can'?t remember)\b", re.I)
_NOT_NOW = re.compile(
    r"\b(?:not now|not right now|later|i'?m busy|busy right now|ask me later|another time|some other time|"
    r"in a (?:bit|while|minute))\b", re.I)
_STOP_ABOUT = re.compile(
    r"\b(?:stop|quit|don'?t|do not|no more|never|enough)\b.{0,24}?\b(?:ask(?:ing)?|wonder(?:ing)?|bring(?:ing)? up)\b"
    r"(?:.{0,12}?\b(?:about|on)\b\s+(?P<subj>[^.?!,;]{1,40}))?", re.I)
_STOP_ALL = re.compile(
    r"\b(?:stop|quit|no more|enough|don'?t|do not|never|cut it out)\b.{0,30}?"
    r"\b(?:questions|asking me|asking things|asking stuff|wondering|being curious|curious|"
    r"ask(?:ing)? me (?:anything|things|stuff|questions))\b", re.I)
_RESUME = re.compile(r"\b(?:you can|feel free to|go ahead and|please)\b.{0,12}\b(?:ask|wonder)\b.{0,20}\b(?:again|questions|me things)\b",
                     re.I)
# A reply to "I think X means Y -- is that right?".
_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|yea|right|correct|exactly|true|indeed|sure|mhm|uh-huh|"
                  r"that'?s (?:it|right|correct)|you got it|spot on)\b(?!\s*,?\s*but\b)", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|not quite|not really|not exactly|wrong|incorrect)\b[\s,.:;!-]*"
                 r"(?:it'?s|it is|it means|that'?s|that is|actually)?[\s,]*(?P<rest>.*)$", re.I)
# The user asking for the questions: answered there and then, one after another.
_ASK_ME = re.compile(
    r"\b(?:what (?:are|were) you (?:wondering|curious) about|(?:do you have|got|have you got) "
    r"(?:any )?questions(?: for me)?|any questions(?: for me)?|ask (?:me )?(?:your|the|those|any|some) questions|"
    r"what do you want to (?:know|ask)|what are you wondering)\b", re.I)
_BATCH_END = re.compile(r"^\s*(?:that'?s (?:all|enough|it)|enough(?: for now)?|done|stop|let'?s stop|"
                        r"no more|later)\b", re.I)
_THAT = re.compile(r"\b(?:that|this|it|him|her|them|those|these|about (?:that|this|it))\b", re.I)
_ORDINALS = {"first": 0, "1st": 0, "one": 0, "1": 0, "former": 0, "second": 1, "2nd": 1, "two": 1, "2": 1,
             "latter": 1, "third": 2, "3rd": 2, "three": 2, "3": 2}

_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_DATEISH = re.compile(r"\d|\b(?:" + "|".join(_MONTHS) + r")[a-z]*\b|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|"
                      r"\b(?:today|tomorrow|tonight|next|last)\b", re.I)

_FILLER = frozenset("hm hmm hmmm um umm uh uhh ugh eh er erm ah oh huh lol idk meh yeah yes yep yup nope nah ok okay "
                    "well so sure fine maybe what why hey wow".split())
_CAPWORD =re.compile(r"[A-Za-z][A-Za-z\-]*")


def norm(s: str) -> str:
    """Lowercase, punctuation to single spaces: the form subjects are keyed by."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _as_text(x: Any) -> str:
    """Whatever arrived, as text: anything that is not a string is nothing said."""
    return x if isinstance(x, str) else ""


def _clean(s: str, n: int = 60) -> str:
    """Printable, single line, bounded: what is safe to speak back."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip(" \t\"'`")
    return s[:n].strip()


def _title(name: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in name.split())


# ----------------------------------------------------------------------------
# Wonders
# ----------------------------------------------------------------------------
# "activity" is a gap noticed in what the user DOES rather than what they said:
# a project they keep returning to that ASH has no idea about (see activity.py).
# "conflict" is two answers that cannot both be true ("Omar is my cousin", later
# "Omar is my brother"); "stale" is an answer old enough to check again ("is
# Dana still your boss?"). Both come from what ASH already knows.
# "routine" is a habit noticed in what the user does (routines.py): whether
# to keep it as something ASH can do for them.
KINDS = ("person", "role", "between", "term", "fact", "ambiguous", "intent", "activity",
         "conflict", "stale", "routine")
BASE_PRIORITY = {"conflict": 0.75, "stale": 0.30, "routine": 0.32,
                 "ambiguous": 0.90, "intent": 0.80, "role": 0.70, "person": 0.60,
                 "between": 0.50, "term": 0.50, "fact": 0.40, "activity": 0.35}
# How much each kind of gap matters RIGHT NOW, before repetition adds to it.
# High: ASH cannot act well without the answer (two people fit, a request it
# could not read). Low: it would merely like to know. Only the high ones are
# worth interrupting a pause for; the rest wait for a free moment.
BASE_URGENCY = {"conflict": 0.55, "stale": 0.12, "routine": 0.10,
                "ambiguous": 0.85, "intent": 0.80, "role": 0.45, "person": 0.35,
                "term": 0.30, "between": 0.25, "fact": 0.20, "activity": 0.10}
# Replies that are noise, not an answer, wherever a definition is wanted.
_FILLER = frozenset("hm hmm mm mmm um uh ok okay yes yeah yep yup no nope nah sure idk maybe".split())
OPEN = ("pending", "asked")
TERMINAL = ("answered", "dismissed", "unclear", "expired")
# Closed for good: never asked again, whatever is said later.
FOREVER = ("answered", "dismissed", "unclear")


@dataclass
class Wonder:
    id: str
    kind: str                         # one of KINDS
    subject: str                      # what it is about, as shown
    question: str                     # base question text (a lead-in may be added when asked)
    key: str = ""                     # dedupe identity
    subjects: List[str] = field(default_factory=list)   # normalised names, for mute and cooldown
    evidence: str = ""                # short redacted snippet; erased on close
    created: float = 0.0
    status: str = "pending"           # pending | asked | answered | dismissed | unclear | expired
    priority: float = 0.5
    urgency: float = 0.3              # 0..1: how much it matters to know now
    mentions: int = 1
    asked_at: float = 0.0
    asked_count: int = 0
    deferrals: int = 0
    not_before: float = 0.0
    followup_used: bool = False
    held: int = 0                     # times the attention gate declined to let it through
    data: Dict[str, Any] = field(default_factory=dict)   # kind-specific: role, names, options...
    source: str = ""
    closed_at: float = 0.0

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Reply:
    """What the user's utterance did to the wondering state."""
    consumed: bool = False            # it was an answer or a control phrase: skip the normal cycle
    say: str = ""                     # what to say back (acknowledgement or the one follow-up)
    outcome: str = ""                 # answered | dismissed | deferred | muted | snoozed | followup |
                                      # unclear | resumed | learned | ""
    wonder_id: str = ""
    learned: List[str] = field(default_factory=list)    # plain sentences for the concept network


@dataclass
class Environment:
    """The moment, as the caller sees it. None / False are the safe answers."""
    idle_s: Optional[float] = None    # seconds since the user touched anything; None: unknown
    muted: bool = False
    quiet_hours: bool = False
    paused: bool = False              # privacy pause
    conversation_open: bool = False
    plan_active: bool = False
    speaking: bool = False
    # What makes a moment a good one for a *non-urgent* question:
    free_moment: bool = False         # a stretch of work just ended, the user paused
    mood: Optional[float] = None      # -1 (irritated/low) .. 1 (good), None: unknown
    activity: str = ""                # what they were just doing ("VS Code", a file...)
    extra: Dict[str, Any] = field(default_factory=dict)   # room for future signals


# A fit scorer says how good THIS moment is for asking THIS wonder: 0 forbids,
# 1 is neutral, above 1 favours. They multiply. New signals (time of day,
# what the user was doing, how the last question landed) are one function
# each, added with `Wonderer.add_fit`.
FitFn = Callable[["Wonder", "Environment"], float]


def mood_fit(w: "Wonder", env: "Environment") -> float:
    """A low mood is no time for idle curiosity. Urgent questions still go."""
    if env.mood is None:
        return 1.0
    if env.mood <= -0.3 and w.urgency < 0.75:
        return 0.0
    return 1.0 + 0.2 * max(0.0, env.mood)


def activity_fit(w: "Wonder", env: "Environment") -> float:
    """A question about something related to what they were just doing is
    better timed than a stray one."""
    if not env.activity:
        return 1.0
    a = norm(env.activity)
    related = any(s and s in a for s in w.subjects) or any(
        t and t in a for s in w.subjects for t in s.split() if len(t) > 3)
    return 1.3 if related else 1.0


@dataclass
class WonderingConfig:
    enabled: bool = True
    max_per_hour: int = 2
    max_per_day: int = 8
    min_gap_s: float = 600.0              # between any two questions
    subject_cooldown_s: float = 6 * HOUR  # before the same subject is raised again
    expire_s: float = 3 * DAY             # a pending wonder this old is let go
    answer_window_s: float = 180.0        # the next reply within this is read as the answer
    late_answer_s: float = DAY            # a reply naming the subject is an answer for this long
    min_idle_s: float = 15.0              # present but paused: not mid-keystroke
    max_idle_s: float = 600.0             # ...and not gone
    hold_backoff_s: float = 600.0         # after the gate declined, before trying again
    max_pending: int = 30
    max_holds: int = 6                    # gate declined this often: let it go
    max_deferrals: int = 2                # "not now" this often means "no"
    reask_unanswered: int = 1             # asked, ignored: ask once more later
    reask_after_s: float = 6 * HOUR
    global_snooze_s: float = DAY          # "stop asking me things"
    urgent_at: float = 0.75               # at or above: may ask at an ordinary pause
    free_moment_s: float = 180.0          # how long a free moment stays open
    min_mentions_weak: int = 2            # a weak cue ("with Sara") needs this many sightings
    keep_evidence: bool = True
    sources: Tuple[str, ...] = ("console", "microphone", "mic")
    kinds: Dict[str, bool] = field(default_factory=lambda: {k: True for k in KINDS})
    ignore_names: Tuple[str, ...] = ()    # the user's own name and the like
    stale_after_s: float = 180 * DAY      # a changeable relation this old is checked when it comes up

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "WonderingConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if not hasattr(cfg, k):
                continue
            if k == "kinds" and isinstance(v, dict):
                cfg.kinds = {**cfg.kinds, **{str(a): bool(b) for a, b in v.items()}}
            elif k in ("sources", "ignore_names") and isinstance(v, (list, tuple)):
                setattr(cfg, k, tuple(str(x) for x in v))
            elif isinstance(getattr(cfg, k), bool):
                setattr(cfg, k, bool(v))
            elif isinstance(getattr(cfg, k), (int, float)) and isinstance(v, (int, float)):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


# ----------------------------------------------------------------------------
# Knowledge: where answers go
# ----------------------------------------------------------------------------
class Knowledge:
    """What ASH has been told about people and terms.

    Backed by a small JSON file of its own, and mirrored into entity memory
    (entities.json) when an EntityManager is supplied, so the narrator's
    entity context carries the relation too. Entity memory also *feeds* the
    gap detectors: a person it already has a relation for is not asked about.
    """

    def __init__(self, path: Optional[str] = None, entity_manager=None,
                 clock: Callable[[], float] = time.time):
        self.path = path
        self.entities = entity_manager
        self.clock = clock
        self._lock = threading.RLock()
        self.people: Dict[str, Dict[str, Any]] = {}     # norm name -> {name, relation, desc, between{}}
        self.roles: Dict[str, str] = {}                 # canonical role -> name
        self.terms: Dict[str, Dict[str, str]] = {}      # norm term -> {term, definition}
        self.facts: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}               # bare name -> full name chosen
        self.procedures: Dict[str, Dict[str, Any]] = {}  # norm name -> {name, steps}
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.people = dict(raw.get("people") or {})
            self.roles = dict(raw.get("roles") or {})
            self.terms = dict(raw.get("terms") or {})
            self.facts = dict(raw.get("facts") or {})
            self.aliases = dict(raw.get("aliases") or {})
            self.procedures = dict(raw.get("procedures") or {})
        except Exception:
            logger.exception("Knowledge file unreadable; starting empty")
            _quarantine(self.path)

    def save(self):
        if not self.path:
            return
        _atomic_write(self.path, {"people": self.people, "roles": self.roles, "terms": self.terms,
                                  "facts": self.facts, "aliases": self.aliases,
                                  "procedures": self.procedures})

    # -- entity memory as a source ----------------------------------------
    def _entity_people(self) -> List[Any]:
        try:
            return [e for e in self.entities.store.all()
                    if getattr(e, "entity_type", "") == "person"] if self.entities else []
        except Exception:
            return []

    @staticmethod
    def _entity_names(e) -> List[str]:
        names = []
        for v in (e.primary_identifiers.get("name"), e.attributes.get("canonical_name"),
                  e.attributes.get("full_name")):
            if v:
                names.append(str(v))
        return names

    @staticmethod
    def _entity_has_relation(e) -> bool:
        a = e.attributes or {}
        return any(a.get(k) for k in ("relation_to_user", "relationship", "relation", "role", "description"))

    # -- queries -----------------------------------------------------------
    def person_known(self, name: str) -> bool:
        """Is there a relation (or description) on record for this person?"""
        n = norm(name)
        with self._lock:
            p = self.people.get(n) or self.people.get(norm(self.aliases.get(n, "")))
            if p and (p.get("relation") or p.get("desc") or p.get("between")):
                return True
            for e in self._entity_people():
                if n in [norm(x) for x in self._entity_names(e)] and self._entity_has_relation(e):
                    return True
        return False

    def person_seen(self, name: str) -> bool:
        n = norm(name)
        with self._lock:
            if n in self.people:
                return True
            return any(n in [norm(x) for x in self._entity_names(e)] for e in self._entity_people())

    def between_known(self, a: str, b: str) -> bool:
        na, nb = norm(a), norm(b)
        with self._lock:
            return bool((self.people.get(na) or {}).get("between", {}).get(nb)
                        or (self.people.get(nb) or {}).get("between", {}).get(na))

    def role_name(self, role: str) -> Optional[str]:
        with self._lock:
            return self.roles.get(ROLE_CANON.get(role, role))

    def term_known(self, term: str) -> bool:
        with self._lock:
            return norm(term) in self.terms

    def fact(self, key: str) -> Optional[str]:
        with self._lock:
            return self.facts.get(key)

    def candidates(self, first: str) -> List[str]:
        """Distinct full names whose first word is `first`, from memory."""
        f = norm(first)
        with self._lock:
            if f in self.aliases:
                return []                                   # already chosen: not ambiguous
            out: Dict[str, str] = {}
            for p in self.people.values():
                full = p.get("name", "")
                if " " in full and norm(full).split()[0] == f:
                    out[norm(full)] = full
            for e in self._entity_people():
                for full in self._entity_names(e):
                    if " " in full.strip() and norm(full).split()[0] == f:
                        out[norm(full)] = _title(full)
        return sorted(out.values())

    # -- writes ------------------------------------------------------------
    def _person(self, name: str) -> Dict[str, Any]:
        n = norm(name)
        return self.people.setdefault(n, {"name": _title(name), "relation": "", "desc": "", "between": {}})

    def relation_of(self, name: str) -> str:
        """The relation on record in ASH's own store ('' if none)."""
        with self._lock:
            return (self.people.get(norm(name)) or {}).get("relation", "")

    def relation_age(self, name: str, now: float) -> Optional[float]:
        """Seconds since the relation was told or confirmed; None if unknown."""
        with self._lock:
            t = (self.people.get(norm(name)) or {}).get("t")
            return None if t is None else max(0.0, now - float(t))

    def touch(self, name: str, now: float) -> None:
        """The person came up. An answer from before timestamps existed starts
        its clock now, so it is not taken for an old one."""
        with self._lock:
            p = self.people.get(norm(name))
            if p is None:
                return
            p["seen"] = now
            if p.get("relation") and "t" not in p:
                p["t"] = now
            self.save()

    def confirm(self, name: str, now: float) -> None:
        with self._lock:
            p = self.people.get(norm(name))
            if p is not None:
                p["t"] = now
                self.save()

    def retire(self, name: str, role: str) -> None:
        """Someone is no longer the user's <role>: keep them, as a former one."""
        with self._lock:
            p = self._person(name)
            p["relation"] = ""
            p["desc"] = f"your former {role}"
            if self.roles.get(role) and norm(self.roles[role]) == norm(name):
                del self.roles[role]
            self.save()
        self._mirror("person", name, {"relation_to_user": "", "description": f"former {role}"})

    def store_relation(self, name: str, role: str = "", desc: str = "") -> None:
        with self._lock:
            p = self._person(name)
            if role:
                p["relation"] = ROLE_CANON.get(role, role)
                p["t"] = self.clock()
                if role in UNIQUE_ROLES or ROLE_CANON.get(role) in UNIQUE_ROLES:
                    self.roles[ROLE_CANON.get(role, role)] = p["name"]
            if desc:
                p["desc"] = _clean(desc, 80)
            self.save()
        self._mirror("person", name, {"relation_to_user": p["relation"]} if p["relation"]
                     else {"description": p["desc"]})

    def store_role(self, role: str, name: str) -> None:
        self.store_relation(name, role=role)

    def store_between(self, a: str, b: str, label: str) -> None:
        with self._lock:
            pa, pb = self._person(a), self._person(b)
            pa["between"][norm(b)] = label
            pb["between"][norm(a)] = label
            self.save()
        self._mirror("person", a, {"relations": {_title(b): label}})
        self._mirror("person", b, {"relations": {_title(a): label}})

    def store_term(self, term: str, definition: str) -> None:
        with self._lock:
            self.terms[norm(term)] = {"term": _clean(term, 40), "definition": _clean(definition, 200)}
            self.save()
        self._mirror("term", term, {"definition": _clean(definition, 200)})

    def store_fact(self, key: str, value: str) -> None:
        with self._lock:
            self.facts[key] = _clean(value, 80)
            self.save()

    def store_procedure(self, name: str, steps: List[str]) -> None:
        """A routine the user agreed to keep: a name and its steps (apps)."""
        name = _clean(name, 40)
        steps = [_clean(s, 40) for s in steps if s][:8]
        if not name or not steps:
            return
        with self._lock:
            self.procedures[norm(name)] = {"name": name, "steps": steps, "t": self.clock()}
            self.save()
        self._mirror("procedure", name, {"steps": steps, "description": "opens " + ", then ".join(steps)})

    def procedure(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.procedures.get(norm(name))

    def store_alias(self, bare: str, full: str) -> None:
        with self._lock:
            self.aliases[norm(bare)] = full
            self.save()

    def _mirror(self, etype: str, name: str, attrs: Dict[str, Any]):
        if self.entities is None:
            return
        try:
            self.entities.ingest({"entity_type": etype,
                                  "primary_identifiers": {"name": norm(name)},
                                  "attributes": {"canonical_name": _title(name), **attrs}})
        except Exception:
            logger.exception("Could not mirror an answer into entity memory")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"people": len(self.people), "roles": len(self.roles),
                    "terms": len(self.terms), "facts": len(self.facts),
                    "procedures": len(self.procedures)}


def _atomic_write(path: str, obj: Any):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _quarantine(path: str):
    try:
        os.replace(path, path + ".bad")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Detection: reading an utterance
# ----------------------------------------------------------------------------
@dataclass
class Mention:
    name: str
    strong: bool                    # said in a way that makes it plainly a person
    start: int = 0
    sentence_start: bool = False


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def extract_people(text: str, ignore: Iterable[str] = ()) -> List[Mention]:
    """Capitalised words said in a way that makes them somebody.

    Sentence-initial capitals prove nothing, so a name there needs something
    after it ("Sara called"); mid-sentence it needs something before it
    ("told Sara") or after it. A bare "Python" or "Paris" with no such cue is
    not a person, and the stop list catches the cued ones ("with Python").
    """
    ignore_n = {norm(x) for x in ignore}
    out: List[Mention] = []
    seen = set()
    for sent in _sentences(text):
        words = list(_CAPWORD.finditer(sent))
        i = 0
        while i < len(words):
            m = words[i]
            w = m.group(0)
            if not (len(w) >= 2 and w[0].isupper() and not w.isupper()
                    and w.lower() not in NOT_PERSON and norm(w) not in ignore_n):
                i += 1
                continue
            # a second capitalised word, adjacent, makes it a full name
            end_i = i
            full = w
            if (i + 1 < len(words) and re.fullmatch(r"\s", sent[m.end():words[i + 1].start()] or "x")
                    and words[i + 1].group(0)[0].isupper() and not words[i + 1].group(0).isupper()
                    and words[i + 1].group(0).lower() not in NOT_PERSON and len(words[i + 1].group(0)) >= 2):
                full = f"{w} {words[i + 1].group(0)}"
                end_i = i + 1
            first_start = m.start()
            last_end = words[end_i].end()
            sentence_start = first_start == len(sent) - len(sent.lstrip())
            prev = words[i - 1].group(0).lower() if i > 0 else ""
            prev2 = words[i - 2].group(0).lower() if i > 1 else ""
            nxt = words[end_i + 1].group(0).lower() if end_i + 1 < len(words) else ""
            poss = sent[last_end:last_end + 2] in ("'s", "’s")
            strong = (prev in _STRONG_BEFORE or (prev2, prev) in _PAIR_BEFORE
                      or nxt in _AFTER_VERBS)
            weak = prev in _WEAK_BEFORE
            pair = (nxt == "and" and end_i + 2 < len(words)
                    and words[end_i + 2].group(0)[:1].isupper())        # "Sara and Omar"
            weak = weak or pair
            if sentence_start and not strong and not pair:
                weak = False
            # "my sister Sara" / "Sara, my sister" are statements, handled by learn_statements
            if strong or weak or (poss and not sentence_start):
                key = norm(full)
                if key and key not in seen and not (norm(full) in ignore_n):
                    seen.add(key)
                    out.append(Mention(_title(full), strong=bool(strong), start=first_start,
                                       sentence_start=sentence_start))
            i = end_i + 1
    return out


_NAME = r"([A-Z][a-z]{1,19}(?:\s[A-Z][a-z]{1,19})?)"
_RE_MY_ROLE_NAME = re.compile(
    r"(?i:\bmy\s+(" + _ROLE_ALT + r")\b\s*(?:,|is|was|named|called|is named|is called)?\s*(?:named|called)?)\s*" + _NAME)
_RE_NAME_IS_MY_ROLE = re.compile(
    _NAME + r"(?i:\s*(?:,|is|'s|’s|was|are)?\s*(?:is\s+)?my\s+(" + _ROLE_ALT + r")\b)")
_RE_NAME_IS_POSS_ROLE = re.compile(
    _NAME + r"\s+(?:is|was)\s+" + _NAME + r"(?:'s|’s)\s+(" + _ROLE_ALT + r")\b")
_RE_NAME_AND_NAME_ARE = re.compile(
    _NAME + r"\s+and\s+" + _NAME + r"\s+(?:are|were)\s+(?:both\s+)?([a-z][a-z\- ]{2,30})")
_RE_MY_ROLE = re.compile(r"\bmy\s+(" + _ROLE_ALT + r")(?:'s|’s)?\b", re.I)
_RE_FACT_STATEMENT = re.compile(r"\bmy\s+(birthday|anniversary)\s+(?:is|was|falls)\s+(?:on\s+)?"
    r"((?=[^.?!]{0,38}(?:\d|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day))[^.?!]{3,40})", re.I)
_RE_QUOTED = re.compile(r"(?:\"|“)([^\"”]{2,30})(?:\"|”)|(?<![\w'])'([A-Za-z][^']{1,28})'(?![\w'])")
_RE_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_RE_PROJECT = re.compile(r"\b(?:project|app|tool|service|server|codename|repo|module|team)\s+([A-Z][A-Za-z0-9\-]{2,20})\b")
_RE_DEFINE = [
    re.compile(r"\b(?P<t>[A-Za-z0-9][\w\- ]{0,28}?)\s+(?:means|stands for|is short for|refers to|is an? acronym for)\s+(?P<d>[^.?!]{2,160})", re.I),
    re.compile(r"\bby\s+(?P<t>[A-Za-z0-9][\w\- ]{0,28}?),?\s+i\s+mean\s+(?P<d>[^.?!]{2,160})", re.I),
    re.compile(r"\b(?P<t>[A-Z][A-Za-z0-9\-]{1,20})\s+(?:is|are)\s+(?:an?|the|my|our)\s+(?P<d>[^.?!]{2,140})"),
]


def _role_of(word: str) -> str:
    return ROLE_CANON.get(word.lower(), "")


def _shouting(sent: str) -> bool:
    letters = [c for c in sent if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


# ----------------------------------------------------------------------------
# The wonderer
# ----------------------------------------------------------------------------
class Wonderer:
    """The queue, and the rules about asking."""

    def __init__(self, cfg: Optional[WonderingConfig] = None, kb: Optional[Knowledge] = None,
                 path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 scrub: Optional[Callable[[str], str]] = None):
        self.cfg = cfg or WonderingConfig()
        self.kb = kb or Knowledge()
        self.kb.clock = clock
        self.path = path
        self.clock = clock
        self.scrub = scrub or (lambda s: s)
        self.fits: List[FitFn] = [mood_fit, activity_fit]
        # Before asking what a term means, look it up (see lookup.py): a
        # function of the wonder returning {"guess", "question", "source"} or
        # None. A guess turns the question into one to confirm.
        self.lookup: Optional[Callable[[Wonder], Optional[Dict[str, Any]]]] = None
        self.lookup_status: Optional[Callable[[], Dict[str, Any]]] = None
        self._lock = threading.RLock()

        self.wonders: Dict[str, Wonder] = {}
        self.closed: Dict[str, Dict[str, Any]] = {}          # key -> {"s": status, "t": time}
        self.mentions: Dict[str, Dict[str, Any]] = {}        # key -> {"n": count, "t": last}
        self.subject_last: Dict[str, float] = {}             # norm subject -> last asked
        self.muted_subjects: Dict[str, float] = {}
        self.asked_times: Deque[float] = deque(maxlen=64)
        self.last_asked: float = 0.0
        self.paused_until: float = 0.0                       # "stop asking me things"
        self._batch_until: float = 0.0                       # "what are you wondering about?" is open
        self._batch_left: int = 0
        self.off: bool = False                               # said "never", or switched off by hand
        self.stats = {"detected": 0, "asked": 0, "answered": 0, "dismissed": 0,
                      "unclear": 0, "expired": 0, "learned": 0, "deduped": 0}
        self._dirty = False
        # Kinds of question the user keeps waving off are benched for a while
        # (backoff.py); asked for ("any questions?"), they are still offered.
        from .backoff import Backoff
        self.backoff = Backoff(clock=lambda: self.clock())
        self._load()

    # ==================================================================
    # persistence
    # ==================================================================
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("wonders", []):
                try:
                    w = Wonder(**{k: v for k, v in d.items() if k in Wonder.__dataclass_fields__})
                    self.wonders[w.id] = w
                except Exception:
                    continue
            self.closed = dict(raw.get("closed") or {})
            self.mentions = dict(raw.get("mentions") or {})
            self.subject_last = {k: float(v) for k, v in (raw.get("subject_last") or {}).items()}
            self.muted_subjects = {k: float(v) for k, v in (raw.get("muted_subjects") or {}).items()}
            self.asked_times = deque((float(t) for t in raw.get("asked_times", [])), maxlen=64)
            self.last_asked = float(raw.get("last_asked", 0.0))
            self.paused_until = float(raw.get("paused_until", 0.0))
            self.off = bool(raw.get("off", False))
            self.backoff.load(raw.get("backoff"))
            self.stats.update({k: int(v) for k, v in (raw.get("stats") or {}).items() if k in self.stats})
        except Exception:
            logger.exception("Wonder state unreadable; starting empty")
            _quarantine(self.path)
            self.wonders, self.closed = {}, {}

    def save(self):
        if not self.path:
            self._dirty = False
            return
        with self._lock:
            self._trim()
            terminal = sorted((w for w in self.wonders.values() if w.status in TERMINAL),
                              key=lambda w: w.closed_at)[-40:]
            keep = [w for w in self.wonders.values() if w.status in OPEN] + terminal
            try:
                _atomic_write(self.path, {
                    "version": 1,
                    "wonders": [w.public() for w in keep],
                    "closed": self.closed, "mentions": self.mentions,
                    "subject_last": self.subject_last, "muted_subjects": self.muted_subjects,
                    "asked_times": list(self.asked_times), "last_asked": self.last_asked,
                    "paused_until": self.paused_until, "off": self.off, "stats": self.stats,
                    "backoff": self.backoff.dump(),
                })
                self._dirty = False
            except Exception:
                logger.exception("Could not save wonder state")

    def _trim(self):
        for d, cap in ((self.closed, 2000), (self.mentions, 1000), (self.subject_last, 1000),
                       (self.muted_subjects, 500)):
            if len(d) > cap:
                def stamp(v):
                    return v.get("t", 0) if isinstance(v, dict) else float(v)
                for k in sorted(d, key=lambda k: stamp(d[k]))[:len(d) - cap]:
                    del d[k]
        for k in [k for k, w in self.wonders.items() if w.status in TERMINAL][:-60]:
            del self.wonders[k]

    def _touch(self):
        self._dirty = True
        self.save()

    # ==================================================================
    # switches
    # ==================================================================
    def active(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.cfg.enabled and not self.off and now >= self.paused_until

    def set_enabled(self, on: bool):
        with self._lock:
            self.off = not on
            if on:
                self.paused_until = 0.0
            self._touch()

    def snooze_all(self, seconds: Optional[float] = None, now: Optional[float] = None):
        now = self.clock() if now is None else now
        with self._lock:
            self.paused_until = now + (self.cfg.global_snooze_s if seconds is None else seconds)
            self._touch()

    # ==================================================================
    # the queue
    # ==================================================================
    def _is_closed(self, key: str, now: float) -> bool:
        c = self.closed.get(key)
        return bool(c and c.get("s") in FOREVER)

    def _subject_muted(self, subjects: Iterable[str]) -> bool:
        return any(s in self.muted_subjects for s in subjects)

    def _make(self, kind: str, subject: str, question: str, key: str, subjects: List[str],
              data: Dict[str, Any], evidence: str, source: str, now: float,
              strong: bool = True) -> Optional[Wonder]:
        """Add a wonder, or bump the one that is already there."""
        if not self.cfg.kinds.get(kind, True):
            return None
        if self._is_closed(key, now) or self._subject_muted(subjects):
            return None
        n = self.mentions.setdefault(key, {"n": 0, "t": now})
        n["n"] += 1
        n["t"] = now
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day not in n.setdefault("days", []):
            n["days"] = (n["days"] + [day])[-10:]
        need = 1 if strong else self.cfg.min_mentions_weak
        existing = next((w for w in self.wonders.values() if w.key == key and w.status in OPEN), None)
        if existing is not None:
            existing.mentions += 1
            existing.priority = self._priority(existing)
            self.stats["deduped"] += 1
            return None
        if n["n"] < need:
            return None
        w = Wonder(id=uuid.uuid4().hex[:8], kind=kind, subject=_clean(subject, 40), question=question,
                   key=key, subjects=subjects, created=now, mentions=n["n"], data=data, source=source,
                   evidence=(_clean(self.scrub(evidence), 80) if self.cfg.keep_evidence else ""))
        w.priority = self._priority(w)
        self.wonders[w.id] = w
        self.stats["detected"] += 1
        pend = [x for x in self.wonders.values() if x.status == "pending"]
        if len(pend) > self.cfg.max_pending:
            worst = min(pend, key=lambda x: (x.priority, -x.created))
            self._close(worst, "expired", now)
        logger.debug("wonder %s detected (%s)", w.id, kind)
        return w

    def _priority(self, w: Wonder) -> float:
        w.urgency = min(1.0, BASE_URGENCY.get(w.kind, 0.3) + 0.10 * min(max(w.mentions - 1, 0), 3))
        return min(1.0, BASE_PRIORITY.get(w.kind, 0.5) + 0.08 * min(max(w.mentions - 1, 0), 3))

    def add_fit(self, fn: FitFn):
        """Register another opinion on whether a moment suits a question."""
        self.fits.append(fn)

    def fit(self, w: Wonder, env: Environment) -> float:
        f = 1.0
        for fn in self.fits:
            try:
                f *= max(0.0, float(fn(w, env)))
            except Exception:
                logger.exception("A moment-fit scorer failed; ignoring it")
        return f

    def _close(self, w: Wonder, status: str, now: float):
        w.status = status
        w.closed_at = now
        w.evidence = ""
        if status in FOREVER:
            self.closed[w.key] = {"s": status, "t": now}
        self.stats[{"answered": "answered", "dismissed": "dismissed", "unclear": "unclear",
                    "expired": "expired"}[status]] += 1

    # ==================================================================
    # reading what the user said
    # ==================================================================
    def hears(self, source: str) -> bool:
        s = (source or "").lower()
        if s.startswith("sim:"):
            s = s[4:]
        return s in {x.lower() for x in self.cfg.sources}

    def observe(self, text: str, source: str = "console", now: Optional[float] = None) -> Reply:
        """Learn what the sentence states and note the gaps it leaves.

        Returns a Reply whose `learned` lists what was learned outright
        (sentences for the concept network); `consumed` is always False --
        the utterance is still the user's to have answered.
        """
        now = self.clock() if now is None else now
        text = _as_text(text)
        rep = Reply()
        if not text or not self.cfg.enabled or self.off:
            return rep
        with self._lock:
            before = (len(self.wonders), self.stats["detected"], self.stats["learned"])
            for sent in _sentences(text)[:12]:
                if _shouting(sent):
                    continue
                self._stated_roles = set()
                rep.learned.extend(self._learn_statements(sent, now))
                self._detect(sent, source, now)
            if (len(self.wonders), self.stats["detected"], self.stats["learned"]) != before or rep.learned:
                self._touch()
        return rep

    # -- statements that close gaps ----------------------------------------
    def _learn_statements(self, sent: str, now: float) -> List[str]:
        out: List[str] = []

        def person(name: str, role: str):
            name = _title(name)
            if norm(name) in {norm(x) for x in self.cfg.ignore_names}:
                return
            canon = ROLE_CANON.get(role, role)
            self._stated_roles.add(canon)
            prev = self.kb.relation_of(name)
            if relations_conflict(prev, canon):
                self._conflict("relation", name, prev, canon, sent, now)
                return
            holder = self.kb.role_name(canon) if canon in REPLACEABLE else None
            if holder and norm(holder) != norm(name):
                self._conflict("role", name, holder, name, sent, now, role=canon)
                return
            self.kb.store_relation(name, role=role)
            self._resolve_open(("person", f"person:{norm(name)}"), now)
            self._resolve_open(("role", f"role:{ROLE_CANON.get(role, role)}"), now)
            self.stats["learned"] += 1
            out.append(f"{name} is the user's {ROLE_CANON.get(role, role)}.")

        for m in _RE_MY_ROLE_NAME.finditer(sent):
            name = m.group(2)
            if name.split()[0].lower() not in NOT_PERSON and name.split()[0].lower() not in ROLE_CANON:
                person(name, m.group(1).lower())
        for m in _RE_NAME_IS_MY_ROLE.finditer(sent):
            if m.group(1).split()[0].lower() not in NOT_PERSON:
                person(m.group(1), m.group(2).lower())
        for m in _RE_NAME_IS_POSS_ROLE.finditer(sent):
            a, b, role = m.group(1), m.group(2), m.group(3).lower()
            if a.split()[0].lower() in NOT_PERSON or b.split()[0].lower() in NOT_PERSON:
                continue
            label = f"{_title(a)} is {_title(b)}'s {ROLE_CANON.get(role, role)}"
            self.kb.store_between(a, b, label)
            self._resolve_open(("between", self._between_key(a, b)), now)
            self.stats["learned"] += 1
            out.append(label + ".")
        for m in _RE_NAME_AND_NAME_ARE.finditer(sent):
            a, b, tail = m.group(1), m.group(2), m.group(3).strip()
            if a.split()[0].lower() in NOT_PERSON or b.split()[0].lower() in NOT_PERSON:
                continue
            label = self._between_label(tail)
            if label:
                self.kb.store_between(a, b, label)
                self._resolve_open(("between", self._between_key(a, b)), now)
                self.stats["learned"] += 1
                out.append(f"{_title(a)} and {_title(b)} are {label}.")
        for m in _RE_FACT_STATEMENT.finditer(sent):
            k, v = m.group(1).lower(), _clean(m.group(2), 40)
            old = self.kb.fact(k)
            if old and norm(old) != norm(v):
                self._conflict("fact", k, old, v, sent, now)
                continue
            self.kb.store_fact(k, v)
            self._resolve_open(("fact", f"fact:{k}"), now)
            self.stats["learned"] += 1
            out.append(f"The user's {k} is {v}.")
        # "X means Y" -- only for terms ASH is, or could be, wondering about
        for rx in _RE_DEFINE:
            for m in rx.finditer(sent):
                term, d = _clean(m.group("t"), 30), _clean(m.group("d"), 160)
                if not term or not d or term.lower() in NOT_PERSON or len(term.split()) > 3:
                    continue
                if self.kb.term_known(term):
                    old = self.kb.terms[norm(term)].get("definition", "")
                    if definitions_differ(old, d):
                        self._conflict("term", term, old, d, sent, now)
                    continue
                if not self._term_relevant(term):
                    continue
                self.kb.store_term(term, d)
                self._resolve_open(("term", f"term:{norm(term)}"), now)
                self.stats["learned"] += 1
                out.append(f"{term} means {d}.")
        return out

    def _conflict(self, what: str, subject: str, old: str, new: str, sent: str, now: float,
                  role: str = ""):
        """Two things said that cannot both be true: keep the old one and ask.
        Nothing is overwritten until the user says which."""
        old_c, new_c = _clean(old, 60), _clean(new, 60)
        if what == "relation":
            q = f"Earlier you said {subject} is your {old_c}, but just now your {new_c}. Which is it?"
        elif what == "role":
            q = f"Is {new_c} your {role} now, instead of {old_c}?"
        elif what == "term":
            q = f"You told me {subject} means {old_c}, and now {new_c}. Which is right?"
        else:
            q = f"I have your {subject} as {old_c}, but you just said {new_c}. Which is right?"
        key = f"conflict:{what}:{norm(subject)}:{norm(new_c)}"
        self._make("conflict", subject, q, key, [norm(subject)],
                   {"what": what, "subject": subject, "old": old_c, "new": new_c,
                    "options": [old_c, new_c], "role": role}, sent, "console", now)

    def _maybe_stale(self, name: str, sent: str, source: str, now: float):
        """A relation that can change, not heard confirmed for a long while,
        is checked when the person comes up again."""
        rel = self.kb.relation_of(name)
        age = self.kb.relation_age(name, now)
        if rel in CHANGEABLE and age is not None and age > self.cfg.stale_after_s:
            t_set = int(now - age)
            self._make("stale", name, f"Is {name} still your {rel}?", f"stale:{norm(name)}:{rel}:{t_set}",
                       [norm(name)], {"name": name, "role": rel}, sent, source, now)

    def _term_relevant(self, term: str) -> bool:
        """Only learn a definition for something that was wondered about or is
        plainly a coined term; otherwise every "Python is a snake" becomes
        a memory."""
        k = f"term:{norm(term)}"
        if any(w.key == k for w in self.wonders.values()) or k in self.closed or k in self.mentions:
            return True
        return bool(_RE_ACRONYM.fullmatch(term.strip()))

    def _resolve_open(self, match: Tuple[str, str], now: float):
        kind, key = match
        for w in list(self.wonders.values()):
            if w.kind == kind and w.key == key and w.status in OPEN:
                self._close(w, "answered", now)
        if key not in self.closed:
            self.closed[key] = {"s": "answered", "t": now}

    @staticmethod
    def _between_key(a: str, b: str) -> str:
        return "between:" + "|".join(sorted((norm(a), norm(b))))

    @staticmethod
    def _between_label(tail: str) -> str:
        for rx, label in BETWEEN_CANON:
            if re.search(rx, tail, re.I):
                return label
        return ""

    # -- gaps --------------------------------------------------------------
    def _detect(self, sent: str, source: str, now: float):
        ign = self.cfg.ignore_names
        people = extract_people(sent, ign)

        # an unknown person
        named = []
        for pm in people:
            first = pm.name.split()[0]
            cands = self.kb.candidates(first) if " " not in pm.name else []
            if len(cands) >= 2:
                key = f"ambiguous:{norm(first)}"
                self._make("ambiguous", first, f"Which {first} do you mean: " + self._or_list(cands) + "?",
                           key, [norm(first)], {"options": cands, "bare": first}, sent, source, now)
                continue
            if self.kb.person_known(pm.name):
                named.append(pm.name)
                self._maybe_stale(pm.name, sent, source, now)
                self.kb.touch(pm.name, now)
                continue
            named.append(pm.name)
            key = f"person:{norm(pm.name)}"
            ctx = context_of(sent)
            if ctx:
                m = self.mentions.setdefault(key, {"n": 0, "t": now})
                acc = m.setdefault("ctx", {})
                for k2, n2 in ctx.items():
                    acc[k2] = acc.get(k2, 0) + n2
            self._make("person", pm.name, f"Who is {pm.name} to you?", key, [norm(pm.name)],
                       {"name": pm.name}, sent, source, now, strong=pm.strong)

        # two people with no known relation
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                a, b = named[i], named[j]
                if norm(a) == norm(b) or self.kb.between_known(a, b):
                    continue
                self._make("between", f"{a} and {b}", f"How do {a} and {b} know each other?",
                           self._between_key(a, b), [norm(a), norm(b)], {"a": a, "b": b},
                           sent, source, now, strong=False)

        # a role with no name
        for m in _RE_MY_ROLE.finditer(sent):
            role = ROLE_CANON.get(m.group(1).lower(), "")
            if not role or role not in UNIQUE_ROLES:
                continue
            after = sent[m.end():m.end() + 40]
            if re.match(r"\s*,?\s*(?:is\s+|named\s+|called\s+)?(?:named\s+|called\s+)?[A-Z][a-z]", after):
                continue                                    # a name follows: learned above
            if self.kb.role_name(role) or role in getattr(self, "_stated_roles", ()):
                continue                                    # known, or named in this very sentence
            self._make("role", f"your {role}", f"What's your {role}'s name?", f"role:{role}",
                       [role], {"role": role}, sent, source, now)

        # a missing fact
        for m in re.finditer(r"\bmy\s+(birthday|anniversary)\b", sent, re.I):
            k = m.group(1).lower()
            if self.kb.fact(k) or _RE_FACT_STATEMENT.search(sent):
                continue
            self._make("fact", f"your {k}", FACT_NOUNS[k], f"fact:{k}", [k], {"fact": k},
                       sent, source, now)

        # terms
        terms: List[Tuple[str, str, bool]] = []                       # (term, how, strong)
        for m in _RE_QUOTED.finditer(sent):
            t = _clean(m.group(1) or m.group(2), 30)
            if t and len(t.split()) <= 3 and not re.fullmatch(r"[\d\W]+", t):
                terms.append((t, "quoted", True))
        for m in _RE_PROJECT.finditer(sent):
            terms.append((m.group(1), "named", True))
        for m in _RE_ACRONYM.finditer(sent):
            a = m.group(1)
            if a not in _COMMON_ACRONYMS and a.lower() not in NOT_PERSON:
                terms.append((a, "acronym", False))
        seen_t = set()
        for t, how, strong in terms:
            nt = norm(t)
            if (not nt or nt in seen_t or self.kb.term_known(t)
                    or nt in {norm(p) for p in named} or t.lower() in NOT_PERSON
                    or nt in {norm(x) for x in ign}):
                continue
            seen_t.add(nt)
            q = (f"What does {t} stand for?" if how == "acronym" else f"What do you mean by \"{t}\"?")
            self._make("term", t, q, f"term:{nt}", [nt], {"term": t, "how": how}, sent, source, now,
                       strong=strong)

    def raise_wonder(self, kind: str, subject: str, question: str, key: str,
                     data: Optional[Dict[str, Any]] = None, now: Optional[float] = None,
                     source: str = "activity") -> Optional[Wonder]:
        """Add a wonder from something other than a sentence the user said --
        what they have been doing, say. Everything that makes a question
        worth asking at a good moment (dedupe, rate caps, cool-downs, never
        again once answered) applies exactly as for the sentence detectors."""
        now = self.clock() if now is None else now
        if not self.cfg.enabled or self.off:
            return None
        with self._lock:
            w = self._make(kind, subject, question, key, [norm(subject)], dict(data or {}),
                           "", source, now)
            if w is not None:
                self._touch()
            return w

    @staticmethod
    def _or_list(items: List[str]) -> str:
        items = items[:4]
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]

    def note_ambiguous_intent(self, text: str, options: List[str], now: Optional[float] = None,
                              source: str = "console") -> Optional[Wonder]:
        """The router could not tell what was meant. `options` are the two
        readings worth putting to the user (human wording)."""
        now = self.clock() if now is None else now
        opts = list(dict.fromkeys(_clean(o, 40) for o in options if o))[:3]
        if len(opts) < 2 or not self.cfg.enabled or self.off:
            return None
        with self._lock:
            key = f"intent:{norm(text)[:60]}"
            w = self._make("intent", opts[0], f"Did you mean {self._or_list(opts)}?", key,
                           [norm(text)[:40]], {"options": opts}, text, source, now)
            if w:
                self._touch()
            return w

    # ==================================================================
    # reading the reply
    # ==================================================================
    def awaiting(self, now: Optional[float] = None) -> Optional[Wonder]:
        """The wonder whose answer the next utterance would be."""
        now = self.clock() if now is None else now
        cands = [w for w in self.wonders.values()
                 if w.status == "asked" and now - w.asked_at <= self.cfg.answer_window_s]
        return max(cands, key=lambda w: w.asked_at) if cands else None

    def _late(self, text: str, now: float) -> Optional[Wonder]:
        """An open question asked a while ago, which this reply names."""
        t = norm(text)
        for w in sorted((w for w in self.wonders.values() if w.status == "asked"),
                        key=lambda w: -w.asked_at):
            if now - w.asked_at > self.cfg.late_answer_s:
                continue
            if w.kind in ("activity", "conflict", "stale", "routine"):
                # Only the next reply answers one of these. A later "open
                # brain.py" names the subject without being an answer to
                # "what is brain.py for?".
                continue
            if any(s and re.search(r"\b" + re.escape(s) + r"\b", t) for s in w.subjects):
                return w
        return None

    def hear(self, text: str, now: Optional[float] = None, addressed: bool = False,
             dismissed: bool = False) -> Reply:
        """Read a user utterance as an answer to a question, or as a request
        about the questions themselves. Safe to call with anything."""
        now = self.clock() if now is None else now
        text = _as_text(text).strip()
        with self._lock:
            try:
                return self._hear(text, now, addressed, dismissed)
            except Exception:
                logger.exception("Reading a reply failed")
                return Reply()

    def _hear(self, text: str, now: float, addressed: bool, dismissed: bool) -> Reply:
        if not self.cfg.enabled:
            return Reply()
        aw = self.awaiting(now)
        recent = aw or self._last_asked_within(now, 30 * 60)

        # the button
        if dismissed:
            if aw:
                return self._defer(aw, now)
            return Reply()

        # "start asking again"
        if _RESUME.search(text) and (self.off or self.paused_until > now):
            self.set_enabled(True)
            return Reply(True, "Okay, I'll ask when something puzzles me.", "resumed")

        # "stop asking me things" / "stop asking about Sara" / "stop asking about that"
        if text and len(text.split()) <= 14:
            ms = _STOP_ABOUT.search(text)
            if ms:
                subj = _clean(ms.group("subj") or "", 40)
                if subj and not _THAT.fullmatch(subj.strip()) and norm(subj):
                    for s in {norm(subj)} | {norm(x) for x in subj.split() if len(x) > 2}:
                        self.muted_subjects[s] = now
                    for w in list(self.wonders.values()):
                        if w.status in OPEN and (norm(subj) in w.subjects or any(x in w.subjects for x in norm(subj).split())):
                            self._close(w, "dismissed", now)
                    self._touch()
                    return Reply(True, f"Okay, I won't ask about {subj} again.", "muted")
                if recent is not None and not _STOP_ALL.search(text):
                    return self._dismiss(recent, now, mute=True)
            if _STOP_ALL.search(text) or (ms and recent is None):
                forever = bool(re.search(r"\b(?:never|ever again|anymore|any more|permanently)\b", text, re.I))
                if forever:
                    self.set_enabled(False)
                    return Reply(True, "Okay, I won't ask you questions. Say so if you change your mind.", "snoozed")
                self.snooze_all(now=now)
                if recent is not None and recent.status in OPEN:
                    self._close(recent, "dismissed", now)
                    self._touch()
                return Reply(True, "Okay, no more questions for a while.", "snoozed")

        if text and len(text.split()) <= 12 and _ASK_ME.search(text) and not self.off:
            return self._open_batch(now)
        if now <= self._batch_until and _BATCH_END.match(text or ""):
            self._batch_until = 0.0
            if aw is not None:
                self._defer(aw, now)
            return Reply(True, "Okay, that's enough for now.", "batch_end")

        rep = self._hear_answer(text, now, aw)
        if rep.consumed and not rep.outcome.startswith("batch") and aw is not None:
            rep = self._batch_next(rep, now)
        return rep

    def _hear_answer(self, text: str, now: float, aw: Optional[Wonder]) -> Reply:
        w = aw
        late = False
        if w is None:
            w = self._late(text, now)
            late = w is not None
        if w is None:
            return Reply()

        # "not now"
        if _NOT_NOW.search(text) and len(text.split()) <= 8 and not late:
            return self._defer(w, now)

        # "I don't know" / "never mind"
        if len(text.split()) <= 8 and _DONT_KNOW.search(text) and not late:
            self._close(w, "dismissed", now)
            self._touch()
            return Reply(True, "Okay, never mind.", "dismissed", w.id)

        # not an answer at all (a question or a command)
        if not late and (text.endswith("?") or _NON_ANSWER_START.match(text)):
            if re.match(r"^\s*(?:no|nope|nah)\W*$", text, re.I):
                self._close(w, "dismissed", now)
                self._touch()
                return Reply(True, "Okay, never mind.", "dismissed", w.id)
            return Reply()

        value = self._parse_answer(w, text)
        if value is not None:
            return self._answered(w, value, now)

        # unclear: one follow-up, only for something short that looks like an attempt
        if late or len(text.split()) > 6 or "?" in text:
            return Reply()
        if not w.followup_used:
            w.followup_used = True
            w.asked_at = now
            self._touch()
            return Reply(True, self._followup(w), "followup", w.id)
        self._close(w, "unclear", now)
        self._touch()
        return Reply(True, "No worries, I'll leave it.", "unclear", w.id)

    def _last_asked_within(self, now: float, secs: float) -> Optional[Wonder]:
        c = [w for w in self.wonders.values() if w.asked_at and now - w.asked_at <= secs and w.status in OPEN]
        return max(c, key=lambda w: w.asked_at) if c else None

    def _defer(self, w: Wonder, now: float) -> Reply:
        w.deferrals += 1
        if w.deferrals > self.cfg.max_deferrals:
            self._close(w, "dismissed", now)
            self._touch()
            return Reply(True, "Okay, I'll drop it.", "dismissed", w.id)
        w.status = "pending"
        w.not_before = now + max(self.cfg.reask_after_s / 2, self.cfg.min_gap_s)
        self._touch()
        return Reply(True, "Okay, I'll check back later.", "deferred", w.id)

    def _dismiss(self, w: Wonder, now: float, mute: bool = False) -> Reply:
        self._close(w, "dismissed", now)
        if mute:
            for s in w.subjects:
                self.muted_subjects[s] = now
        self._touch()
        return Reply(True, "Okay, I won't ask about that again.", "muted" if mute else "dismissed", w.id)

    def _answered(self, w: Wonder, value: Dict[str, Any], now: float) -> Reply:
        sentence = self._store(w, value)
        self._close(w, "answered", now)
        self.stats["learned"] += 1
        self._touch()
        return Reply(True, value.get("ack", "Got it."), "answered", w.id, [sentence] if sentence else [])

    # -- parsing answers --------------------------------------------------
    @staticmethod
    def _strip_lead(text: str) -> str:
        t = text.strip().rstrip(".!")
        t = re.sub(r"^(?:well|um+|uh+|oh|so|yeah|yes|yep|ah|hmm+)[,\s]+", "", t, flags=re.I)
        t = re.sub(r"^(?:(?:his|her|their|its)\s+name(?:'s| is)|it'?s|it is|that'?s|that is|he'?s|she'?s|they'?re|"
                   r"they are|he is|she is|(?:he|she|they)(?:'s| is| are)? (?:called|named)|(?:called|named)|"
                   r"his name'?s|her name'?s|its name is|i think (?:it'?s|its|that'?s))\s+", "", t, flags=re.I)
        return t.strip(" ,.")

    def _parse_name(self, text: str) -> str:
        t = self._strip_lead(text)
        m = re.match(r"^" + _NAME + r"\b", t)
        if m and m.group(1).split()[0].lower() not in NOT_PERSON:
            return m.group(1)
        parts = re.split(r"[,;]| and | but | - ", t)[0].split()
        if len(parts) == 1 and re.fullmatch(r"[A-Za-z][A-Za-z'\-]{2,19}", parts[0]):
            p = parts[0].lower()
            if p in NOT_PERSON or p in ROLE_CANON or p in _FILLER or _DONT_KNOW.search(t):
                return ""
            return _title(p)
        # a late reply that carries the name inside a sentence: "my boss Dana is nice"
        names = [m.group(0) for m in re.finditer(r"\b[A-Z][a-z]{1,19}\b", t)
                 if m.group(0).lower() not in NOT_PERSON and m.group(0).lower() not in ROLE_CANON]
        if len(set(names)) == 1 and not _DONT_KNOW.search(t):
            return names[0]
        return ""

    def _parse_role(self, text: str) -> str:
        t = text.lower()
        m = re.search(r"\b(" + _ROLE_ALT + r")s?\b", t)
        return ROLE_CANON[m.group(1)] if m else ""

    def _parse_choice(self, text: str, options: List[str]) -> int:
        t = norm(text)
        words = t.split()
        for wd in words:
            if wd in _ORDINALS and _ORDINALS[wd] < len(options):
                return _ORDINALS[wd]
        hits = []
        for i, o in enumerate(options):
            toks = [x for x in norm(o).split() if len(x) >= 3]
            if toks and any(re.search(r"\b" + re.escape(x) + r"\b", t) for x in toks):
                hits.append(i)
        return hits[0] if len(hits) == 1 else -1

    @staticmethod
    def _parse_distinct(text: str, options: List[str]) -> int:
        """Which option the reply names, by the words that tell the options
        apart ("14" between "12 March" and "14 March"). -1 if none or both."""
        toks = [set(norm(o).split()) for o in options]
        common = set.intersection(*toks) if toks else set()
        said = set(norm(text).split())
        hits = [i for i, t in enumerate(toks)
                if any(x in said for x in t - common if len(x) >= 3 or x.isdigit())]
        return hits[0] if len(hits) == 1 else -1

    def _parse_answer(self, w: Wonder, text: str) -> Optional[Dict[str, Any]]:
        k, d = w.kind, w.data
        if k == "role":
            name = self._parse_name(text)
            if not name:
                return None
            return {"name": name, "ack": f"Got it -- {name} is your {d['role']}."}
        if k == "person" and d.get("guess"):
            g = d["guess"]
            role = self._parse_role(text)
            if role:
                return {"role": role, "ack": f"Got it -- {w.subject} is your {role}."}
            if _YES.match(text):
                if g == "family":
                    return {"desc": "family", "ack": f"Got it -- {w.subject} is family."}
                return {"role": g, "ack": f"Got it -- {w.subject} is your {g}."}
            m = _NO.match(text)
            if m:
                rest = self._strip_lead(m.group("rest") or "")
                rest = re.sub(r"^(?:(?:a|an|the|just|my|someone|somebody|one of)\s+)+", "", rest, flags=re.I)
                if 1 <= len(rest.split()) <= 7 and re.fullmatch(r"[\w' \-,]+", rest) and not _DONT_KNOW.search(rest):
                    return {"desc": rest, "ack": f"Got it -- {w.subject} is {_clean(rest, 50)}."}
                return None
        if k == "person":
            role = self._parse_role(text)
            if role:
                return {"role": role, "ack": f"Got it -- {w.subject} is your {role}."}
            t = self._strip_lead(text)
            t = re.sub(r"^(?:(?:a|an|the|just|my|someone|somebody|one of)\s+)+", "", t, flags=re.I)
            if 1 <= len(t.split()) <= 7 and re.fullmatch(r"[\w' \-,]+", t) and not _DONT_KNOW.search(t):
                return {"desc": t, "ack": f"Got it -- {w.subject} is {_clean(t, 50)}."}
            return None
        if k == "between":
            m = re.search(_NAME + r"\s+is\s+" + _NAME + r"(?:'s|’s)\s+(" + _ROLE_ALT + r")\b", text)
            if m:
                label = f"{m.group(1)} is {m.group(2)}'s {ROLE_CANON[m.group(3).lower()]}"
                return {"label": label, "ack": "Got it."}
            label = self._between_label(text)
            if label:
                return {"label": label, "ack": f"Got it -- {w.data['a']} and {w.data['b']} are {label}."}
            return None
        if k in ("term", "activity"):
            g = d.get("guess")
            if g:
                if _YES.match(text):
                    return {"definition": g, "ack": f"Great, noted: {w.subject} is {_clean(g, 60)}."}
                m = _NO.match(text)
                if m:
                    rest = self._strip_lead(m.group("rest") or "")
                    rest = re.sub(r"^(?:stands for|means|mean|is short for|refers to|is)\s+", "", rest, flags=re.I)
                    if len(rest) >= 2 and not _DONT_KNOW.search(rest) and rest.lower().strip(" .!") not in _FILLER:
                        return {"definition": _clean(rest, 200), "ack": f"Got it -- {w.subject} is {_clean(rest, 60)}."}
                    return None
            t = self._strip_lead(text)
            t = re.sub(r"^(?:stands for|means|mean|is short for|refers to|is an? acronym for|is)\s+", "", t, flags=re.I)
            n = len(t.split())
            if k == "activity" and (len(t) < 3 or t.lower().strip(" .!?") in _FILLER):
                return None              # "hm", "ok", a bare "yes": not a definition
            if 1 <= n <= 30 and len(t) >= 2 and not _DONT_KNOW.search(t):
                return {"definition": _clean(t, 200), "ack": f"Got it -- {w.subject} is {_clean(t, 60)}."}
            return None
        if k == "fact":
            if len(text.split()) <= 10 and _DATEISH.search(text):
                v = _clean(self._strip_lead(text), 40)
                return {"value": v, "ack": f"Got it -- {w.subject} is {v}."}
            return None
        if k == "routine":
            if _NO.match(text) and not _YES.match(text):
                return {"decline": True, "ack": "Okay, I won't."}
            m = re.search(r"\b(?:call it|name it|call that|it'?s (?:my|the|our)|as)\s+(?:my\s+|the\s+|a\s+)?"
                          r"([\w' \-]{2,30})", text, re.I)
            name = m.group(1) if m else ""
            if not name and _YES.match(text):
                rest = re.sub(r"^\s*(?:yes|yeah|yep|yup|sure|right|ok(?:ay)?|please|go ahead)\b[\s,.!:-]*", "",
                              text, flags=re.I).strip(" .!")
                rest = re.sub(r"^(?:please|sure)\b[\s,]*", "", rest, flags=re.I)
                if rest and len(rest.split()) <= 4 and not re.search(r"\b(?:that|this|it|so|please)\b$", rest, re.I):
                    name = rest
            if not name and not _YES.match(text) and not re.search(r"\b(?:sure|please|go ahead|ok(?:ay)?)\b",
                                                                   text, re.I):
                return None
            name = _clean(re.sub(r"^(?:my|the|a)\s+", "", name or "usual start", flags=re.I), 30) or "usual start"
            return {"name": name, "ack": f"Got it -- I'll remember \"{name}\": " + ", then ".join(d["steps"]) + "."}
        if k == "conflict":
            if d.get("what") == "role":
                if _YES.match(text):
                    return {"choice": d["new"], "ack": f"Got it -- {d['new']} is your {d['role']} now."}
                if _NO.match(text) or re.search(r"\bstill\b", text, re.I):
                    return {"choice": d["old"], "ack": f"Okay -- {d['old']} is still your {d['role']}."}
                return None
            if d.get("what") == "relation" and re.search(r"\bboth\b", text, re.I):
                return {"choice": d["new"], "both": True, "ack": "Got it, both."}
            opts = d.get("options") or []
            i = self._parse_distinct(text, opts)
            if i < 0:
                i = self._parse_choice(text, opts)
            if i < 0:
                # "the second one" / "the new one" / "the old one"
                if re.search(r"\b(?:new|latest|second|last|now)\b", text, re.I):
                    i = 1
                elif re.search(r"\b(?:old|first|earlier|before|original)\b", text, re.I):
                    i = 0
            if i < 0:
                return None
            return {"choice": opts[i], "ack": f"Got it -- {opts[i]}."}
        if k == "stale":
            role = self._parse_role(text)
            if role and role != d["role"]:
                return {"role": role, "ack": f"Got it -- {d['name']} is your {role} now."}
            if _YES.match(text) or re.search(r"\bstill\b", text, re.I) or role == d["role"]:
                return {"confirm": True, "ack": "Good to know, thanks."}
            retire = {"retire": True, "ack": f"Okay, I'll remember {d['name']} as your former {d['role']}."}
            if re.search(r"\b(?:anymore|any more|no longer|left|quit|moved (?:out|away|on)|broke up|"
                         r"retired|fired|changed jobs?|new job)\b", text, re.I):
                return retire
            m = _NO.match(text)
            if m:
                rest = m.group("rest").strip(" .!")
                if rest and not _DONT_KNOW.search(rest) and len(rest.split()) <= 8:
                    return {"desc": _clean(self._strip_lead(rest), 60), "ack": "Got it."}
                if w.followup_used:
                    return retire
            return None
        if k in ("ambiguous", "intent"):
            opts = d.get("options") or []
            i = self._parse_choice(text, opts)
            if i < 0:
                return None
            return {"choice": opts[i], "ack": f"Got it -- {opts[i]}."}
        return None

    def _store(self, w: Wonder, v: Dict[str, Any]) -> str:
        k, d = w.kind, w.data
        if k == "role":
            self.kb.store_role(d["role"], v["name"])
            return f"{v['name']} is the user's {d['role']}."
        if k == "person":
            if v.get("role"):
                self.kb.store_relation(d["name"], role=v["role"])
                return f"{d['name']} is the user's {v['role']}."
            self.kb.store_relation(d["name"], desc=v["desc"])
            return f"{d['name']} is {v['desc']}."
        if k == "between":
            self.kb.store_between(d["a"], d["b"], v["label"])
            return f"{d['a']} and {d['b']}: {v['label']}."
        if k in ("term", "activity"):
            self.kb.store_term(d["term"], v["definition"])
            return f"{d['term']} means {v['definition']}."
        if k == "fact":
            self.kb.store_fact(d["fact"], v["value"])
            return f"The user's {d['fact']} is {v['value']}."
        if k == "ambiguous":
            self.kb.store_alias(d["bare"], v["choice"])
            return f"When the user says {d['bare']} they mean {v['choice']}."
        if k == "intent":
            self.kb.store_fact("intent:" + norm(w.subject)[:30], v["choice"])
            return ""
        if k == "conflict":
            what, subj, choice = d["what"], d["subject"], v["choice"]
            if what == "relation":
                if v.get("both"):
                    self.kb.store_relation(subj, role=d["new"], desc=f"also your {d['old']}")
                    return f"{subj} is the user's {d['new']} and also their {d['old']}."
                self.kb.store_relation(subj, role=choice)
                return f"{subj} is the user's {choice}."
            if what == "role":
                if choice == d["new"]:
                    self.kb.retire(d["old"], d["role"])
                    self.kb.store_relation(d["new"], role=d["role"])
                    return f"{d['new']} is the user's {d['role']} now, not {d['old']}."
                self.kb.confirm(d["old"], self.clock())
                return ""
            if what == "term":
                self.kb.store_term(subj, choice)
                return f"{subj} means {choice}."
            self.kb.store_fact(subj, choice)
            return f"The user's {subj} is {choice}."
        if k == "routine":
            if v.get("decline"):
                return ""
            self.kb.store_procedure(v["name"], d["steps"])
            return f"The user's routine \"{v['name']}\" is opening " + ", then ".join(d["steps"]) + "."
        if k == "stale":
            name, role = d["name"], d["role"]
            if v.get("confirm"):
                self.kb.confirm(name, self.clock())
                return f"{name} is still the user's {role}."
            if v.get("role"):
                self.kb.store_relation(name, role=v["role"])
                return f"{name} is the user's {v['role']} now."
            if v.get("desc"):
                self.kb.retire(name, role)
                self.kb.store_relation(name, desc=v["desc"])
                return f"{name} is no longer the user's {role}: {v['desc']}."
            self.kb.retire(name, role)
            return f"{name} is no longer the user's {role}."
        return ""

    def _followup(self, w: Wonder) -> str:
        k, d = w.kind, w.data
        if k == "role":
            return f"Sorry, I didn't catch the name -- what do you call your {d['role']}?"
        if k == "person" and d.get("guess"):
            return f"Okay -- who is {w.subject} to you, then?"
        if k == "person":
            return f"Sorry, I didn't catch that. Is {w.subject} family, a friend, or a colleague?"
        if k == "between":
            return f"Sorry, I missed that. Are {d['a']} and {d['b']} family, friends, or colleagues?"
        if k in ("term", "activity"):
            if d.get("guess"):
                return f"Okay -- what does {w.subject} mean, then?"
            return f"Sorry, I didn't get that. In a few words, what is {w.subject}?"
        if k == "fact":
            return f"Sorry, I missed the date. When is {w.subject}?"
        if k == "stale":
            return f"Okay -- what is {d['name']} to you now?"
        if k == "routine":
            return "Sorry -- shall I remember it, yes or no?"
        if k == "conflict" and d.get("what") == "role":
            return f"Sorry -- is {d['new']} your {d['role']} now, yes or no?"
        opts = d.get("options") or []
        return "Sorry, which one: " + self._or_list(opts) + "?"

    # ==================================================================
    # asking
    # ==================================================================
    def housekeeping(self, now: Optional[float] = None):
        """Expire the old; let go of questions that went unanswered."""
        now = self.clock() if now is None else now
        with self._lock:
            changed = False
            for w in list(self.wonders.values()):
                if w.status == "pending" and now - w.created > self.cfg.expire_s:
                    self._close(w, "expired", now)
                    changed = True
                elif w.status == "asked" and now - w.asked_at > self.cfg.late_answer_s:
                    # asked and ignored for a day: one more try later, then let go
                    if w.asked_count <= self.cfg.reask_unanswered:
                        w.status = "pending"
                        w.not_before = max(w.not_before, w.asked_at + self.cfg.reask_after_s)
                    else:
                        self._close(w, "expired", now)
                    changed = True
                elif w.status == "asked" and now - w.asked_at > self.cfg.answer_window_s and w.asked_count <= self.cfg.reask_unanswered \
                        and now - w.asked_at > self.cfg.reask_after_s:
                    w.status = "pending"
                    w.not_before = w.asked_at + self.cfg.reask_after_s
                    changed = True
            if changed:
                self._touch()

    def can_ask(self, env: Environment, now: Optional[float] = None) -> Tuple[bool, str]:
        """The rules about WHEN, as one inspectable verdict."""
        now = self.clock() if now is None else now
        c = self.cfg
        if not c.enabled or self.off:
            return False, "wondering is off"
        if now < self.paused_until:
            return False, "told to stop asking for now"
        if env.muted:
            return False, "muted"
        if env.paused:
            return False, "privacy pause"
        if env.quiet_hours:
            return False, "quiet hours"
        if env.plan_active:
            return False, "following a plan"
        if env.speaking:
            return False, "already speaking"
        if env.conversation_open:
            return False, "mid-conversation"
        if env.idle_s is None:
            return False, "cannot tell whether the user is free"
        if env.idle_s < c.min_idle_s:
            return False, "busy right now"
        if env.idle_s > c.max_idle_s:
            return False, "away"
        if self.awaiting(now) is not None:
            return False, "still waiting for an answer"
        while self.asked_times and now - self.asked_times[0] > DAY:
            self.asked_times.popleft()
        if sum(1 for t in self.asked_times if now - t <= HOUR) >= c.max_per_hour:
            return False, "hourly cap"
        if len(self.asked_times) >= c.max_per_day:
            return False, "daily cap"
        if now - self.last_asked < c.min_gap_s:
            return False, "asked too recently"
        return True, "ok"

    def eligible(self, now: Optional[float] = None) -> List[Wonder]:
        now = self.clock() if now is None else now
        out = []
        for w in self.wonders.values():
            if w.status != "pending" or now < w.not_before:
                continue
            if self._subject_muted(w.subjects):
                continue
            if any(now - self.subject_last.get(s, -1e12) < self.cfg.subject_cooldown_s for s in w.subjects):
                continue
            if self._is_closed(w.key, now):
                continue
            if w.urgency < self.cfg.urgent_at and self.backoff.benched(w.kind, now):
                continue                     # this kind has been waved off lately
            out.append(w)
        return out

    def next_question(self, env: Environment, now: Optional[float] = None) -> Optional[Wonder]:
        """The wonder to ask now, or None. Does not mark it asked.

        Urgent ones may be asked at any ordinary pause; the rest are kept
        until `env.free_moment`. Among those allowed, the best is the one with
        the highest urgency/importance, scaled by how well the moment fits it.
        """
        now = self.clock() if now is None else now
        with self._lock:
            self.housekeeping(now)
            ok, _why = self.can_ask(env, now)
            if not ok:
                return None
            cands = self.eligible(now)
            if not cands:
                return None
            scored = []
            for w in cands:
                if w.urgency < self.cfg.urgent_at and not env.free_moment:
                    continue                     # saved for a free moment
                f = self.fit(w, env)
                if f > 0:
                    scored.append(((0.6 * w.urgency + 0.4 * w.priority) * f * self.usefulness(w, now),
                                   -w.created, w))
            if not scored:
                return None
            best = max(scored, key=lambda x: (x[0], x[1]))[2]
            self._prepare(best, now)
            return best

    def usefulness(self, w: Wonder, now: float) -> float:
        """How much an answer would be used, as a multiplier on the ranking.

        A subject that comes back day after day will come back again, so
        knowing it pays off more often than knowing something said once in a
        burst; and one that came up today is more on the user's mind (and the
        answer more likely to be used soon) than one from last week."""
        m = self.mentions.get(w.key) or {}
        days = len(m.get("days") or []) or 1
        f = 1.0 + 0.12 * min(days - 1, 3)
        last = float(m.get("t", w.created) or w.created)
        if now - last <= DAY:
            f += 0.08
        elif now - last > 3 * DAY:
            f -= 0.1
        return f

    def _top_open(self, now: float) -> Optional[Wonder]:
        """The best pending question when the user has asked for them: no
        pacing, but never one they muted or already closed."""
        cands = [w for w in self.wonders.values()
                 if w.status == "pending" and not self._subject_muted(w.subjects)
                 and not self._is_closed(w.key, now)]
        if not cands:
            return None
        return max(cands, key=lambda w: ((0.6 * w.urgency + 0.4 * w.priority) * self.usefulness(w, now),
                                         -w.created))

    def _ask_now(self, now: float) -> Optional[str]:
        w = self._top_open(now)
        if w is None:
            return None
        self._prepare(w, now)
        text = self.render(w, now)
        self.mark_asked(w.id, now)
        return text

    def _open_batch(self, now: float) -> Reply:
        """"What are you wondering about?" -- the questions, there and then."""
        n = sum(1 for w in self.wonders.values() if w.status == "pending")
        q = self._ask_now(now)
        if q is None:
            return Reply(True, "Nothing right now -- I'll ask when something puzzles me.", "batch")
        self._batch_until = now + 15 * 60
        self._batch_left = 4
        lead = "A few things. First: " if n > 1 else ""
        return Reply(True, lead + q, "batch")

    def _batch_next(self, rep: Reply, now: float) -> Reply:
        """After an answer in a batch, the next question rides on the reply."""
        if now > self._batch_until or self._batch_left <= 0 or rep.outcome == "followup":
            return rep
        q = self._ask_now(now)
        if q is None:
            rep.say = (rep.say + " That's everything I was wondering about.").strip()
            return rep
        self._batch_left -= 1
        rep.say = (rep.say + " Next: " + q).strip()
        return rep

    def _prepare(self, w: Wonder, now: float):
        """Look a term up once, just before it is asked. A guess becomes the
        question ("I think X means Y. Is that right?"); nothing found, or a
        lookup that fails, leaves the plain question."""
        if w.kind == "person" and not w.data.get("guessed"):
            self._guess_person(w)
            return
        if self.lookup is None or w.kind not in ("term", "activity") or w.data.get("looked_up"):
            return
        w.data["looked_up"] = True
        try:
            g = self.lookup(w)
        except Exception:
            logger.exception("Looking a term up failed; asking plainly")
            g = None
        if g and g.get("guess") and g.get("question"):
            w.data["guess"] = _clean(str(g["guess"]), 140)
            w.data["guess_source"] = str(g.get("source", ""))
            w.question = _clean(str(g["question"]), 240)
        self._touch()

    def _guess_person(self, w: Wonder):
        """If the company a name keeps points one way, ask a yes/no question."""
        w.data["guessed"] = True
        ctx = dict((self.mentions.get(w.key) or {}).get("ctx") or {})
        if not ctx:
            return
        best = max(ctx, key=lambda k: ctx[k])
        rest = max([v for k, v in ctx.items() if k != best] or [0])
        if ctx[best] < GUESS_MIN_CUES or ctx[best] < 2 * rest:
            return
        role, where, _rx = CONTEXT_CUES[best]
        name = w.subject
        w.data["guess"] = role
        art = "" if role == "family" else "a "
        w.question = f"You often mention {name} {where}. Is {name} {art}{role}?"
        self._touch()

    def render(self, w: Wonder, now: Optional[float] = None) -> str:
        """The question as it is said: a lead-in when it has been a while."""
        now = self.clock() if now is None else now
        q = w.question
        if w.kind in ("person", "term", "role", "between") and now - w.created > 10 * 60:
            lead = {"person": f"Earlier you mentioned {w.subject}. ",
                    "role": "Earlier you mentioned your " + str(w.data.get("role", "")) + ". ",
                    "between": "Earlier you mentioned them both. ",
                    "term": f"Earlier you said {w.subject}. "}[w.kind]
            q = lead + q
        if w.asked_count and not q.startswith("Earlier"):
            keep = q.startswith(("I ", "I'"))          # "I think..." keeps its capital
            q = "Still curious: " + (q if keep else q[:1].lower() + q[1:])
        return q

    def mark_asked(self, wid: str, now: Optional[float] = None) -> Optional[Wonder]:
        now = self.clock() if now is None else now
        with self._lock:
            w = self.wonders.get(wid)
            if w is None or w.status != "pending":
                return None
            w.status = "asked"
            w.asked_at = now
            w.asked_count += 1
            w.held = 0
            self.last_asked = now
            self.asked_times.append(now)
            for s in w.subjects:
                self.subject_last[s] = now
            self.stats["asked"] += 1
            self._touch()
            return w

    def note_held(self, wid: str, now: Optional[float] = None, reason: str = ""):
        """The attention gate would not let it through. Try later, a few times."""
        now = self.clock() if now is None else now
        with self._lock:
            w = self.wonders.get(wid)
            if w is None or w.status != "pending":
                return
            w.held += 1
            if w.held >= self.cfg.max_holds:
                self._close(w, "expired", now)
            else:
                w.not_before = now + self.cfg.hold_backoff_s * w.held
            self._touch()

    # ==================================================================
    # inspection and control
    # ==================================================================
    def pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = sorted((w for w in self.wonders.values() if w.status in OPEN),
                          key=lambda w: (-w.priority, w.created))
            return [{"id": w.id, "kind": w.kind, "subject": w.subject, "question": w.question,
                     "status": w.status, "priority": round(w.priority, 2),
                     "urgency": round(w.urgency, 2), "mentions": w.mentions,
                     "created": w.created, "asked_count": w.asked_count,
                     "not_before": w.not_before} for w in rows]

    def dismiss(self, wid: str, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        with self._lock:
            w = self.wonders.get(wid)
            if w is None or w.status not in OPEN:
                return False
            self._close(w, "dismissed", now)
            self._touch()
            return True

    def clear(self):
        with self._lock:
            for w in list(self.wonders.values()):
                if w.status in OPEN:
                    self._close(w, "dismissed", self.clock())
            self._touch()

    def report(self, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        rows = self.pending()
        state = ("off" if (not self.cfg.enabled or self.off)
                 else f"snoozed for {int((self.paused_until - now) / 60)} more min" if now < self.paused_until
                 else "on")
        lines = [f"Wondering: {state}. {len(rows)} open."]
        for r in rows:
            age = int((now - r["created"]) / 60)
            lines.append(f"  {r['id']}  {r['kind']:<9} {r['status']:<8} p={r['priority']:.2f} u={r['urgency']:.2f} "
                         f"x{r['mentions']} {age}m  {r['question']}")
        return "\n".join(lines)

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        with self._lock:
            return {
                "enabled": self.cfg.enabled and not self.off,
                "snoozed_s": max(0, int(self.paused_until - now)),
                "pending": sum(1 for w in self.wonders.values() if w.status == "pending"),
                "asked": sum(1 for w in self.wonders.values() if w.status == "asked"),
                "asked_last_hour": sum(1 for t in self.asked_times if now - t <= HOUR),
                "asked_last_day": sum(1 for t in self.asked_times if now - t <= DAY),
                "stats": dict(self.stats),
                "knowledge": self.kb.status(),
                **({"lookup": self.lookup_status()} if self.lookup_status else {}),
            }
