"""
src/daemon/activity.py

Activity memory: what the user does, kept the way a conversation is.

ASH sees which window has focus and, with the screen sensors on, what is in
it. Until now that went into a day-long journal and was summarised, once, at
the end of the day, as "mostly in Code.exe". Nothing in it was ever connected
to anything ASH already knew, and nothing in it was ever asked about.

A person who watched you work would do three things, and this does the same:

  REMEMBER   "Immortal spent 40 minutes on brain.py in A.S.H (VS Code)" goes
             into episodic memory as it happens, as a sentence, the way a
             turn of conversation does -- so "what was I doing this morning"
             has something to find. Not every glance: a stretch has to last
             a few minutes, and there is a cap per hour.

  CONNECT    Most of what anyone does is one-off. What matters is what ties
             into something already known. A topic is kept, and made an
             entity, only if it is connected to something:

               entity        its name contains an entity ASH already has
               conversation  it shares a distinctive word with what the user
                             has said to ASH recently
               with          it keeps turning up in the same sitting as
                             another topic
               recurs        it keeps coming back, over days or many sittings

             Anything connected to nothing is let go -- it is in the journal
             if it is wanted, and it would only be noise as an entity.

  WONDER     A topic that is connected but unexplained -- ASH has no idea what
             "A.S.H" is *for*, only that it keeps showing up beside the thing
             you were talking about -- becomes a question, asked at a free
             moment through the same machinery as every other question (see
             wondering.py: rate caps, cool-downs, quiet hours, never again
             once answered). The answer is stored where every other answer is.

Privacy
-------
It reads window title fields (the file, the project) and, if the screen
reader is on, a handful of KEYWORDS from what was read -- never the text. A
window the privacy rules withhold (blocked, or a chat or mailbox while that
protection is on) arrives with no subject, so nothing is learned from it. The
topic file lives in state/activity/, which is not committed. `activity.enabled`
turns all of it off; `activity.keep_terms` stops the keywords being kept.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .social import friendly_app
from src.memory.atomic import log_io_failure, write_json

logger = logging.getLogger("ash.daemon.activity")

HOUR = 3600.0
DAY = 86400.0

# Words that are not distinctive: they would "connect" a window to a
# conversation by accident. Plus the words an interface puts in its titles.
_STOP = frozenset("""
about above after again against along also always among another around because before being below between
both cannot could doing during each either every first from further have having here into itself just
later least less like make many might more most much must never next none often once only other others
over own really right same should since some still such than that their them then there these they
this those though through under until upon very want were what when where which while whose will with
within without would your yours yourself
google search chrome opera firefox edge safari browser window windows untitled document documents
new tab home page settings preferences loading welcome start menu desktop folder file files open
editor visual studio code notepad explorer terminal command prompt python console
""".split())

# A window whose title is nothing but this is not a thing someone is doing.
_GENERIC_SUBJECTS = frozenset("""
new tab|untitled|home|loading|settings|welcome|start|menu|desktop|downloads|documents|inbox|
speed dial|blank|about blank|search|results|sign in|log in|login|dashboard|overview|index|
task manager|file explorer|windows explorer|this pc|control panel|snipping tool|calculator|
task view|program manager|notification center|run|system settings|device manager
""".replace("\n", "").split("|"))

# Entity types that name a number or a moment, not a thing. And the user.
_SKIP_ENTITY_TYPES = frozenset({"date", "time", "cardinal", "ordinal", "money", "percent",
                                "quantity", "speaker_user", "language"})

# In "<page> - Google Gemini - Opera" the middle field is the website, not
# something the user is working on, and asking what Google Gemini is would be
# absurd. The page title is still a topic; the site is not.
_BROWSERS = frozenset({"opera", "chrome", "msedge", "edge", "firefox", "brave", "vivaldi",
                       "safari", "arc", "chromium", "iexplore", "waterfox", "tor"})

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{4,}")
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def clean(s: str, n: int = 60) -> str:
    return _WS.sub(" ", str(s or "")).strip()[:n]


def keywords(text: str, n: int = 8) -> List[str]:
    """The few most repeated distinctive words in `text`. Words, not sentences:
    this is the only thing ever kept from what was read off a screen."""
    c = Counter(w for w in (x.lower().strip("'-") for x in _WORD.findall(text or ""))
                if len(w) >= 5 and w not in _STOP)
    return [w for w, _ in c.most_common(n)]


def _is_self(subject: str) -> bool:
    low = (subject or "").strip().lower()
    return low == "a.s.h" or "ashd.py" in low or "python ashd" in low


# ----------------------------------------------------------------------------
@dataclass
class ActivityConfig:
    enabled: bool = True
    remember: bool = True              # write the conversation-like episodes
    ask: bool = True                   # raise questions about connected topics
    keep_terms: bool = True            # keep keywords read off the screen
    min_stretch_s: float = 180.0       # shorter than this is a glance
    episode_min_s: float = 300.0       # a stretch this long is worth a sentence
    idle_cut_s: float = 120.0          # this long away ends a stretch
    long_stretch_s: float = 1800.0     # an unbroken stretch is recorded in pieces
    session_gap_s: float = 900.0       # topics this close together were one sitting
    min_recurrence: int = 3            # stretches before a topic "recurs"
    ask_after_stretches: int = 2       # seen at least twice before it is asked about
    max_episodes_per_hour: int = 6
    max_topics: int = 400
    conv_days: float = 14.0            # how far back "what the user said" reaches
    sources: Tuple[str, ...] = ("active_window", "screenshot")

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ActivityConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            if k == "sources" and isinstance(v, (list, tuple)):
                cfg.sources = tuple(str(x) for x in v)
            elif isinstance(cur, bool):
                setattr(cfg, k, bool(v))
            elif isinstance(cur, (int, float)) and isinstance(v, (int, float)):
                setattr(cfg, k, type(cur)(v))
        return cfg


@dataclass
class Topic:
    key: str
    name: str
    kind: str                          # "subject" (a file, a page) | "project"
    stretches: int = 0
    seconds: float = 0.0
    first: float = 0.0
    last: float = 0.0
    days: List[str] = field(default_factory=list)
    apps: List[str] = field(default_factory=list)
    terms: Dict[str, int] = field(default_factory=dict)
    together: Dict[str, int] = field(default_factory=dict)   # topic key -> sittings shared
    links: List[Dict[str, str]] = field(default_factory=list)
    by_day: Dict[str, float] = field(default_factory=dict)  # local date -> seconds, last two weeks
    remembered: bool = False           # made an entity: it is connected to something
    entity_id: str = ""
    asked: bool = False


@dataclass
class _Stretch:
    subject: str
    project: str
    process: str
    start: float
    last: float
    terms: Counter = field(default_factory=Counter)


# ----------------------------------------------------------------------------
class ActivityMemory:
    def __init__(self, cfg: Optional[ActivityConfig] = None, path: Optional[str] = None,
                 entities=None, episodic=None, wonder=None, knowledge=None,
                 user: str = "", clock: Callable[[], float] = time.time,
                 scrub: Optional[Callable[[str], str]] = None):
        self.cfg = cfg or ActivityConfig()
        self.path = path
        self.entities = entities           # an EntityManager, or None
        self.episodic = episodic           # an EpisodicManager, or None
        self.wonder = wonder               # a Wonderer, or None
        self.kb = knowledge                # its Knowledge, or None
        self.user = str(user or "")
        self.clock = clock
        self.scrub = scrub or (lambda s: s)
        self._lock = threading.RLock()
        self.topics: Dict[str, Topic] = {}
        self._cur: Optional[_Stretch] = None
        self._session: List[Tuple[str, float, str]] = []     # (topic key, end, project) of recent stretches
        self._episodes: Deque[float] = deque(maxlen=64)
        self._ent_cache: Tuple[float, List[Tuple[str, str, str]]] = (0.0, [])
        self._conv_cache: Tuple[float, frozenset] = (0.0, frozenset())
        self._dirty = False
        self._saved_at = 0.0
        self.stats = {"stretches": 0, "episodes": 0, "saved": 0, "asked": 0, "ignored": 0}
        # Told of each stretch recorded: (what it was on, start, end). The
        # celebrations (cheer.py) join the pieces of a long one back up.
        self.on_stretch: Optional[Callable[[str, float, float], None]] = None
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for k, d in (raw.get("topics") or {}).items():
                self.topics[k] = Topic(**{f: d[f] for f in Topic.__dataclass_fields__ if f in d})
            self.stats.update({k: int(v) for k, v in (raw.get("stats") or {}).items()
                               if k in self.stats})
        except Exception:
            logger.exception("Could not read the activity file; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except OSError:
                pass

    def save(self, force: bool = False):
        if not self.path or not (self._dirty or force):
            return
        now = self.clock()
        if not force and now - self._saved_at < 30.0:
            return
        self._saved_at = now                    # even a failed attempt waits: a full disk is not retried every tick
        try:
            write_json(self.path, {"v": 1, "stats": self.stats,
                                   "topics": {k: asdict(t) for k, t in self.topics.items()}})
            self._dirty = False
        except Exception as e:
            log_io_failure(logger, "the activity file", e)

    # ---- input -----------------------------------------------------------
    def observe(self, ev, now: Optional[float] = None) -> None:
        """Feed it the events the ambient loop sees. Anything else is ignored."""
        if not self.cfg.enabled:
            return
        src = str(getattr(ev, "source", "") or "").lower()
        src = src[4:] if src.startswith("sim:") else src
        if src not in self.cfg.sources:
            return
        now = self.clock() if now is None else now
        meta = getattr(ev, "meta", None) or {}
        with self._lock:
            if src == "active_window":
                self._on_window(meta, now)
            elif src == "screenshot":
                self._on_reading(meta, now)

    def tick(self, now: Optional[float] = None, idle_s: Optional[float] = None) -> None:
        """Once per loop tick. Away ends a stretch where it last had the user;
        a very long one is recorded in pieces so it is not lost to a crash and
        is not one enormous entry."""
        if not self.cfg.enabled:
            return
        now = self.clock() if now is None else now
        with self._lock:
            st = self._cur
            if st is not None:
                if idle_s is not None and idle_s >= self.cfg.idle_cut_s:
                    self._close(st.last)
                else:
                    st.last = now
                    if now - st.start >= self.cfg.long_stretch_s:
                        again = _Stretch(st.subject, st.project, st.process, now, now)
                        self._close(now)
                        self._cur = again
            self.save()

    # ---- stretches -------------------------------------------------------
    def _on_window(self, meta: Dict[str, Any], now: float):
        subject = clean(meta.get("subject") or "")
        project = clean(meta.get("project") or "")
        process = str(meta.get("process") or "")
        self._close(now)                  # the window changed: whatever was open ended
        if (meta.get("blocked") or not subject or _is_self(subject)
                or norm(subject) in _GENERIC_SUBJECTS or len(norm(subject)) < 3):
            self.stats["ignored"] += 1
            return
        self._cur = _Stretch(subject, project, process, now, now)

    def _on_reading(self, meta: Dict[str, Any], now: float):
        st = self._cur
        text = meta.get("on_screen") or ""
        if st is None or not text or not self.cfg.keep_terms:
            return
        title = norm(meta.get("title") or "")
        if title and norm(st.subject) not in title:
            return                        # read off a window that is no longer this one
        st.terms.update(keywords(text, 12))

    def _close(self, end: float):
        st, self._cur = self._cur, None
        if st is None:
            return
        dur = end - st.start
        if dur < self.cfg.min_stretch_s:
            return
        self._record(st, end, dur)

    # ---- recording -------------------------------------------------------
    def _record(self, st: _Stretch, end: float, dur: float):
        self.stats["stretches"] += 1
        if self.on_stretch is not None:
            try:
                self.on_stretch(clean(st.project or st.subject or ""), st.start, end)
            except Exception:
                logger.debug("on_stretch failed", exc_info=True)
        minutes = max(1, int(round(dur / 60.0)))
        day = datetime.fromtimestamp(st.start).strftime("%Y-%m-%d")
        app = friendly_app(st.process)

        touched: List[Topic] = []
        stem = re.sub(r"\.(?:exe|app|bin)$", "", (st.process or "").strip().lower())
        project = "" if stem in _BROWSERS else st.project
        for kind, name in (("project", project), ("subject", st.subject)):
            if name and norm(name) not in _GENERIC_SUBJECTS and len(norm(name)) >= 3:
                touched.append(self._touch(kind, name, st, dur, day, end, app))

        # topics from the same sitting are connected to each other
        # ...but not a project to its own files, nor two files of one project:
        # working on brain.py and then system2.py inside A.S.H is just working
        # on A.S.H, and would otherwise "connect" every file to its project.
        recent = [(k, e, p) for k, e, p in self._session if st.start - e <= self.cfg.session_gap_s]
        proj = norm(project)
        for t in touched:
            for k, _e, p in recent:
                if k == t.key or k not in self.topics:
                    continue
                if proj and proj == p:
                    continue
                if t.kind == "subject" and k == f"project:{proj}":
                    continue
                if self.topics[k].kind == "subject" and t.key == f"project:{p}":
                    continue
                t.together[k] = t.together.get(k, 0) + 1
                self.topics[k].together[t.key] = self.topics[k].together.get(t.key, 0) + 1
        self._session = ([(t.key, end, proj) for t in touched] + recent)[:24]

        for t in touched:
            t.links = self._links(t, end)
            if t.links and not t.remembered:
                self._promote(t, end)
            self._maybe_ask(t, end)

        if self.cfg.remember and dur >= self.cfg.episode_min_s:
            self._episode(st, minutes, touched, end, project)
        self._trim()
        self._dirty = True
        self.save()

    def _touch(self, kind: str, name: str, st: _Stretch, dur: float, day: str,
               end: float, app: str) -> Topic:
        key = f"{kind}:{norm(name)}"
        t = self.topics.get(key)
        if t is None:
            t = self.topics[key] = Topic(key=key, name=name, kind=kind, first=st.start)
        t.stretches += 1
        t.seconds += dur
        t.last = end
        if day not in t.days:
            t.days = (t.days + [day])[-30:]
        t.by_day[day] = t.by_day.get(day, 0.0) + dur
        if len(t.by_day) > 14:
            for old in sorted(t.by_day)[:len(t.by_day) - 14]:
                del t.by_day[old]
        if app and app not in t.apps:
            t.apps = (t.apps + [app])[-5:]
        if self.cfg.keep_terms:
            for w, n in st.terms.most_common(8):
                t.terms[w] = t.terms.get(w, 0) + n
            if len(t.terms) > 24:
                t.terms = dict(Counter(t.terms).most_common(24))
        return t

    def _trim(self):
        if len(self.topics) <= self.cfg.max_topics:
            return
        # least worth keeping first: unconnected, seldom seen, long ago
        order = sorted(self.topics.values(),
                       key=lambda t: (t.remembered, t.stretches, t.last))
        for t in order[:len(self.topics) - self.cfg.max_topics]:
            self.topics.pop(t.key, None)

    # ---- connecting ------------------------------------------------------
    def _user_said(self, horizon: Optional[float] = None) -> List[str]:
        """What the user has said to ASH, from episodic memory: their own
        turns only, never ASH's replies."""
        out: List[str] = []
        me = f"{self.user}:" if self.user else None
        try:
            for ep in (self.episodic.store.all() if self.episodic is not None else []):
                if getattr(ep, "event_type", "") != "interaction":
                    continue
                if horizon is not None and getattr(ep, "timestamp", 0) < horizon:
                    continue
                text = str(getattr(ep, "summary", ""))
                if (me and not text.startswith(me)) or text.startswith("A.S.H:"):
                    continue
                out.append(text)
        except Exception:
            logger.exception("Could not read episodic memory")
        return out

    def _entity_index(self) -> List[Tuple[str, str, str]]:
        """(needle, name as shown, id) for each entity ASH already has that a
        window title could plausibly be about. Cached: it is read per stretch.

        Only an entity the *user* has said counts as known. Entity memory
        also holds names minted from ASH's own replies -- "Google Gemini" and
        a whole document title, from a remark ASH made about the window it was
        looking at -- and treating those as already known made ASH ask "I only
        know the name, what is it?" about a thing it had named itself."""
        at, cached = self._ent_cache
        now = self.clock()
        if cached and now - at < 60.0:
            return cached
        out: List[Tuple[str, str, str]] = []
        me = norm(self.user)
        said = norm(" ".join(self._user_said()))
        try:
            for e in (self.entities.store.all() if self.entities is not None else []):
                if str(getattr(e, "entity_type", "")).lower() in _SKIP_ENTITY_TYPES:
                    continue
                attrs = getattr(e, "attributes", None) or {}
                if attrs.get("source") == "activity":
                    continue                  # one this module made is not "already known"
                etype = str(getattr(e, "entity_type", "")).lower()
                shown = str(attrs.get("canonical_name") or
                            (getattr(e, "primary_identifiers", None) or {}).get("name") or "")
                needle = norm(shown)
                if len(needle) < 5 or needle in _STOP or needle == me:
                    continue
                told = attrs.get("source") not in ("chat", "ash")    # an answer, or ASH itself
                if not (told or etype == "speaker_self"
                        or re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", said)):
                    continue                  # nothing the user said names it
                out.append((needle, shown, getattr(e, "entity_id", "")))
        except Exception:
            logger.exception("Could not read entity memory")
        self._ent_cache = (now, out)
        return out

    def _conversation_words(self) -> frozenset:
        """Distinctive words in what the user has said to ASH lately."""
        at, cached = self._conv_cache
        now = self.clock()
        if cached and now - at < 120.0:
            return cached
        words: set = set()
        for text in self._user_said(now - self.cfg.conv_days * DAY):
            words.update(w.lower().strip("'-") for w in _WORD.findall(text))
        words = {w for w in words if len(w) >= 5 and w not in _STOP}
        self._conv_cache = (now, frozenset(words))
        return self._conv_cache[1]

    def _links(self, t: Topic, now: float) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        text = norm(t.name)
        for needle, shown, eid in self._entity_index():
            if re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", text):
                # The topic IS this entity ("named"), or merely mentions it.
                links.append({"type": "named" if needle == text else "entity",
                              "to": shown, "id": eid})
                if len(links) >= 3:
                    break

        mine = {w.lower().strip("'-") for w in _WORD.findall(t.name)} | set(t.terms)
        shared = sorted(w for w in mine if w in self._conversation_words() and w not in _STOP)
        if shared:
            links.append({"type": "conversation", "to": ", ".join(shared[:2])})

        for k, n in sorted(t.together.items(), key=lambda kv: -kv[1])[:2]:
            if n >= 2 and k in self.topics:
                links.append({"type": "with", "to": self.topics[k].name})

        if t.stretches >= self.cfg.min_recurrence or (t.stretches >= 2 and len(t.days) >= 2):
            links.append({"type": "recurs", "to": ""})
        return links

    def _promote(self, t: Topic, now: float):
        """Connected to something: keep it, as an entity and in the episodes."""
        t.remembered = True
        self.stats["saved"] += 1
        if self.entities is None:
            return
        try:
            who = self.user or "The user"
            e = self.entities.ingest({
                "entity_type": "project" if t.kind == "project" else "topic",
                "primary_identifiers": {"name": norm(t.name)},
                "attributes": {"canonical_name": t.name, "source": "activity",
                               "last_seen_context": f"{who} spent time on {t.name}",
                               "updated_at": now}})
            t.entity_id = getattr(e, "entity_id", "") or ""
        except Exception:
            logger.exception("Could not make an entity of an activity topic")

    def _episode(self, st: _Stretch, minutes: int, topics: List[Topic], now: float,
                 project: str = ""):
        while self._episodes and now - self._episodes[0] > HOUR:
            self._episodes.popleft()
        if len(self._episodes) >= self.cfg.max_episodes_per_hour or self.episodic is None:
            return
        who = self.user or "The user"
        where = f" on {st.subject}" + (f" in {project}" if project else "")
        app = friendly_app(st.process)
        text = (f"{who} spent {minutes} {'minute' if minutes == 1 else 'minutes'}{where}"
                + (f" ({app})" if app else ""))
        ids = [t.entity_id for t in topics if t.entity_id]
        for t in topics:
            ids += [l["id"] for l in t.links if l.get("id")]
        try:
            self.episodic.add_episode(
                summary=self.scrub(text), event_type="activity",
                related_entities=list(dict.fromkeys(ids)),
                importance=min(0.6, 0.25 + minutes / 120.0))
            self._episodes.append(now)
            self.stats["episodes"] += 1
        except Exception:
            logger.exception("Could not write an activity episode")

    # ---- wondering -------------------------------------------------------
    @staticmethod
    def _question(t: Topic) -> str:
        by = {l["type"]: l for l in t.links}
        name = t.name
        if "named" in by:
            return f"I keep seeing you in {name}, and I only know the name. What is it?"
        if "entity" in by:
            return (f"You've spent a while on {name}. Is that to do with "
                    f"{by['entity']['to']}? What is it?")
        if "conversation" in by:
            return (f"You keep going back to {name}, and you've talked about "
                    f"{by['conversation']['to']}. Is it the same thing? What is it?")
        if "with" in by:
            return (f"You keep switching between {name} and {by['with']['to']}. "
                    f"What are you working on?")
        return f"I keep seeing you in {name}. What is it?"

    def _maybe_ask(self, t: Topic, now: float):
        if not (self.cfg.ask and self.wonder is not None) or t.asked or not t.links:
            return
        kinds = {l["type"] for l in t.links}
        # Recurrence alone is a reason to ask about a project, not about a page.
        if not (kinds & {"named", "entity", "conversation", "with"}
                or ("recurs" in kinds and t.kind == "project")):
            return
        if t.stretches < self.cfg.ask_after_stretches:
            return
        if self.kb is not None and self.kb.term_known(t.name):
            t.asked = True                # already told: nothing to wonder
            return
        try:
            w = self.wonder.raise_wonder(
                "activity", t.name, self._question(t), key=f"activity:{norm(t.name)}",
                data={"term": t.name, "how": "activity",
                      "links": [l["type"] for l in t.links]}, now=now)
        except Exception:
            logger.exception("Could not raise a wonder about an activity topic")
            return
        if w is not None:
            t.asked = True
            self.stats["asked"] += 1

    # ---- inspection ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"topics": len(self.topics),
                    "connected": sum(1 for t in self.topics.values() if t.remembered),
                    "asked": sum(1 for t in self.topics.values() if t.asked),
                    "now": (self._cur.subject if self._cur else None), **self.stats}

    def report(self, limit: int = 10) -> str:
        with self._lock:
            s = self.status()
            lines = [f"activity: {s['topics']} topics, {s['connected']} connected, "
                     f"{s['episodes']} episodes written, {s['asked']} asked about"]
            top = sorted(self.topics.values(), key=lambda t: -t.seconds)[:limit]
            for t in top:
                why = ", ".join(sorted({l['type'] for l in t.links})) or "unconnected"
                lines.append(f"  {t.name[:40]:40s} {t.kind:7s} {t.seconds / 60:5.0f} min "
                             f"x{t.stretches}  [{why}]")
            return "\n".join(lines)

    def forget(self):
        with self._lock:
            self.topics.clear()
            self._session = []
            self._cur = None
            self._dirty = True
            self.save(force=True)
