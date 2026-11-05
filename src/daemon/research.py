"""
src/daemon/research.py

"Look into X and tell me later."

A request to research something is not a question to answer now: it is a
job. This takes "look into ZXQ", "research the Raft algorithm", "find out
about Kubernetes when you can", acknowledges it, does the looking on its own
thread (so the ambient loop never waits on the network), and has the result
ready for the next free moment, where it competes with the other things ASH
could say (see freemoment.py), and for the evening briefing.

The looking uses what ASH has: the lookup sources (a Wikipedia summary,
definitions in the user's files, past sentences in episodic memory; see
lookup.py), plus where the topic comes up in the user's files and how often
they have mentioned it. The report says where each part came from, and says
plainly when nothing was found. Rule-based; no model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .freemoment import Activity, Candidate
from .lookup import TEXT_EXT, Lookup

logger = logging.getLogger("ash.daemon.research")

_ASK = re.compile(
    r"^\s*(?:(?:hey\s+)?(?:ash[,\s]+)?)?(?:please\s+|can you\s+|could you\s+)?"
    r"(?:look into|research|dig into|read up on|find out (?:more )?about)\s+(?P<topic>[^?!]{2,60}?)"
    r"(?:\s+(?:for me|later|tonight|when you can|when you get a chance|and (?:tell|let) me(?: know)?(?: later)?|"
    r"and get back to me))*\s*[.?!]?\s*$", re.I)


def parse_request(text: str) -> Optional[str]:
    """The topic a sentence asks to be researched, or None."""
    m = _ASK.match((text or "").strip()) if text and len(text) <= 140 else None
    if not m:
        return None
    topic = re.sub(r"^(?:the|a|an|some)\s+", "", m.group("topic").strip(" .,"), flags=re.I)
    if not topic or len(topic.split()) > 6 or topic.lower() in ("it", "that", "this", "them", "something"):
        return None
    return topic


@dataclass
class Job:
    id: str
    topic: str
    created: float
    status: str = "queued"          # queued | working | ready | told | failed
    report: str = ""
    found: bool = False
    finished: float = 0.0


@dataclass
class ResearchConfig:
    enabled: bool = True
    max_jobs: int = 10
    keep_s: float = 3 * 86400.0      # an untold report is still offered for this long
    file_mentions: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ResearchConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                cur = getattr(c, k)
                try:
                    setattr(c, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return c


class Research:
    def __init__(self, cfg: Optional[ResearchConfig] = None, lookup: Optional[Lookup] = None,
                 episodic=None, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 background: bool = True):
        self.cfg = cfg or ResearchConfig()
        self.lookup = lookup
        self.episodic = episodic
        self.path = path
        self.clock = clock
        self.background = background
        self._lock = threading.RLock()
        self.jobs: Dict[str, Job] = {}
        self.stats = {"asked": 0, "found": 0, "empty": 0, "told": 0}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("jobs", []):
                j = Job(**{k: v for k, v in d.items() if k in Job.__dataclass_fields__})
                if j.status == "working":
                    j.status = "queued"             # interrupted: do it again
                self.jobs[j.id] = j
        except Exception:
            logger.exception("Research state unreadable; starting empty")

    def save(self):
        if not self.path:
            return
        with self._lock:
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = f"{self.path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"jobs": [asdict(j) for j in self.jobs.values()]}, fh, indent=1)
                os.replace(tmp, self.path)
            except Exception:
                logger.exception("Could not save research jobs")

    # -- requests --------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        topic = parse_request(text)
        if topic is None:
            return None
        return self.ask(topic, now)

    def ask(self, topic: str, now: Optional[float] = None) -> Optional[str]:
        """Start a look-up of `topic` (already parsed); what to say back."""
        if not self.cfg.enabled or not (topic or "").strip():
            return None
        topic = topic.strip()
        now = self.clock() if now is None else now
        with self._lock:
            for j in self.jobs.values():
                if j.topic.lower() == topic.lower() and j.status in ("queued", "working", "ready"):
                    return f"I'm already looking into {topic}."
            active = [j for j in self.jobs.values() if j.status in ("queued", "working", "ready")]
            if len(active) >= self.cfg.max_jobs:
                return "I've got a lot to look into already. Let me get through those first."
            j = Job(id=uuid.uuid4().hex[:6], topic=topic, created=now)
            self.jobs[j.id] = j
            self.stats["asked"] += 1
            self.save()
        self._start(j)
        return f"Okay, I'll look into {topic} and tell you what I find."

    def _start(self, j: Job):
        if self.background:
            threading.Thread(target=self._run, args=(j,), daemon=True, name="ash-research").start()
        else:
            self._run(j)

    def _run(self, j: Job):
        with self._lock:
            if j.status != "queued":
                return
            j.status = "working"
        try:
            report, found = self._investigate(j.topic)
        except Exception:
            logger.exception("Research on a topic failed")
            report, found = "", False
        with self._lock:
            j.report = report or f"I looked into {j.topic} but found nothing beyond what you've told me."
            j.found = found
            j.status = "ready"
            j.finished = self.clock()
            self.stats["found" if found else "empty"] += 1
            self.save()

    def _investigate(self, topic: str):
        parts: List[str] = []
        found = False
        lk = self.lookup
        if lk is not None:
            # The web and the user's own material, each said for what it is:
            # a summary from Wikipedia, and a definition only if it is a
            # strong one (a to-do line that starts "ZXQ:" is not).
            web = lk.from_web(topic)
            if web:
                found = True
                parts.append(f"Wikipedia says {topic} is {web[0].text}.")
            local = [f for f in lk.from_memory(topic) + lk.from_files(topic) if f.confidence >= 0.75]
            if local:
                found = True
                f = max(local, key=lambda f: f.confidence)
                src = "From what you've told me" if f.source == "memory" else f"From {f.where or 'your files'}"
                parts.append(f"{src}: {topic} means {f.text}.")
            if self.cfg.file_mentions:
                files = self._files_mentioning(topic, lk)
                if files:
                    found = True
                    shown = ", ".join(files[:3])
                    parts.append(f"It comes up in {len(files)} of your files" + (f" ({shown})." if shown else "."))
        if self.episodic is not None:
            try:
                eps = self.episodic.retrieve(topic, top_k=10) or []
                n = sum(1 for e in eps if topic.lower() in str(getattr(e, "summary", "")).lower())
                if n:
                    found = True
                    parts.append(f"You've mentioned it {n} time{'s' if n != 1 else ''} before.")
            except Exception:
                logger.exception("episodic search failed")
        if not parts:
            return "", False
        return f"About {topic}: " + " ".join(parts), found

    @staticmethod
    def _files_mentioning(topic: str, lk: Lookup) -> List[str]:
        cfg = lk.cfg
        if not cfg.files or not cfg.roots:
            return []
        out: List[str] = []
        deadline = lk.clock() + cfg.budget_s
        needle = topic.lower()
        for path in lk._walk():
            if lk.clock() > deadline or len(out) >= 50:
                break
            if os.path.splitext(path)[1].lower() not in TEXT_EXT:
                continue
            try:
                if os.path.getsize(path) > cfg.max_bytes:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    if needle in fh.read().lower():
                        out.append(os.path.basename(path))
            except OSError:
                continue
        return out

    # -- telling ---------------------------------------------------------------
    def ready(self, now: Optional[float] = None) -> Optional[Job]:
        now = self.clock() if now is None else now
        with self._lock:
            r = [j for j in self.jobs.values() if j.status == "ready" and now - j.finished <= self.cfg.keep_s]
            return min(r, key=lambda j: j.finished) if r else None

    def mark_told(self, jid: str):
        with self._lock:
            j = self.jobs.get(jid)
            if j is not None and j.status == "ready":
                j.status = "told"
                self.stats["told"] += 1
                self.save()

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        """For the evening briefing: a finished report not yet told."""
        j = self.ready(now)
        if j is None or kind != "evening":
            return None
        self.mark_told(j.id)
        return j.report

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.cfg.enabled,
                    "pending": sum(1 for j in self.jobs.values() if j.status in ("queued", "working")),
                    "ready": sum(1 for j in self.jobs.values() if j.status == "ready"), **self.stats}


class ResearchActivity(Activity):
    """A finished report, offered at a free moment."""
    name = "research"
    weight = 1.0
    cooldown_s = 600.0

    def __init__(self, research: Research):
        self.research = research
        self._proposed: Optional[str] = None

    def propose(self, env, now):
        j = self.research.ready(now)
        if j is None:
            return None
        self._proposed = j.id
        return Candidate(self.name, "You asked me to look into something. " + j.report, 0.85 if j.found else 0.5)

    def spoken(self, now):
        if self._proposed:
            self.research.mark_told(self._proposed)
            self._proposed = None
