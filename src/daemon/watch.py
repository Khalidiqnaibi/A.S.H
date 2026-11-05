"""
src/daemon/watch.py

Watching something for the user, and saying when it happens.

The system sensor already speaks up at a threshold (battery at 8%, disk 92%
full). What nobody did was watch something over time, or watch something
because the user asked:

    asked    "tell me when the download finishes", "let me know when chrome
             closes", "tell me when the battery is charged" (or "at 80%"),
             "tell me when the CPU calms down", "tell me when report.docx
             changes". Read by rule, acknowledged at once, checked every few
             seconds, said once when it happens, and let go after a day if it
             never does. "Stop watching" drops them.
    trend    disk use sampled over hours: if, at the rate it is growing, the
             disk will be full within two days, ASH says so once a day.

What cannot be seen is said so plainly ("I can't see builds directly -- tell
me the program's name") rather than promised. The machine is read through a
`Probe`, so tests hand in their own. Rule-based throughout.
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
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ash.daemon.watch")

HOUR, DAY = 3600.0, 86400.0
PARTIAL = (".crdownload", ".part", ".partial", ".download", ".tmp", ".opdownload")


class Probe:
    """The machine, read live. Every method may return None if it cannot."""

    def processes(self) -> Optional[Set[str]]:
        try:
            import psutil
            return {(p.info.get("name") or "").lower() for p in psutil.process_iter(["name"])}
        except Exception:
            return None

    def battery(self) -> Optional[Tuple[float, bool]]:
        try:
            import psutil
            b = psutil.sensors_battery()
            return (float(b.percent), bool(b.power_plugged)) if b else None
        except Exception:
            return None

    def cpu(self) -> Optional[float]:
        try:
            import psutil
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def disk(self) -> Optional[Tuple[float, float]]:
        """(used bytes, total bytes) of the system drive."""
        try:
            import shutil
            u = shutil.disk_usage(os.path.abspath(os.sep))
            return float(u.used), float(u.total)
        except Exception:
            return None

    def listdir(self, path: str) -> Optional[Dict[str, Tuple[int, float]]]:
        try:
            out = {}
            for e in os.scandir(os.path.expanduser(path)):
                if e.is_file():
                    st = e.stat()
                    out[e.name] = (int(st.st_size), float(st.st_mtime))
            return out
        except Exception:
            return None

    def mtime(self, path: str) -> Optional[float]:
        try:
            return float(os.path.getmtime(os.path.expanduser(path)))
        except Exception:
            return None


@dataclass
class Watch:
    id: str
    kind: str                       # process | download | battery | cpu | file
    what: str                       # said back: "the download", "chrome"
    created: float
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"          # active | done | expired | cancelled
    last_check: float = 0.0


@dataclass
class WatchConfig:
    enabled: bool = True
    check_every_s: float = 5.0
    expire_s: float = DAY
    downloads_dir: str = "~/Downloads"
    cpu_calm_below: float = 30.0
    cpu_calm_hold_s: float = 30.0
    download_settle_s: float = 10.0
    max_watches: int = 20
    disk_trend: bool = True
    disk_sample_s: float = HOUR
    disk_warn_s: float = 2 * DAY

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "WatchConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                continue
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, str):
                    setattr(c, k, str(v))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                else:
                    setattr(c, k, float(v))
            except (TypeError, ValueError):
                pass
        return c


# ----------------------------------------------------------------------------
# Reading a request
# ----------------------------------------------------------------------------
_ASK = re.compile(r"\b(?:tell|let|ping|notify|alert|remind|warn)\s+me\s+(?:know\s+)?"
                  r"(?:when|once|as soon as|if)\s+(?P<what>[^?!]{2,120})", re.I)
_CANCEL = re.compile(r"^\s*(?:(?:please\s+)?stop watching|cancel (?:the |all )?(?:watch(?:es)?|alerts?)|"
                     r"never ?mind (?:the |about the )?(?P<what>download|battery|cpu|[\w.\-]+)\b|"
                     r"don'?t (?:tell|let) me (?:know )?(?:about|when) (?:the )?(?P<what2>[\w.\-]+))", re.I)
_DONE = r"(?:finish(?:es|ed)?|is (?:done|finished|over|complete)|completes?|ends|is through)"
_NOT_PROCS = frozenset("the it this that my a an download downloads build battery cpu computer machine laptop pc "
                       "file update upload install installation copy transfer render export meeting call".split())


def parse_request(text: str, cfg: Optional[WatchConfig] = None) -> Optional[Dict[str, Any]]:
    """What a sentence asks to be watched: {"kind", "what", ...}, or
    {"kind": "unsupported", "say": ...}, or None if it asks nothing of the sort."""
    cfg = cfg or WatchConfig()
    t = (text or "").strip()
    if not t or len(t) > 200:
        return None
    m = _ASK.search(t)
    if not m:
        return None
    what = m.group("what").strip(" .")
    w = what.lower()
    if re.search(r"\bdownloads?\b", w) and re.search(_DONE + r"|\bdone\b", w):
        return {"kind": "download", "what": "the download", "dir": cfg.downloads_dir}
    mb = re.search(r"\bbattery\b", w)
    if mb:
        n = re.search(r"\b(\d{1,3})\s*%?", w)
        if n and 1 <= int(n.group(1)) <= 100:
            return {"kind": "battery", "what": f"the battery reaches {int(n.group(1))}%", "pct": int(n.group(1))}
        if re.search(r"\b(?:full|charged|fully charged)\b", w):
            return {"kind": "battery", "what": "the battery is charged", "pct": 100}
        return None
    if re.search(r"\b(?:cpu|computer|machine|laptop|pc|fans?)\b", w) and re.search(
            r"\b(?:calms? down|cools? down|is free|frees up|is idle|quiets? down|settles?|drops)\b", w):
        return {"kind": "cpu", "what": "the CPU calms down", "below": cfg.cpu_calm_below}
    mf = re.search(r"(?P<path>[~\w\-. /\\:]+\.[A-Za-z0-9]{1,5})\s+(?:changes|is (?:updated|modified|changed|saved)|"
                   r"gets (?:updated|written|saved))\b", what)
    if mf:
        path = mf.group("path").strip()
        path = re.sub(r"^(?:the|my)\s+", "", path, flags=re.I)
        return {"kind": "file", "what": f"{os.path.basename(path)} changes", "path": path}
    if re.search(r"\b(?:build|tests?|pipeline|deploy(?:ment)?|ci)\b", w):
        return {"kind": "unsupported",
                "say": "I can't see builds directly. Tell me the program's name, like "
                       "\"tell me when npm finishes\", and I'll watch for it to close."}
    mp = re.match(r"(?:the\s+|my\s+)?(?P<proc>[\w.\-]{2,40})(?:\s+(?:app|program|process))?\s+(?:" + _DONE +
                  r"|closes|exits|quits|stops(?: running)?|is closed)\b", w)
    if mp and mp.group("proc") not in _NOT_PROCS:
        return {"kind": "process", "what": mp.group("proc"), "proc": mp.group("proc")}
    return None


def wants_cancel(text: str) -> Optional[str]:
    """None, or what to stop watching ('' for everything)."""
    m = _CANCEL.match((text or "").strip())
    if not m:
        return None
    return (m.group("what") or m.group("what2") or "").lower()


def _proc_matches(name: str, running: Set[str]) -> bool:
    n = name.lower()
    stem = re.sub(r"\.(?:exe|app)$", "", n)
    return any(r == n or r == stem + ".exe" or re.sub(r"\.(?:exe|app)$", "", r) == stem for r in running)


# ----------------------------------------------------------------------------
class Watcher:
    def __init__(self, cfg: Optional[WatchConfig] = None, path: Optional[str] = None,
                 probe: Optional[Probe] = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or WatchConfig()
        self.path = path
        self.probe = probe or Probe()
        self.clock = clock
        self._lock = threading.RLock()
        self.watches: Dict[str, Watch] = {}
        self.disk_samples: Deque[Tuple[float, float]] = deque(maxlen=48)
        self.disk_warned = 0.0
        self.stats = {"set": 0, "fired": 0, "expired": 0, "cancelled": 0, "trend_warnings": 0}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for d in raw.get("watches", []):
                w = Watch(**{k: v for k, v in d.items() if k in Watch.__dataclass_fields__})
                if w.status == "active":
                    self.watches[w.id] = w
            self.disk_samples = deque((tuple(x) for x in raw.get("disk_samples", [])), maxlen=48)
            self.disk_warned = float(raw.get("disk_warned", 0.0))
            self.stats.update({k: int(v) for k, v in (raw.get("stats") or {}).items() if k in self.stats})
        except Exception:
            logger.exception("Watch state unreadable; starting empty")
            try:
                os.replace(self.path, self.path + ".bad")
            except Exception:
                pass

    def save(self):
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"watches": [asdict(w) for w in self.watches.values() if w.status == "active"],
                           "disk_samples": list(self.disk_samples), "disk_warned": self.disk_warned,
                           "stats": self.stats}, fh, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("Could not save watches")

    # -- requests --------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """A request to watch (or to stop): returns what to say back, or None
        if the sentence asks nothing of the kind."""
        if not self.cfg.enabled:
            return None
        now = self.clock() if now is None else now
        c = wants_cancel(text)
        if c is not None:
            return self.cancel(c, now)
        req = parse_request(text, self.cfg)
        if req is None:
            return None
        if req["kind"] == "unsupported":
            return req["say"]
        return self.add(req, now)

    def add(self, req: Dict[str, Any], now: float) -> str:
        p = self.probe
        kind = req["kind"]
        data = {k: v for k, v in req.items() if k not in ("kind", "what")}
        if kind == "process":
            running = p.processes()
            if running is None:
                return "I can't see running programs on this machine, so I can't watch for that."
            if not _proc_matches(req["proc"], running):
                return f"I don't see {req['proc']} running right now."
        elif kind == "download":
            listing = p.listdir(data["dir"])
            if listing is None:
                return "I can't read your Downloads folder, so I can't watch it."
            data["baseline"] = sorted(n for n in listing if not n.lower().endswith(PARTIAL))
            data["partials"] = sorted(n for n in listing if n.lower().endswith(PARTIAL))
        elif kind == "battery":
            b = p.battery()
            if b is None:
                return "I can't read a battery on this machine."
            if b[0] >= req["pct"]:
                return f"The battery is already at {b[0]:.0f}%."
        elif kind == "cpu":
            if p.cpu() is None:
                return "I can't read the CPU load here."
        elif kind == "file":
            mt = p.mtime(data["path"])
            if mt is None:
                return f"I can't find {req['path']}."
            data["mtime"] = mt
        with self._lock:
            for w in self.watches.values():
                if w.status == "active" and w.kind == kind and w.what == req["what"]:
                    return f"I'm already watching for that."
            active = [w for w in self.watches.values() if w.status == "active"]
            if len(active) >= self.cfg.max_watches:
                return "I'm watching too many things already. Say \"stop watching\" to clear them."
            w = Watch(id=uuid.uuid4().hex[:6], kind=kind, what=req["what"], created=now, data=data,
                      last_check=now)
            self.watches[w.id] = w
            self.stats["set"] += 1
            self.save()
        return {"process": f"Okay, I'll tell you when {req['what']} closes.",
                "download": "Okay, I'll tell you when the download finishes.",
                "battery": f"Okay, I'll tell you when {req['what']}.",
                "cpu": "Okay, I'll tell you when the CPU calms down.",
                "file": f"Okay, I'll tell you when {req['what']}."}[kind]

    def cancel(self, what: str, now: float) -> str:
        with self._lock:
            hit = [w for w in self.watches.values() if w.status == "active"
                   and (not what or what in w.what.lower() or what == w.kind)]
            for w in hit:
                w.status = "cancelled"
                self.stats["cancelled"] += 1
            self.save()
        if not hit:
            return "I wasn't watching anything like that."
        return "Okay, I've stopped watching." if len(hit) > 1 or not what else f"Okay, I'll stop watching {hit[0].what}."

    # -- checking -------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> List[str]:
        """Check what is due. Returns the things to say, each once."""
        if not self.cfg.enabled:
            return []
        now = self.clock() if now is None else now
        out: List[str] = []
        with self._lock:
            changed = False
            for w in list(self.watches.values()):
                if w.status != "active":
                    continue
                if now - w.created > self.cfg.expire_s:
                    w.status = "expired"
                    self.stats["expired"] += 1
                    changed = True
                    continue
                if now - w.last_check < self.cfg.check_every_s:
                    continue
                w.last_check = now
                try:
                    say = self._check(w, now)
                except Exception:
                    logger.exception("A watch check failed")
                    say = None
                if say:
                    w.status = "done"
                    self.stats["fired"] += 1
                    out.append(say)
                    changed = True
            trend = self._disk_trend(now)
            if trend:
                out.append(trend)
                changed = True
            if changed:
                self.save()
        return out

    def _check(self, w: Watch, now: float) -> Optional[str]:
        p, d = self.probe, w.data
        if w.kind == "process":
            running = p.processes()
            if running is not None and not _proc_matches(d["proc"], running):
                return f"{d['proc']} has closed."
        elif w.kind == "download":
            listing = p.listdir(d["dir"])
            if listing is None:
                return None
            base = set(d.get("baseline") or [])
            partial_now = [n for n in listing if n.lower().endswith(PARTIAL)]
            new = {n: v for n, v in listing.items() if n not in base and not n.lower().endswith(PARTIAL)}
            settled = [n for n, (_size, mt) in new.items() if now - mt >= self.cfg.download_settle_s]
            if settled and not partial_now:
                names = sorted(settled)
                return ("Your download finished: " + names[0] + "." if len(names) == 1
                        else f"Your downloads finished: {', '.join(names[:3])}.")
        elif w.kind == "battery":
            b = p.battery()
            if b is not None and b[0] >= d["pct"]:
                return "The battery is charged." if d["pct"] >= 100 else f"The battery is at {b[0]:.0f}%."
        elif w.kind == "cpu":
            c = p.cpu()
            if c is None:
                return None
            if c < d.get("below", self.cfg.cpu_calm_below):
                since = d.setdefault("calm_since", now)
                if now - since >= self.cfg.cpu_calm_hold_s:
                    return f"The CPU has calmed down ({c:.0f}%)."
            else:
                d.pop("calm_since", None)
        elif w.kind == "file":
            mt = p.mtime(d["path"])
            if mt is not None and mt > d.get("mtime", 0.0) + 0.5:
                return f"{os.path.basename(d['path'])} has changed."
        return None

    def _disk_trend(self, now: float) -> Optional[str]:
        if not self.cfg.disk_trend:
            return None
        if self.disk_samples and now - self.disk_samples[-1][0] < self.cfg.disk_sample_s:
            return None
        du = self.probe.disk()
        if du is None:
            return None
        used, total = du
        self.disk_samples.append((now, used))
        window = [s for s in self.disk_samples if now - s[0] <= 12 * HOUR]
        if len(window) < 3 or now - self.disk_warned < DAY:
            return None
        (t0, u0), (t1, u1) = window[0], window[-1]
        if t1 - t0 < 2 * HOUR or u1 <= u0:
            return None
        rate = (u1 - u0) / (t1 - t0)                 # bytes per second
        left = (total - used) / rate
        if left > self.cfg.disk_warn_s:
            return None
        self.disk_warned = now
        self.stats["trend_warnings"] += 1
        hours = left / HOUR
        when = f"about {hours:.0f} hours" if hours >= 2 else "under two hours"
        return f"Your disk is filling up fast: at this rate it will be full in {when}."

    # -- inspection ------------------------------------------------------------
    def active(self) -> List[Watch]:
        with self._lock:
            return [w for w in self.watches.values() if w.status == "active"]

    def report(self) -> str:
        rows = self.active()
        lines = [f"Watching: {len(rows)}."]
        for w in rows:
            lines.append(f"  {w.id}  {w.kind:<8} {w.what}")
        return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "active": len(self.active()), **self.stats}
