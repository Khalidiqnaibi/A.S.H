"""
src/senses/browsertabs.py

What is open in the user's browser, from a small extension -- no model, no screen.

A window title only says which tab is in front. A browser extension (see
extension/ash-tabs) reports, on every change and every 30 seconds, the list of
tabs and which one is active. It posts to 127.0.0.1 only, with a token that ASH
wrote into the extension's own config, so nothing else on the machine can feed
it and nothing leaves the machine.

What is kept of a tab is deliberately little:

    * the host, never the path, query or fragment
    * a category (pagekind.py: table, keywords, what the user taught it)
    * how long it was in front of a focused browser window
    * that tabs were opened and dropped within seconds (churn), not which

What is never kept: incognito/private windows; banks, health, mail, chat, adult,
password managers, private-network pages and pages whose title says
"symptoms" or "password" (pagekind.is_sensitive) -- those are dropped on arrival
and only counted; the page text; the form fields. Titles are used in memory to
classify and are stored only if ledger keep_titles is on.

An episode is the time one site was the active tab of a focused browser window.
A switch, the browser losing focus for `grace_s`, or the extension going quiet
for `stale_s` ends it. Under `min_s` it is churn, not a visit.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from .base import Capability, Modality, Sensor, SensorEvent

logger = logging.getLogger("ash.senses.browsertabs")

_EXT_ORIGINS = ("chrome-extension://", "moz-extension://", "edge-extension://", "safari-web-extension://")
MAX_BODY = 262144
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_LONGNUM = re.compile(r"\d{5,}")


def load_or_create_token(path: str) -> str:
    """The shared secret between ASH and its extension, created once."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tok = fh.read().strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(24)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tok)
    return tok


def write_extension_config(ext_dir: str, port: int, token: str) -> bool:
    """Tell the extension where ASH listens and the token to send. False if there is no extension folder."""
    if not os.path.isdir(ext_dir):
        return False
    try:
        with open(os.path.join(ext_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"port": int(port), "token": token}, fh)
        return True
    except OSError:
        return False


def tidy_title(title: str, limit: int = 100) -> str:
    t = _EMAIL.sub("", str(title or ""))
    t = _LONGNUM.sub("", t)
    return re.sub(r"\s+", " ", t).strip()[:limit]


class BrowserTabSensor(Sensor):
    capability = Capability(
        name="browser_tabs",
        description="Which sites are open and in front in the browser, from a local extension. Hosts only.",
        privacy_sensitive=True,
    )
    default_interval = 2.0
    modality = Modality.TEXT

    def __init__(self, interval: Optional[float] = None, enabled: bool = True, classifier=None, ledger=None,
                 clock: Callable[[], float] = time.time, port: int = 0, token: str = "",
                 min_s: float = 3.0, grace_s: float = 15.0, stale_s: float = 90.0, churn_s: float = 10.0,
                 idle_s: Optional[Callable[[], float]] = None, idle_after_s: float = 180.0,
                 scrub: Optional[Callable[[str], str]] = None, listen: bool = False):
        from src.daemon.pagekind import Classifier   # lazy: senses must not import the daemon at load time
        self.cls = classifier if classifier is not None else Classifier()
        self.ledger = ledger
        self.clock = clock
        self.port = int(port)
        self.token = token
        self.min_s, self.grace_s, self.stale_s, self.churn_s = float(min_s), float(grace_s), float(stale_s), float(churn_s)
        self.idle_s, self.idle_after_s = idle_s, float(idle_after_s)
        self.scrub = scrub
        self.listen = bool(listen)
        self.paused = False
        self.dropped: Dict[str, int] = {"private": 0, "sensitive": 0, "scheme": 0}
        self.tabs_n = 0
        self.stats = {"reports": 0, "episodes": 0, "churn": 0, "rejected": 0}
        self._lock = threading.RLock()
        self._pending: deque = deque(maxlen=64)
        self._cur: Optional[Dict[str, Any]] = None       # the open episode
        self._lost: Optional[float] = None               # when the active tab was lost, if it was
        self._last_report = 0.0
        self._born: Dict[Any, float] = {}                # tab id -> first seen (ids only, no pages)
        self._first = True
        self._server: Optional[ThreadingHTTPServer] = None
        self._last_rollup = 0.0
        super().__init__(interval=interval, enabled=enabled)

    # -- what it is looking at ---------------------------------------------------
    @property
    def now_site(self) -> str:
        c = self._cur
        return c["site"] if c else ""

    def current(self) -> Optional[Dict[str, Any]]:
        c = self._cur
        return dict(c) if c else None

    def teach_current(self, category: str) -> Optional[str]:
        """"That's work": the site in front is `category`, from now on and for pages like it."""
        c = self._cur
        if not c or not self.cls.teach(c["host"], category, c.get("title", "")):
            return None
        c["category"], c["source"] = category, "user"
        return c["site"]

    # -- receiving ----------------------------------------------------------------
    def submit(self, report: Dict[str, Any]) -> bool:
        """A report from the extension: {"focused": bool, "tabs": [{id, url, title, active, audible, incognito}]}."""
        if self.paused or not isinstance(report, dict) or not isinstance(report.get("tabs"), list):
            self.stats["rejected"] += 1
            return False
        with self._lock:
            self._pending.append((self.clock(), report))
        return True

    # -- cleaning -------------------------------------------------------------------
    def sanitize(self, tab: Any) -> Optional[Dict[str, Any]]:
        from src.daemon.pagekind import host_of, is_sensitive, site_of
        if not isinstance(tab, dict):
            return None
        if tab.get("incognito"):
            self.dropped["private"] += 1
            return None
        url = str(tab.get("url") or "")
        if not re.match(r"^https?://", url, re.I):
            self.dropped["scheme"] += 1
            return None
        host = host_of(url)
        raw_title = str(tab.get("title") or "")
        if is_sensitive(host, raw_title):
            self.dropped["sensitive"] += 1
            return None
        title = tidy_title(raw_title)
        if self.scrub is not None:
            try:
                title = self.scrub(title)
            except Exception:
                title = ""
        cat, src = self.cls.classify(host, title)
        return {"id": tab.get("id"), "host": host, "site": site_of(host), "title": title, "category": cat,
                "source": src, "active": bool(tab.get("active")), "audible": bool(tab.get("audible"))}

    # -- the loop ------------------------------------------------------------------------
    def read(self) -> List[SensorEvent]:
        if self.listen and self._server is None:
            self.start_listener()
        now = self.clock()
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        events: List[SensorEvent] = []
        for ts, rep in pending:
            try:
                events += self._ingest(ts, rep)
            except Exception:
                logger.exception("Could not read a tab report")
        events += self._tick(now)
        return events

    def _ingest(self, now: float, rep: Dict[str, Any]) -> List[SensorEvent]:
        self.stats["reports"] += 1
        self._last_report = now
        raw = [t for t in rep["tabs"][:300] if isinstance(t, dict)]
        self.tabs_n = len(raw)
        self._count_churn(now, raw)
        clean = [c for c in (self.sanitize(t) for t in raw) if c]
        active_raw = next((t for t in raw if t.get("active")), None)
        focused = bool(rep.get("focused", True)) and not self._idle()
        cand = None
        if focused and active_raw is not None:
            cand = next((c for c in clean if c["active"]), None)     # None if the active tab was dropped
        out: List[SensorEvent] = []
        if cand is None:
            if self._cur is not None and self._lost is None:
                self._lost = now
                if active_raw is not None and focused:               # a page we must not record: end at once
                    out += self._finish(now, "stopped")
            return out
        self._lost = None
        cur = self._cur
        if cur is not None and cur["site"] == cand["site"]:
            cur["last"] = now
            cur["audible"] = cur["audible"] or cand["audible"]
            cur["title"] = cand["title"]
            if cand["source"] == "user" or cur["category"] == "unknown":
                cur["category"], cur["source"] = cand["category"], cand["source"]
            return out
        out += self._finish(now, "changed" if cur is not None else "started")
        self._cur = {"site": cand["site"], "host": cand["host"], "title": cand["title"], "category": cand["category"],
                     "source": cand["source"], "start": now, "last": now, "audible": cand["audible"]}
        out.append(self._event("changed" if cur is not None else "started", now, None))
        return out

    def _idle(self) -> bool:
        if self.idle_s is None:
            return False
        try:
            return float(self.idle_s()) >= self.idle_after_s
        except Exception:
            return False

    def _count_churn(self, now: float, raw: List[Dict[str, Any]]):
        ids = {t.get("id") for t in raw if t.get("id") is not None}
        if self._first:
            self._born = {i: now - 1e6 for i in ids}                  # what was already open is not new
            self._first = False
            return
        short = 0
        for i in list(self._born):
            if i not in ids:
                if now - self._born.pop(i) < self.churn_s:
                    short += 1
        for i in ids:
            self._born.setdefault(i, now)
        if short:
            self.stats["churn"] += short
            if self.ledger is not None:
                self.ledger.churn(short, now)

    def _tick(self, now: float) -> List[SensorEvent]:
        out: List[SensorEvent] = []
        c = self._cur
        if c is not None:
            if self._lost is not None and now - self._lost >= self.grace_s:
                out += self._finish(self._lost, "stopped")
            elif now - self._last_report >= self.stale_s:
                out += self._finish(self._last_report, "stopped")
        if self.ledger is not None and now - self._last_rollup >= 6 * 3600:
            self._last_rollup = now
            try:
                self.ledger.rollup(now)
            except Exception:
                logger.exception("Ledger rollup failed")
        return out

    def _finish(self, end: float, how: str) -> List[SensorEvent]:
        c, self._cur = self._cur, None
        if c is None:
            return []
        secs = max(0.0, end - c["start"])
        done = {"site": c["site"], "category": c["category"], "secs": secs, "how": how}
        # A glance shorter than min_s is simply not a visit. (It used to also be counted as churn, which counted a
        # tab opened, glanced at and closed twice, and counted quick switches between tabs that were never closed.
        # Churn is only a tab that came and went: _count_churn.)
        if secs >= self.min_s:
            self.stats["episodes"] += 1
            if self.ledger is not None:
                self.ledger.record(c["start"], end, c["site"], c["category"], c["source"], c.get("title", ""),
                                   c.get("audible", False), self.tabs_n)
        if how == "stopped":
            return [self._event("stopped", end, done)]
        self._finished = done                            # picked up by the event that follows
        return []

    def _event(self, what: str, now: float, done: Optional[Dict[str, Any]]) -> SensorEvent:
        done = done or getattr(self, "_finished", None)
        self._finished = None
        meta: Dict[str, Any] = {"event": what, "tabs": self.tabs_n}
        c = self._cur
        if c is not None:
            meta.update(site=c["site"], category=c["category"], how=c["source"], audible=bool(c["audible"]))
        if done is not None:
            meta.update(finished_site=done["site"], finished_category=done["category"], finished_s=int(done["secs"]))
        text = (f"browsing {c['site']} ({c['category']})" if c is not None else "left the browser")
        return SensorEvent(source=self.name, modality=Modality.TEXT, text=text, salience=0.05, ts=now,
                           persistable=False, meta=meta)

    # -- control ---------------------------------------------------------------------
    def pause(self):
        """Stop watching. Anything in progress is closed; nothing more is read until resume()."""
        with self._lock:
            self.paused = True
            self._pending.clear()
        self._finish(self.clock(), "stopped")
        self._lost = None

    def resume(self):
        self.paused = False

    def status(self) -> Dict[str, Any]:
        return {"paused": self.paused, "listening": self._server is not None, "port": self.port,
                "site": self.now_site, "tabs": self.tabs_n, "dropped": dict(self.dropped), **self.stats}

    # -- the loopback listener ----------------------------------------------------------
    def start_listener(self) -> bool:
        if self._server is not None or not self.token:
            return self._server is not None
        sensor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):          # silence
                pass

            def _origin_ok(self) -> bool:
                o = self.headers.get("Origin", "")
                return (not o) or o.startswith(_EXT_ORIGINS)

            def _cors(self):
                o = self.headers.get("Origin", "")
                if o.startswith(_EXT_ORIGINS):
                    self.send_header("Access-Control-Allow-Origin", o)
                    self.send_header("Vary", "Origin")

            def do_OPTIONS(self):
                if not self._origin_ok():
                    self.send_response(403)
                    self.end_headers()
                    return
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "content-type, x-ash-token")
                self.end_headers()

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    n = -1
                body = b""
                if 0 < n <= MAX_BODY:
                    body = self.rfile.read(n)             # read what was sent before answering, or the client sees a reset
                ok = (self.path == "/tabs" and self._origin_ok()
                      and hmac.compare_digest(self.headers.get("X-ASH-Token", ""), sensor.token))
                if not ok:
                    sensor.stats["rejected"] += 1
                    self.send_response(403)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    return
                try:
                    if not body:
                        raise ValueError("size")
                    sensor.submit(json.loads(body.decode("utf-8")))
                    self.send_response(204)
                except Exception:
                    sensor.stats["rejected"] += 1
                    self.send_response(400)
                self._cors()
                self.end_headers()

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as e:
            logger.warning("Browser tab listener could not start on port %s: %s", self.port, e)
            self._server = None
            return False
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, name="ash-tabs", daemon=True).start()
        return True

    def stop_listener(self):
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
