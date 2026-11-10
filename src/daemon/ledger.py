"""
src/daemon/ledger.py

What the user did on the web, as a local ledger.

One row per stretch of attention ("episode"): when, which site, what category,
how long. Plain SQLite on this machine, no server, no model. It keeps:

    episodes   one row per stretch. Raw, so it expires: `retention_days` (30).
               Page titles are NOT stored unless keep_titles is on.
    daily      per day and category: seconds, episodes, and churn (tabs seen
               and dropped within seconds). Aggregates, kept for good.
    sites      per site and category over all time: seconds and visits, with
               the last day it was seen. It is what "interests" is built from.

`rollup()` turns expired episodes into daily/sites aggregates and deletes the
raw rows; nothing about a single visit outlives the retention window.
Forgetting is real: `forget(site=...)`, `forget(category=...)` and
`forget_all()` delete the rows and subtract nothing -- they remove the data.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.ledger")

DAY = 86400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start REAL NOT NULL, end REAL NOT NULL,
    site TEXT NOT NULL, category TEXT NOT NULL, source TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '', audible INTEGER NOT NULL DEFAULT 0, tabs INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ep_start ON episodes(start);
CREATE TABLE IF NOT EXISTS daily (
    day TEXT NOT NULL, category TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0, episodes INTEGER NOT NULL DEFAULT 0, churn INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, category)
);
CREATE TABLE IF NOT EXISTS sites (
    site TEXT NOT NULL, category TEXT NOT NULL,
    seconds REAL NOT NULL DEFAULT 0, visits INTEGER NOT NULL DEFAULT 0, last_day TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (site, category)
);
"""


def day_of(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class Ledger:
    def __init__(self, path: Optional[str] = None, retention_days: float = 30.0, keep_titles: bool = False,
                 clock=time.time):
        self.path = path or ":memory:"
        self.retention_days = float(retention_days)
        self.keep_titles = bool(keep_titles)
        self.clock = clock
        self._lock = threading.RLock()
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self):
        with self._lock:
            try:
                self.db.close()
            except Exception:
                pass

    # -- writing ------------------------------------------------------------------
    def record(self, start: float, end: float, site: str, category: str, source: str = "", title: str = "",
               audible: bool = False, tabs: int = 0) -> bool:
        """One stretch of attention. Returns False for an empty or backwards one."""
        if not site or end <= start:
            return False
        with self._lock:
            self.db.execute(
                "INSERT INTO episodes(start,end,site,category,source,title,audible,tabs) VALUES (?,?,?,?,?,?,?,?)",
                (float(start), float(end), site, category, source, (title[:120] if self.keep_titles else ""),
                 1 if audible else 0, int(tabs)))
            self.db.commit()
        return True

    def churn(self, n: int = 1, ts: Optional[float] = None):
        """Tabs that were opened and left within seconds: a signal, never a page."""
        if n <= 0:
            return
        d = day_of(self.clock() if ts is None else ts)
        with self._lock:
            self.db.execute("INSERT INTO daily(day,category,seconds,episodes,churn) VALUES (?, '_churn', 0, 0, ?) "
                            "ON CONFLICT(day,category) DO UPDATE SET churn = churn + excluded.churn", (d, int(n)))
            self.db.commit()

    # -- reading ------------------------------------------------------------------
    def episodes(self, since: float = 0.0, until: Optional[float] = None, limit: int = 5000) -> List[Dict[str, Any]]:
        until = self.clock() + 1 if until is None else until
        with self._lock:
            rows = self.db.execute("SELECT start,end,site,category,source,title,audible,tabs FROM episodes "
                                   "WHERE start >= ? AND start < ? ORDER BY start LIMIT ?",
                                   (since, until, limit)).fetchall()
        return [dict(start=r[0], end=r[1], site=r[2], category=r[3], source=r[4], title=r[5], audible=bool(r[6]),
                     tabs=r[7]) for r in rows]

    def by_category(self, days: float = 7.0) -> Dict[str, float]:
        """Seconds per category over the last `days` days (raw and rolled-up together)."""
        now = self.clock()
        first = day_of(now - days * DAY)
        out: Dict[str, float] = {}
        with self._lock:
            for cat, secs in self.db.execute("SELECT category, SUM(seconds) FROM daily WHERE day >= ? AND category != '_churn' "
                                              "GROUP BY category", (first,)):
                out[cat] = out.get(cat, 0.0) + float(secs or 0)
            for cat, secs in self.db.execute("SELECT category, SUM(end-start) FROM episodes WHERE start >= ? GROUP BY category",
                                              (now - days * DAY,)):
                out[cat] = out.get(cat, 0.0) + float(secs or 0)
        return out

    def top_sites(self, limit: int = 8, category: str = "") -> List[Tuple[str, str, float, int]]:
        """(site, category, seconds, visits) over all time, raw episodes included."""
        with self._lock:
            rows = self.db.execute("SELECT site,category,seconds,visits FROM sites").fetchall()
            raw = self.db.execute("SELECT site,category,SUM(end-start),COUNT(*) FROM episodes GROUP BY site,category").fetchall()
        agg: Dict[Tuple[str, str], List[float]] = {}
        for s, c, secs, v in list(rows) + list(raw):
            a = agg.setdefault((s, c), [0.0, 0])
            a[0] += float(secs or 0)
            a[1] += int(v or 0)
        out = [(s, c, a[0], int(a[1])) for (s, c), a in agg.items() if not category or c == category]
        out.sort(key=lambda r: -r[2])
        return out[:limit]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            e = self.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            d = self.db.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
            s = self.db.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
        return {"episodes": e, "daily": d, "sites": s}

    # -- aging ---------------------------------------------------------------------
    def rollup(self, now: Optional[float] = None) -> int:
        """Fold expired episodes into the aggregates and delete them. Returns how many."""
        now = self.clock() if now is None else now
        cut = now - self.retention_days * DAY
        with self._lock:
            rows = self.db.execute("SELECT start,end,site,category FROM episodes WHERE start < ?", (cut,)).fetchall()
            for start, end, site, cat in rows:
                d = day_of(start)
                self.db.execute("INSERT INTO daily(day,category,seconds,episodes,churn) VALUES (?,?,?,1,0) "
                                "ON CONFLICT(day,category) DO UPDATE SET seconds = seconds + excluded.seconds, "
                                "episodes = episodes + 1", (d, cat, end - start))
                self.db.execute("INSERT INTO sites(site,category,seconds,visits,last_day) VALUES (?,?,?,1,?) "
                                "ON CONFLICT(site,category) DO UPDATE SET seconds = seconds + excluded.seconds, "
                                "visits = visits + 1, last_day = MAX(last_day, excluded.last_day)",
                                (site, cat, end - start, d))
            self.db.execute("DELETE FROM episodes WHERE start < ?", (cut,))
            self.db.commit()
        return len(rows)

    # -- forgetting ----------------------------------------------------------------
    def forget(self, site: str = "", category: str = "") -> int:
        """Delete everything about a site or a category. Returns rows removed."""
        n = 0
        with self._lock:
            if site:
                for t in ("episodes", "sites"):
                    n += self.db.execute(f"DELETE FROM {t} WHERE site = ?", (site,)).rowcount
            if category:
                for t in ("episodes", "sites", "daily"):
                    n += self.db.execute(f"DELETE FROM {t} WHERE category = ?", (category,)).rowcount
            self.db.commit()
        return n

    def forget_all(self) -> int:
        with self._lock:
            n = sum(self.db.execute(f"DELETE FROM {t}").rowcount for t in ("episodes", "daily", "sites"))
            self.db.commit()
            try:
                self.db.execute("VACUUM")
            except Exception:
                pass
        return n
