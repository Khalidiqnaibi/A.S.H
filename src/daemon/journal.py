"""
src/daemon/journal.py

The raw ambient stream, and how it becomes memory.

An always-on ASH observes something like 5,000-20,000 events a day. Writing
each one into episodic memory would destroy that store's usefulness inside a
week -- retrieval would return "switched to Chrome" forty times for every
query. So ambient perception uses a two-stage memory path that mirrors how
biological consolidation actually works:

    STAGE 1 (continuous)  Every event appends one line to a daily JSONL
                          journal. Cheap, append-only, TTL-bounded, never
                          searched during a live turn.

    STAGE 2 (sleep)       During the maintenance phase, the journal is rolled
                          up: events are bucketed into activity blocks,
                          summarized, and only the summaries are written to
                          episodic memory as durable episodes.

So the day is remembered as "worked in VS Code on the brain package for two
hours, three short breaks, one call at 14:00" rather than as 6,000 window
switches. The raw journal remains available for a week if something needs to
be reconstructed precisely, then it expires.

Rollup is deliberately deterministic -- bucketing and counting, no LLM. It
runs while idle on a machine that may be doing something else, and a
summarizer that costs nothing and never hallucinates is worth more here than
a prettier sentence.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.journal")

JOURNAL_DIR = os.environ.get("ASH_JOURNAL_DIR", os.path.join(os.getcwd(), "state", "journal"))

# Events closer together than this belong to the same activity block.
BLOCK_GAP_SECONDS = 12 * 60


class Journal:
    """Append-only daily event log with deterministic rollup."""

    def __init__(self, directory: str = JOURNAL_DIR, flush_every: int = 20):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self.flush_every = flush_every
        self.written = 0
        self._last_rollup: Optional[str] = None

    # ------------------------------------------------------------------
    def _path(self, ts: Optional[float] = None) -> str:
        day = datetime.fromtimestamp(ts or time.time()).strftime("%Y-%m-%d")
        return os.path.join(self.dir, f"{day}.jsonl")

    def append(self, event) -> None:
        if not getattr(event, "persistable", True):
            return
        rec = event.as_record()
        with self._lock:
            self._buffer.append(rec)
            self.written += 1
            if len(self._buffer) >= self.flush_every:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        # Group by day so a buffer spanning midnight lands in both files.
        by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self._buffer:
            by_day[self._path(r["ts"])].append(r)
        for path, rows in by_day.items():
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Journal write failed for %s", path)
        self._buffer.clear()

    # ------------------------------------------------------------------
    def read_day(self, day: Optional[str] = None) -> List[Dict[str, Any]]:
        day = day or datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.dir, f"{day}.jsonl")
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def days(self) -> List[str]:
        return sorted(f[:-6] for f in os.listdir(self.dir) if f.endswith(".jsonl"))

    # ------------------------------------------------------------------
    # Rollup
    # ------------------------------------------------------------------
    @staticmethod
    def _blocks(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split a day into activity blocks separated by quiet gaps."""
        rows = sorted(rows, key=lambda r: r["ts"])
        blocks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        last_ts = None
        for r in rows:
            if last_ts is not None and r["ts"] - last_ts > BLOCK_GAP_SECONDS:
                if current:
                    blocks.append(current)
                current = []
            current.append(r)
            last_ts = r["ts"]
        if current:
            blocks.append(current)
        return blocks

    @staticmethod
    def summarize_block(block: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic summary of one activity block."""
        start, end = block[0]["ts"], block[-1]["ts"]
        by_source = Counter(r["source"] for r in block)

        # What the user was actually in. Window titles are the most
        # informative field by a wide margin.
        apps = Counter()
        for r in block:
            proc = (r.get("meta") or {}).get("process")
            if proc:
                apps[proc] += 1
            elif r["source"] == "active_window":
                apps[r["text"][:40]] += 1

        spoken = [r["text"] for r in block if r["source"] == "microphone"][:12]
        alerts = [r["text"] for r in block if r.get("urgency", 0) >= 0.5][:8]
        peak = max((r.get("salience", 0) for r in block), default=0.0)

        duration = end - start
        parts = [
            f"{datetime.fromtimestamp(start).strftime('%H:%M')}-"
            f"{datetime.fromtimestamp(end).strftime('%H:%M')} "
            f"({duration / 60:.0f} min)"
        ]
        if apps:
            top = ", ".join(f"{a}" for a, _ in apps.most_common(4))
            parts.append(f"mostly in {top}")
        if spoken:
            parts.append(f"heard: {' / '.join(s[:70] for s in spoken[:3])}")
        if alerts:
            parts.append(f"alerts: {'; '.join(alerts[:2])}")

        return {
            "start": start,
            "end": end,
            "duration_min": round(duration / 60, 1),
            "events": len(block),
            "sources": dict(by_source),
            "apps": [a for a, _ in apps.most_common(6)],
            "summary": " | ".join(parts),
            # Importance drives whether the episode survives pruning. A block
            # with speech or alerts in it mattered; forty window switches did
            # not.
            "importance": min(0.95, 0.25 + 0.25 * bool(spoken) + 0.3 * bool(alerts)
                              + 0.2 * peak),
        }

    def rollup(self, day: Optional[str] = None, episodic=None,
               min_events: int = 3) -> Dict[str, Any]:
        """Consolidate one day of raw events into episodic memory.

        Idempotent per day: a marker file records that a day has been rolled
        up, so a scheduler that fires twice doesn't duplicate every episode.
        """
        day = day or datetime.now().strftime("%Y-%m-%d")
        marker = os.path.join(self.dir, f".{day}.rolled")
        if os.path.exists(marker):
            return {"day": day, "skipped": "already rolled up"}

        self.flush()
        rows = self.read_day(day)
        if len(rows) < min_events:
            return {"day": day, "skipped": f"only {len(rows)} events"}

        blocks = self._blocks(rows)
        summaries = [self.summarize_block(b) for b in blocks if len(b) >= min_events]

        written = 0
        if episodic is not None:
            for s in summaries:
                try:
                    episodic.add_episode(
                        summary=s["summary"],
                        event_type="ambient_block",
                        related_entities=s["apps"][:4],
                        importance=s["importance"],
                    )
                    written += 1
                except Exception:
                    logger.exception("Failed to write rollup episode")

        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"rolled_at": time.time(), "blocks": len(summaries)}))
        except Exception:
            logger.exception("Failed to write rollup marker")

        logger.info("Journal rollup %s: %d events -> %d block(s) -> %d episode(s)",
                    day, len(rows), len(summaries), written)
        return {
            "day": day, "events": len(rows), "blocks": len(summaries),
            "episodes_written": written,
            "summaries": [s["summary"] for s in summaries],
        }

    def rollup_pending(self, episodic=None) -> List[Dict[str, Any]]:
        """Roll up every complete day that hasn't been done yet. Today is
        skipped -- it isn't over."""
        today = datetime.now().strftime("%Y-%m-%d")
        out = []
        for day in self.days():
            if day >= today:
                continue
            r = self.rollup(day, episodic=episodic)
            if not r.get("skipped"):
                out.append(r)
        return out

    # ------------------------------------------------------------------
    def recent(self, seconds: float = 3600, limit: int = 50) -> List[Dict[str, Any]]:
        """Raw events from the last N seconds. Used to give a response
        conversational context about what just happened."""
        self.flush()
        cutoff = time.time() - seconds
        rows = [r for r in self.read_day() if r["ts"] >= cutoff]
        return rows[-limit:]

    def context_block(self, seconds: float = 1800, limit: int = 12) -> str:
        """Compact text of what has been going on, for the LLM prompt."""
        rows = self.recent(seconds, limit)
        if not rows:
            return ""
        lines = []
        for r in rows:
            when = datetime.fromtimestamp(r["ts"]).strftime("%H:%M")
            lines.append(f"{when} [{r['source']}] {r['text'][:120]}")
        return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        return {
            "dir": self.dir,
            "written": self.written,
            "buffered": len(self._buffer),
            "days": self.days()[-7:],
        }
