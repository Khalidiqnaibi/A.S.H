"""
src/daemon/unrouted.py

Which lines reached the language model, and why.

The model is meant to be the last resort -- a server of information when nothing else (a rule, a tool, memory, what ASH keeps about the
user, its own record of what it did) can answer. Measured on the real sessions of 17-23 Nov, 62 of 112 addressed lines went to it, and about
one in ten needed to. The rest were commands, questions about ASH and its memory, short replies, complaints, statements: each reached the model
because no rule had been written for that exact wording yet, and nothing recorded that it had.

This is that record. One line per model turn in state/unrouted.jsonl (the words, the reason it fell through, whether a tool ran, how long it
took), capped, scrubbed like the journal, never for a guest. `common()` groups the lines by meaning (the encoder ASH already loads), so the
repeated ones are the next things to answer without the model; `rate()` is the share of turns the model answered, and is what a test can hold down.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.unrouted")

MAX_LINES = 500


def reason_of(response: Any) -> str:
    """Why this turn went to the model, as a short code. From what the brain wrote about the turn."""
    tool = str(getattr(response, "tool_used", "") or "")
    notes = " ".join(str(n) for n in (getattr(getattr(response, "trace", None), "notes", None) or []))
    low = notes.lower()
    if tool:
        return "tool_narrated"
    if "no proposal cleared" in low:
        return "no_tool_matched"
    if "deadlock" in low or "clarif" in low:
        return "ambiguous"
    if "veto" in low or "blocked" in low:
        return "blocked"
    if "escalat" in low:
        return "escalated"
    return "conversation"


class Unrouted:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time,
                 scrub: Optional[Callable[[str], str]] = None, max_lines: int = MAX_LINES):
        self.path = path
        self.clock = clock
        self.scrub = scrub or (lambda s: s)
        self.max_lines = max_lines
        self._lock = threading.Lock()
        self.rows: List[Dict[str, Any]] = []
        self.stats = {"turns": 0, "model": 0}
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self.rows.append(json.loads(line))
                    except ValueError:
                        continue
            self.rows = self.rows[-self.max_lines:]
        except OSError:
            logger.debug("unrouted ledger not read", exc_info=True)

    def answered_elsewhere(self):
        """A turn answered by a rule, a tool or memory: it counts in the rate, and is not written."""
        self.stats["turns"] += 1

    def model_turn(self, text: str, reason: str, took_s: float = 0.0, tool: bool = False, guest: bool = False):
        self.stats["turns"] += 1
        self.stats["model"] += 1
        if guest or not (text or "").strip():
            return
        row = {"t": round(self.clock(), 1), "text": str(self.scrub(text))[:300], "why": reason, "tool": bool(tool),
               "took_s": round(float(took_s), 1)}
        with self._lock:
            self.rows.append(row)
            if len(self.rows) > self.max_lines:
                self.rows = self.rows[-self.max_lines:]
            self._write(row)

    def _write(self, row: Dict[str, Any]):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            if len(self.rows) >= self.max_lines and os.path.exists(self.path) and os.path.getsize(self.path) > self.max_lines * 400:
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    for r in self.rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(tmp, self.path)
                return
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            logger.debug("unrouted line not written", exc_info=True)

    def rate(self) -> float:
        """The share of this run's turns the model answered (0..1)."""
        t = self.stats["turns"]
        return self.stats["model"] / t if t else 0.0

    def common(self, embed: Optional[Callable[[List[str]], Any]] = None, limit: int = 8, min_size: int = 2) -> List[Dict[str, Any]]:
        """The lines that keep reaching the model, grouped by meaning: [{"example", "n", "why", "texts"}], biggest first. With no
        encoder, the same words (lowercased) are the group."""
        rows = list(self.rows)
        if not rows:
            return []
        groups: List[Dict[str, Any]] = []
        vecs = None
        if embed is not None:
            try:
                import numpy as np
                m = np.asarray(embed([r["text"] for r in rows]), dtype="float32")
                n = np.linalg.norm(m, axis=1, keepdims=True)
                vecs = m / np.where(n == 0, 1.0, n)
            except Exception:
                logger.debug("could not embed the unrouted lines", exc_info=True)
                vecs = None
        for i, r in enumerate(rows):
            placed = False
            for g in groups:
                same = (float(vecs[i] @ vecs[g["i0"]]) >= 0.72) if vecs is not None else (r["text"].lower().strip(" .!?") == g["example"].lower().strip(" .!?"))
                if same:
                    g["n"] += 1
                    g["texts"].append(r["text"])
                    placed = True
                    break
            if not placed:
                groups.append({"i0": i, "example": r["text"], "n": 1, "why": r.get("why", ""), "texts": [r["text"]]})
        groups = [g for g in groups if g["n"] >= min_size]
        groups.sort(key=lambda g: -g["n"])
        for g in groups:
            g.pop("i0", None)
        return groups[:limit]

    def report(self, embed: Optional[Callable[[List[str]], Any]] = None) -> str:
        by: Dict[str, int] = {}
        for r in self.rows:
            by[r.get("why", "?")] = by.get(r.get("why", "?"), 0) + 1
        head = (f"The model answered {self.stats['model']} of {self.stats['turns']} turns this run; "
                f"{len(self.rows)} lines kept (" + ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])) + ").")
        lines = [head]
        for g in self.common(embed):
            lines.append(f"  x{g['n']}  {g['example'][:80]}   [{g['why']}]")
        return "\n".join(lines)

    def status(self) -> Dict[str, Any]:
        return {"turns": self.stats["turns"], "model": self.stats["model"], "kept": len(self.rows), "rate": round(self.rate(), 3)}
