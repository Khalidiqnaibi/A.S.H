"""
src/daemon/remarks.py

A log of everything ASH says, and how it landed -- the instrument for tuning
ASH on real days instead of guesses.

Every line that leaves ASH is written down (time, what kind of line it was,
whether the user had asked, a shortened copy of the text), and so is the
user's reaction when the reaction tracker reads one (engaged, accepted,
rejected, deferred, dismissed, silence). `ashctl review` reads it back:

    how much ASH said, asked and unasked, per hour and per kind
    how each kind landed
    lines said over and over, remarks in the small hours, the busiest hour
    and, by rule, what to tune: "jokes were waved off 6 of 8 times --
    consider raising their bar or turning them off"

One JSON line per event, a file per day, under state/remarks/, kept for two
weeks. `keep_text: false` keeps only the kind and the reaction, never the
words.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.remarks")

# The reaction tracker's outcomes (social.REWARD): waved off, and met with silence.
NEGATIVE = ("rejected", "dismissed")
UNANSWERED = ("silence",)


def _id(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()[:10]


def kind_of(meta: Dict[str, Any]) -> str:
    """What kind of line this was, from what the runtime passed with it."""
    meta = meta or {}
    if meta.get("addressed"):
        return "answer" if meta.get("pathway") not in ("wonder", "explain", "watch", "reminder") else \
            str(meta.get("pathway"))
    att = meta.get("attention") or {}
    k = att.get("kind") if isinstance(att, dict) else None
    return str(k or meta.get("pathway") or meta.get("source") or "remark")


class RemarkLog:
    def __init__(self, directory: Optional[str], keep_text: bool = True, keep_days: int = 14,
                 clock: Callable[[], float] = time.time):
        self.dir = directory
        self.keep_text = keep_text
        self.keep_days = keep_days
        self.clock = clock
        self._lock = threading.Lock()
        self._pruned = ""

    def _path(self, ts: float) -> Optional[str]:
        if not self.dir:
            return None
        return os.path.join(self.dir, datetime.fromtimestamp(ts).strftime("%Y-%m-%d") + ".jsonl")

    def _write(self, rec: Dict[str, Any]):
        p = self._path(rec["t"])
        if p is None:
            return
        try:
            with self._lock:
                os.makedirs(self.dir, exist_ok=True)
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._prune(rec["t"])
        except Exception:
            logger.exception("Could not write the remark log")

    def _prune(self, now: float):
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if day == self._pruned or not self.dir:
            return
        self._pruned = day
        cutoff = (datetime.fromtimestamp(now) - timedelta(days=self.keep_days)).strftime("%Y-%m-%d")
        try:
            for fn in os.listdir(self.dir):
                if fn.endswith(".jsonl") and fn[:10] < cutoff:
                    os.remove(os.path.join(self.dir, fn))
        except Exception:
            logger.exception("Could not prune the remark log")

    def said(self, text: str, meta: Dict[str, Any], now: Optional[float] = None):
        now = self.clock() if now is None else now
        meta = meta or {}
        rec = {"t": now, "kind": kind_of(meta), "asked": bool(meta.get("addressed")),
               "pathway": str(meta.get("pathway", "")), "id": _id(text)}
        if self.keep_text:
            rec["text"] = (text or "")[:120]
        self._write(rec)

    def reacted(self, text: str, outcome: str, kind: str = "", now: Optional[float] = None):
        now = self.clock() if now is None else now
        self._write({"t": now, "react": outcome, "kind": kind, "id": _id(text)})

    # -- reading back ----------------------------------------------------------
    def load(self, days: int = 1, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = self.clock() if now is None else now
        out: List[Dict[str, Any]] = []
        if not self.dir:
            return out
        for i in range(days):
            p = self._path(now - i * 86400.0)
            if not p or not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                continue
        lo = now - days * 86400.0
        return sorted((r for r in out if r.get("t", 0) >= lo), key=lambda r: r["t"])

    def review(self, days: int = 1, now: Optional[float] = None) -> Dict[str, Any]:
        rows = self.load(days, now)
        said = [r for r in rows if "react" not in r]
        reacts = [r for r in rows if "react" in r]
        by_kind: Dict[str, Counter] = defaultdict(Counter)
        kind_of_id = {}
        for r in said:
            by_kind[r["kind"]]["said"] += 1
            kind_of_id[r["id"]] = r["kind"]
        for r in reacts:
            k = r.get("kind") or kind_of_id.get(r["id"], "remark")
            by_kind[k][r["react"]] += 1
        unasked = [r for r in said if not r.get("asked")]
        per_hour = Counter(datetime.fromtimestamp(r["t"]).strftime("%Y-%m-%d %H") for r in unasked)
        repeats = Counter(r["id"] for r in unasked)
        repeated = [(next((x.get("text", "") for x in unasked if x["id"] == i), ""), n)
                    for i, n in repeats.most_common() if n >= 3]
        night = [r for r in unasked if datetime.fromtimestamp(r["t"]).hour < 6]
        tips: List[str] = []
        for k, c in sorted(by_kind.items()):
            n = c["said"]
            neg = sum(c[o] for o in NEGATIVE)
            ign = sum(c[o] for o in UNANSWERED)
            if n >= 5 and neg / n > 0.5:
                tips.append(f"{k}: waved off {neg} of {n} times -- consider raising its bar or turning it off.")
            elif n >= 5 and ign / n > 0.7:
                tips.append(f"{k}: unanswered {ign} of {n} times -- it may be coming at the wrong moments.")
        busiest = per_hour.most_common(1)[0] if per_hour else None
        if busiest and busiest[1] > 6:
            tips.append(f"{busiest[1]} unprompted remarks in one hour ({busiest[0]}:00) -- too chatty there.")
        for text, n in repeated[:3]:
            tips.append(f"said {n} times: \"{text[:60]}\" -- repetitive.")
        if night:
            tips.append(f"{len(night)} unprompted remark(s) between midnight and 6am -- check quiet hours.")
        return {"said": len(said), "asked": len(said) - len(unasked), "unasked": len(unasked),
                "by_kind": {k: dict(v) for k, v in by_kind.items()},
                "busiest_hour": busiest, "repeated": repeated[:5], "night": len(night), "tips": tips}

    def report(self, days: int = 1, now: Optional[float] = None) -> str:
        r = self.review(days, now)
        lines = [f"Last {days} day(s): ASH said {r['said']} things -- {r['asked']} answers, "
                 f"{r['unasked']} unasked."]
        for k, c in sorted(r["by_kind"].items(), key=lambda kv: -kv[1].get("said", 0)):
            reacts = ", ".join(f"{o} {n}" for o, n in sorted(c.items()) if o != "said")
            lines.append(f"  {k:<14} {c.get('said', 0):>4}" + (f"   ({reacts})" if reacts else ""))
        if r["busiest_hour"]:
            lines.append(f"Busiest hour: {r['busiest_hour'][0]}:00 with {r['busiest_hour'][1]} unasked.")
        lines.append("To tune:" if r["tips"] else "Nothing stands out to tune.")
        lines += [f"  - {t}" for t in r["tips"]]
        return "\n".join(lines)
