"""
src/daemon/digest.py

The day's failures as a file you can hand over, not an afternoon to reconstruct.

25 Nov: working out what had gone wrong took a whole afternoon of reading logs. ASH already keeps the pieces -- state/failures.jsonl (repair mode: the lines that
showed frustration and what had been said before them; a fact it held that was wrong and was corrected) and state/unrouted.jsonl (every line that reached the
language model and why, and how long it took). This joins them for one day, at the nightly sleep, into state/digest/digest-YYYY-MM-DD.md:

    the failures (when, what was heard, what was said before, what was done)
    the lines that went to the model, grouped by why and by how long it took (the slow ones named)
    the lines that came up more than once (the next rules to write)

and "what went wrong today" answers from the same numbers. Local files only; nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_ASK = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|and|ok(?:ay)?)[,\s]+)?(?:what\s+(?:went\s+wrong|has\s+gone\s+wrong|did\s+you\s+get\s+wrong|did\s+you\s+fail\s+at)|"
                  r"how\s+many\s+(?:times\s+)?(?:did\s+you\s+(?:fail|get\s+(?:it|things)\s+wrong)|(?:things\s+)?went\s+wrong)|show\s+me\s+(?:the\s+)?(?:failures|your\s+failures|what\s+went\s+wrong)|"
                  r"(?:what\s+are|list)\s+(?:the\s+|your\s+)?(?:failures|mistakes))(?:\s+(?:today|yesterday|this\s+week))?\s*[.!?]*\s*$", re.I)


def asks_digest(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= 70 and bool(_ASK.match(t))


def _rows(path: Optional[str], lo: float, hi: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if lo <= float(d.get("t") or 0.0) < hi:
                    out.append(d)
    except OSError:
        pass
    return out


def _key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))[:80]


def summary(failures_path: Optional[str], unrouted_path: Optional[str], now: Optional[float] = None, hours: float = 24.0) -> Dict[str, Any]:
    now = time.time() if now is None else now
    lo = now - hours * 3600.0
    fails = _rows(failures_path, lo, now + 1)
    un = _rows(unrouted_path, lo, now + 1)
    model = [r for r in un if not r.get("tool")]
    by_why = Counter(str(r.get("why") or "?") for r in model)
    slow = sorted((r for r in model if float(r.get("took_s") or 0.0) >= 15.0), key=lambda r: -float(r.get("took_s") or 0.0))
    again: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in model:
        again[_key(str(r.get("text") or ""))].append(r)
    repeated = sorted(((k, v) for k, v in again.items() if k and len(v) >= 2), key=lambda kv: -len(kv[1]))
    return {"failures": fails, "model_lines": model, "by_why": by_why, "slow": slow, "repeated": repeated, "since": lo}


def render(s: Dict[str, Any], day: str) -> str:
    lines = [f"# ASH digest -- {day}", "",
             f"{len(s['failures'])} failures kept, {len(s['model_lines'])} lines went to the language model "
             f"({len(s['slow'])} of them took 15 s or more).", ""]
    if s["failures"]:
        lines += ["## Failures", ""]
        for f in s["failures"]:
            if f.get("what"):
                lines.append(f"- {f.get('when', '')} -- {f['what']}: heard “{f.get('heard', '')}”, said “{f.get('said', '')}”")
            else:
                lines.append(f"- {f.get('when', '')} -- repair mode started. Heard: " + " / ".join(f"“{x}”" for x in (f.get("lines") or [])[-4:])
                             + ". Before that: " + " / ".join(f"“{x}”" for x in (f.get("said") or [])[-4:]))
        lines.append("")
    if s["by_why"]:
        lines += ["## Why lines reached the model", ""] + [f"- {n} x {w}" for w, n in s["by_why"].most_common(8)] + [""]
    if s["slow"]:
        lines += ["## Slow (15 s or more)", ""] + [f"- {float(r.get('took_s') or 0):.0f} s -- “{str(r.get('text') or '')[:90]}”" + (f" -- {r['note']}" if r.get("note") else "") for r in s["slow"][:10]] + [""]
    if s["repeated"]:
        lines += ["## Said more than once and still sent to the model (the next rules)", ""]
        lines += [f"- {len(v)} x “{str(v[0].get('text') or '')[:90]}”" for _, v in s["repeated"][:12]] + [""]
    if len(lines) <= 4:
        lines.append("Nothing went wrong that was kept.")
    return "\n".join(lines).rstrip() + "\n"


def write(out_dir: str, failures_path: Optional[str], unrouted_path: Optional[str], now: Optional[float] = None, extra: Optional[List[str]] = None) -> Optional[str]:
    """state/digest/digest-YYYY-MM-DD.md for the last 24 hours; the path, or None if it could not be written."""
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"digest-{day}.md")
        with open(path, "w", encoding="utf-8") as fh:
            text = render(summary(failures_path, unrouted_path, now), day)
            if extra:
                text = text.replace("Nothing went wrong that was kept.\n", "").rstrip() + "\n\n## What you taught me\n\n" + "\n".join(f"- {x}" for x in extra) + "\n"
            fh.write(text)
        return path
    except OSError:
        return None


def say(failures_path: Optional[str], unrouted_path: Optional[str], now: Optional[float] = None) -> str:
    s = summary(failures_path, unrouted_path, now)
    nf, nm, ns = len(s["failures"]), len(s["model_lines"]), len(s["slow"])
    if not nf and not nm:
        return "Nothing went wrong that I kept in the last day."
    bits = []
    if nf:
        bits.append(f"{nf} time{'s' if nf != 1 else ''} you were frustrated or I got a fact wrong")
    if nm:
        bits.append(f"{nm} lines I had to ask the language model about" + (f", {ns} of them slow" if ns else ""))
    top = s["repeated"][0][1][0].get("text") if s["repeated"] else ""
    return "In the last day: " + "; ".join(bits) + "." + (f" The one that kept coming back: “{str(top)[:80]}”." if top else "") + " The full list is in the digest file."
