"""
src/daemon/recall.py

Three JARVIS moves, from what ASH already keeps on this computer -- by rule, never the model:

    "Pull up everything on NoSQL" / "what do you have on Omar"   one answer from every place ASH keeps something: who it is, what it means, the
                                                                  time on it this week, the notes, the timetable, reminders, the files about it.
    "Find that page about liquid biopsy from yesterday"            the window titles of the last week (the journal), newest match first; still open
                                                                  in the browser, it is brought forward on "open it".
    "Tell me after 30 minutes of YouTube" / "stop me at midnight"   a limit the person sets, counted from now, said once when it is reached.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.recall")

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _span(secs: float) -> str:
    m = int(round(secs / 60.0))
    if m < 60:
        return f"{max(1, m)} minute{'s' if m != 1 else ''}"
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {m} min" if m else "")


# ---------------------------------------------------------------------------------------------------------------------------------- dossier
_DOSSIER = re.compile(_A + r"(?:(?:pull\s+up|show\s+me|give\s+me|tell\s+me|bring\s+up)\s+everything\s+(?:on|about|you\s+(?:have|know|got)\s+(?:on|about))\s+"
                      r"|everything\s+you\s+(?:have|know|got)\s+(?:on|about)\s+|what\s+(?:do\s+you\s+have|have\s+you\s+got)\s+on\s+"
                      r"|(?:open|pull\s+up|give\s+me)\s+(?:a\s+|the\s+)?(?:dossier|file)\s+on\s+)(?P<s>[\w.'&+ -]{2,40}?)" + _Z, re.I)
_NOT_SUBJECT = re.compile(r"^(?:me|myself|my\s+browsing|my\s+(?:tabs|habits|data)|you|yourself|it|that|this)$", re.I)


def dossier_subject(text: str) -> Optional[str]:
    m = _DOSSIER.match((text or "").strip())
    if not m:
        return None
    s = re.sub(r"^(?:the|my)\s+", "", m.group("s").strip(), flags=re.I)
    return None if not s or _NOT_SUBJECT.match(s) else s


def dossier(rt: Any, subject: str, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    q, qn = subject.strip(), _norm(subject)
    bits: List[str] = []
    kb = getattr(getattr(rt, "wonder", None), "kb", None)
    shown = q
    try:
        if kb is not None:
            p = kb.people.get(qn)
            if p:
                shown = p.get("name") or q
                rel = p.get("relation") or p.get("desc")
                if rel:
                    bits.append(f"{shown} is your {rel}" if p.get("relation") else f"{shown} is {rel}")
            t = kb.terms.get(qn)
            if t and t.get("definition"):
                from .wondering import _yours
                bits.append(f"you told me {t.get('term') or q} is {_yours(str(t['definition']))[:80]}")
    except Exception:
        logger.debug("dossier: people/terms failed", exc_info=True)
    try:
        from .working import time_on
        week, hit = time_on(getattr(rt, "activity", None), q, 7, now)
        today, _h = time_on(getattr(rt, "activity", None), q, 1, now)
        if week >= 60:
            bits.append(f"{_span(week)} on it this week" + (f", {_span(today)} of that today" if today >= 60 else ""))
            a = getattr(rt, "activity", None)
            last = max((t.last for t in a.topics.values() if t.name in hit), default=0.0) if a is not None else 0.0
            if last and now - last > 3600:
                bits.append("last on it " + _when(last, now))
    except Exception:
        logger.debug("dossier: time failed", exc_info=True)
    try:
        notes = [n for n in list(getattr(getattr(rt, "notes", None), "items", None) or []) if qn and qn in _norm(n.said)]
        if notes:
            last = max(notes, key=lambda n: n.at)
            bits.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''} -- the latest: {last.back}" if len(notes) > 1 else f"a note: {last.back}")
    except Exception:
        logger.debug("dossier: notes failed", exc_info=True)
    try:
        tt = getattr(rt, "timetable", None)
        for c in (getattr(tt, "classes", None) or []):
            if qn and (qn in _norm(c.get("name")) or _norm(c.get("name")) in qn):
                bits.append("on your timetable: " + tt._describe(c))
                break
    except Exception:
        logger.debug("dossier: timetable failed", exc_info=True)
    try:
        r = getattr(rt, "reminders", None)
        if r is not None:
            with r._lock:
                due = [x for x in r.items.values() if x.status == "set" and qn and qn in _norm(x.text)]
            if due:
                x = min(due, key=lambda x: x.due)
                bits.append(f"a reminder {_when(x.due, now, future=True)}: {x.text}")
    except Exception:
        logger.debug("dossier: reminders failed", exc_info=True)
    try:
        fs = getattr(rt, "filesearch", None)
        if fs is not None:
            hits = fs.search(q, now, n=3)
            names = [h.entry.name if hasattr(h.entry, "name") else str(getattr(h.entry, "path", ""))[-40:] for h in hits]
            names = [n for n in names if n]
            if names:
                bits.append("files: " + ", ".join(names[:3]))
    except Exception:
        logger.debug("dossier: files failed", exc_info=True)
    if not bits:
        return f"I don't have anything on {q} yet -- not in my notes, my memory of your time, your timetable or your files."
    s = f"On {shown}: " + "; ".join(bits) + "."
    return s


def _when(t: float, now: float, future: bool = False) -> str:
    d0, d1 = datetime.fromtimestamp(now).date(), datetime.fromtimestamp(t).date()
    hm = datetime.fromtimestamp(t).strftime("%H:%M")
    if d0 == d1:
        return f"today at {hm}"
    diff = (d1 - d0).days
    if diff == -1:
        return f"yesterday at {hm}"
    if diff == 1:
        return f"tomorrow at {hm}"
    return datetime.fromtimestamp(t).strftime("%A") + (" at " + hm if future else "")


# -------------------------------------------------------------------------------------------------------------------------------- page recall
_PAGE = re.compile(_A + r"(?:find|open|show\s+me|bring\s+back|get\s+me|where(?:'s|\s+is|\s+was)|what\s+was)\s+(?:that|the)\s+"
                   r"(?:page|site|tab|article|video|website|thing|chat)\s+(?:about|on|with|for)\s+(?P<q>[\w.'&+ -]{3,50}?)"
                   r"(?:\s+(?:that\s+|which\s+)?i\s+(?:had|was\s+(?:on|reading|watching)|looked\s+at|saw|opened)(?:\s+open)?)?"
                   r"(?:\s+(?:from\s+|on\s+)?(?P<when>yesterday|today|earlier|this\s+morning|last\s+night|this\s+week|the\s+other\s+day))?" + _Z, re.I)


def page_query(text: str) -> Optional[Tuple[str, str]]:
    m = _PAGE.match((text or "").strip())
    if not m:
        return None
    q = re.sub(r"\s+(?:i\s+had|i\s+was\s+on)$", "", m.group("q").strip(), flags=re.I)
    return q, (m.group("when") or "").lower()


def find_page(journal: Any, q: str, when: str, now: float, days: int = 7) -> Optional[Dict[str, Any]]:
    """The newest window title holding every word of `q` (in the journal's last `days` days; 'yesterday' looks at yesterday only)."""
    words = [w for w in _norm(q).split() if len(w) >= 3]
    if not words or journal is None:
        return None
    today = datetime.fromtimestamp(now).date()
    for back in range(days):
        day = today.fromordinal(today.toordinal() - back)
        if when in ("yesterday", "last night") and back != 1:
            continue
        if when in ("today", "earlier", "this morning") and back != 0:
            continue
        try:
            rows = journal.read_day(day.strftime("%Y-%m-%d"))
        except Exception:
            rows = []
        best = None
        for r in rows:
            meta = r.get("meta") or {}
            subj = str(meta.get("subject") or "")
            if not subj or meta.get("blocked") or meta.get("private") or meta.get("withheld"):
                continue
            n = _norm(subj + " " + str(meta.get("project") or ""))
            if all(w in n for w in words):
                t = float(r.get("ts") or r.get("t") or 0.0)
                if best is None or t >= best["t"]:
                    best = {"t": t, "title": subj, "site": str(meta.get("project") or ""), "process": str(meta.get("process") or "")}
        if best is not None:
            return best
    return None


# ---------------------------------------------------------------------------------------------------------------------------------- limits
_LIMIT = re.compile(_A + r"(?:tell|warn|remind|stop|nudge|ping)\s+me\s+(?:when\s+i'?ve\s+(?:spent|had|been\s+on)\s+|after\s+|if\s+i\s+spend\s+more\s+than\s+|"
                    r"once\s+i'?ve\s+(?:spent|had)\s+)(?P<n>\d+|an?|one|two|three|half\s+an|fifteen|twenty|thirty|forty|forty-five|ninety)\s*"
                    r"(?P<u>minutes?|mins?|hours?|hrs?)\s+(?:of|on|in)\s+(?P<what>[\w.' -]{2,30}?)(?:\s+today)?" + _Z, re.I)
_LIMIT_OFF = re.compile(_A + r"(?:cancel|clear|drop|forget|remove|stop)\s+(?:my\s+|the\s+|all\s+(?:my\s+)?)?(?:time\s+)?limits?(?:\s+(?:on|for)\s+(?P<what>[\w.' -]{2,30}?))?" + _Z, re.I)
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "half an": 0.5, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "ninety": 90}


def parse_limit(text: str) -> Optional[Tuple[str, float]]:
    m = _LIMIT.match((text or "").strip())
    if not m:
        return None
    n = m.group("n").lower()
    v = float(n) if n.isdigit() else float(_NUMW.get(re.sub(r"\s+", " ", n), 1))
    secs = v * (3600.0 if m.group("u").lower().startswith("h") else 60.0)
    what = re.sub(r"^(?:the|my)\s+", "", m.group("what").strip(), flags=re.I)
    return (what, secs) if 60.0 <= secs <= 12 * 3600 else None


def parse_limit_off(text: str) -> Optional[str]:
    m = _LIMIT_OFF.match((text or "").strip())
    return None if not m else (m.group("what") or "").strip()


def seconds_on(rt: Any, what: str, now: float) -> float:
    """Seconds today on `what`: the browser tab record for a site ('youtube' -> youtube.com), else the time on matching activity topics."""
    q = _norm(what).replace(" ", "")
    total = 0.0
    try:
        bt = rt.periphery.sensors.get("browser_tabs") if getattr(rt, "periphery", None) is not None else None
        led = getattr(bt, "ledger", None)
        if led is not None:
            day0 = time.mktime(datetime.fromtimestamp(now).date().timetuple())
            for site, sec, _n in led.totals(day0, now + 1, by="site"):
                if q and q in _norm(site).replace(" ", ""):
                    total += sec
    except Exception:
        logger.debug("limits: ledger failed", exc_info=True)
    if total <= 0:
        try:
            from .working import time_on
            total, _h = time_on(getattr(rt, "activity", None), what, 1, now)
        except Exception:
            total = 0.0
    return float(total)


class Limits:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.items: List[Dict[str, Any]] = []

    def add(self, rt: Any, what: str, secs: float, now: float) -> str:
        self.items = [x for x in self.items if _norm(x["what"]) != _norm(what)]
        self.items.append({"what": what, "limit": secs, "base": seconds_on(rt, what, now), "set": now, "told": False})
        return f"Okay -- I'll tell you after {_span(secs)} of {what}, counting from now."

    def clear(self, what: str = "") -> str:
        n = len(self.items)
        if what:
            self.items = [x for x in self.items if _norm(x["what"]) != _norm(what)]
            return f"Okay -- no limit on {what}." if len(self.items) < n else f"There wasn't a limit on {what}."
        self.items = []
        return "Okay -- limits cleared." if n else "There weren't any limits set."

    def due(self, rt: Any, now: float) -> List[str]:
        """Lines for the limits reached since the last look; each said once, then dropped. A limit left for a day goes."""
        out = []
        keep = []
        for x in self.items:
            if now - x["set"] > 86400.0:
                continue
            spent = seconds_on(rt, x["what"], now) - x["base"]
            if spent >= x["limit"]:
                out.append(f"That's {_span(x['limit'])} of {x['what']} -- the limit you asked me to watch.")
                continue
            keep.append(x)
        self.items = keep
        return out
