"""
src/daemon/lifelog.py

ASH's memory of what you saw, copied, planned and did -- all kept on this computer, nothing needing the internet:

    ScreenMemory   the text the screen reader saw, kept a day: "what was that error I saw an hour ago?" / "where did I read about NPUs?"
    ClipHistory    (opt-in: "keep everything I copy") the last 50 things copied, a day: "what did I copy about the invoice?" / "put the previous one back"
    Timeline       "what did I do last Tuesday?" -- hour by hour, from the window record
    Goals          "my target is to finish chapter 3 by Friday" -- time spent on it since, due-day mentions in the briefing, "target done"
    Recurring      "remind me to call Omar every two weeks"
    Layout         "put Chrome on the left and Word on the right"
    Screenshots    "screenshot this and remember it as the TPU diagram"
Each takes its reader/writer as an argument, so tests never touch the screen, clipboard or windows of this machine.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"
DAY = 86400.0
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


def _words(s: str) -> List[str]:
    stop = {"the", "that", "this", "was", "what", "where", "when", "which", "about", "with", "from", "have", "had", "did", "does", "for", "and", "saw", "read", "see",
            "you", "your", "i", "my", "me", "a", "an", "of", "in", "on", "at", "to", "is", "it", "ago", "earlier", "today", "yesterday", "copy", "copied"}
    return [w for w in _norm(s).split() if len(w) >= 3 and w not in stop]


def _ago(s: str, now: float) -> Optional[float]:
    """'an hour ago' / '20 minutes ago' / 'yesterday' / 'this morning' -> the earliest time that is meant (seconds), or None."""
    t = (s or "").lower()
    m = re.search(r"(\d+|an?|one|two|three|four|five)\s*(hour|hr|minute|min)s?\s+ago", t)
    if m:
        n = m.group(1)
        v = float(n) if n.isdigit() else {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}[n]
        span = v * (3600.0 if m.group(2).startswith("h") else 60.0)
        return now - span * 1.6 - 600.0
    if "this morning" in t:
        return time.mktime(datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timetuple())
    if "yesterday" in t:
        return now - 2 * DAY
    return None


# ----------------------------------------------------------------------------------------------------------------------------- screen memory
class ScreenMemory:
    def __init__(self, path: Optional[str] = None, keep_s: float = DAY, max_items: int = 400, clock: Callable[[], float] = time.time):
        self.path, self.keep_s, self.max, self.clock = path, keep_s, max_items, clock
        self.items: List[Dict[str, Any]] = []
        self._last_key = ""
        if path and os.path.exists(path):
            try:
                import json
                for ln in open(path, encoding="utf-8"):
                    try:
                        self.items.append(json.loads(ln))
                    except ValueError:
                        pass
            except OSError:
                pass
            self._trim()

    def _trim(self):
        cut = self.clock() - self.keep_s
        self.items = [x for x in self.items if x.get("t", 0) >= cut][-self.max:]

    def add(self, text: str, process: str = "", subject: str = "", now: Optional[float] = None) -> bool:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(t) < 40:
            return False
        key = _norm(t)[:120]
        if key == self._last_key:
            return False                                   # the same screen twice
        self._last_key = key
        now = self.clock() if now is None else now
        item = {"t": now, "app": process, "subject": subject[:60], "text": t[:1500]}
        self.items.append(item)
        self._trim()
        if self.path:
            try:
                import json
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(item) + "\n")
            except OSError:
                pass
        return True

    _ASK = re.compile(_A + r"(?:what|where)\s+(?:was|were|is)\s+(?:that|the|those)\s+(?P<q>[\w .'-]{3,50}?)\s+(?:that\s+)?(?:i|we)\s+(?:saw|read|had\s+open|was\s+(?:looking|reading)|were\s+looking)"
                      r"(?:\s+(?:at|on))?(?P<when>\s+(?:an?\s+hour|\d+\s+(?:hours?|minutes?)|one\s+hour|two\s+hours|an?\s+few\s+minutes|a\s+while)\s+ago|\s+yesterday|\s+this\s+morning|\s+earlier(?:\s+today)?)?" + _Z, re.I)
    _ASK2 = re.compile(_A + r"(?:where|when)\s+did\s+i\s+(?:see|read|look\s+at)\s+(?:something\s+about\s+)?(?P<q>[\w .'-]{3,50}?)(?P<when>\s+(?:yesterday|today|this\s+morning|earlier))?" + _Z, re.I)

    def ask(self, text: str, now: Optional[float] = None) -> Optional[str]:
        t = (text or "").strip()
        m = self._ASK.match(t) or self._ASK2.match(t)
        if not m:
            return None
        now = self.clock() if now is None else now
        words = _words(m.group("q"))
        if not words:
            return None
        since = _ago(m.group("when") or "", now)
        best, score = None, 0
        for it in self.items:
            if since is not None and it["t"] < since:
                continue
            low = _norm(it["text"] + " " + it.get("subject", ""))
            hit = sum(1 for w in words if w in low)
            if hit > score or (hit == score and best and it["t"] > best["t"]):
                best, score = it, hit
        if best is None or score < max(1, (len(words) + 1) // 2):
            return f"I don't see anything about {m.group('q').strip()} in what I've read off the screen today."
        low = best["text"]
        pos = min((low.lower().find(w) for w in words if w in low.lower()), default=0)
        snip = low[max(0, pos - 60): pos + 140].strip()
        where = best["app"] or "the screen"
        return f"At {time.strftime('%H:%M', time.localtime(best['t']))} in {where}: “…{snip}…”"

    def recent_text(self, within_s: float = 180.0, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        for it in reversed(self.items):
            if now - it["t"] <= within_s:
                return it["text"]
        return ""


# ----------------------------------------------------------------------------------------------------------------------------- clipboard history
class ClipHistory:
    def __init__(self, read: Callable[[], str], write: Callable[[str], bool], clock: Callable[[], float] = time.time, keep: int = 50):
        self.read, self.write, self.clock, self.keep = read, write, clock, keep
        self.on = False
        self.items: List[Dict[str, Any]] = []

    def poll(self) -> bool:
        if not self.on:
            return False
        try:
            t = str(self.read() or "")
        except Exception:
            return False
        if len(t.strip()) < 4 or (self.items and self.items[-1]["text"] == t):
            return False
        from .clipboard import looks_secret
        if looks_secret(t):
            return False
        self.items.append({"t": self.clock(), "text": t[:4000]})
        cut = self.clock() - DAY
        self.items = [x for x in self.items if x["t"] >= cut][-self.keep:]
        return True

    _ON = re.compile(_A + r"(?:keep|remember|save|record)\s+(?:everything|all)\s+(?:that\s+)?i\s+copy(?:\s+(?:for|over)\s+(?:the\s+)?(?:next\s+)?(?:hour|day|while))?" + _Z, re.I)
    _OFF = re.compile(_A + r"(?:stop\s+(?:keeping|remembering|saving|recording)\s+(?:what\s+i\s+copy|my\s+clipboard|everything\s+i\s+copy)|forget\s+(?:what\s+i\s+copied|my\s+clipboard\s+history))" + _Z, re.I)
    _FIND = re.compile(_A + r"what\s+(?:did\s+i|have\s+i)\s+copy(?:\s+(?:about|with|that\s+(?:had|has|mentioned)|from)\s+(?P<q>[\w .'-]{2,40}?))?(?P<when>\s+(?:earlier|today|an?\s+hour\s+ago|yesterday))?" + _Z, re.I)
    _BACK = re.compile(_A + r"(?:put|bring|copy)\s+(?:back\s+)?(?:the\s+)?(?:previous|last|one\s+before|earlier)\s+(?:thing\s+)?(?:i\s+copied|one|clipboard|copy)(?:\s+back)?" + _Z, re.I)

    def hear(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if self._ON.match(t):
            self.on = True
            return "Okay -- I'll keep the last fifty things you copy, for a day. Say “stop keeping what I copy” to end it."
        if self._OFF.match(t):
            n = len(self.items)
            self.on, self.items = False, []
            return f"Okay -- stopped, and forgot {n} copied item{'s' if n != 1 else ''}."
        if self._BACK.match(t):
            if not self.on:
                return "I'm not keeping your clipboard. Say “keep everything I copy” first."
            if len(self.items) < 2:
                return "There's no earlier copy yet."
            prev = self.items[-2]["text"]
            return ("Put it back on your clipboard: " + prev[:60].replace("\n", " ") + ("…" if len(prev) > 60 else "")) if self.write(prev) else "I couldn't write to the clipboard."
        m = self._FIND.match(t)
        if m:
            if not self.on:
                return "I'm not keeping your clipboard. Say “keep everything I copy” and I will, from then on."
            words = _words(m.group("q") or "")
            since = _ago(m.group("when") or "", self.clock())
            hits = [x for x in reversed(self.items) if (since is None or x["t"] >= since) and (not words or all(w in _norm(x["text"]) for w in words))]
            if not hits:
                return "Nothing like that in what you've copied."
            h = hits[0]
            return f"At {time.strftime('%H:%M', time.localtime(h['t']))} you copied: “{h['text'][:140].strip()}”" + (f" -- and {len(hits) - 1} more like it." if len(hits) > 1 else "")
        return None


# ----------------------------------------------------------------------------------------------------------------------------- timeline
_TL = re.compile(_A + r"(?:show\s+me|what\s+(?:did\s+i|was\s+i)\s+(?:do|doing|working\s+on)|(?:give\s+me\s+)?(?:a\s+)?timeline\s+(?:for|of))\s+(?:on\s+)?(?P<d>yesterday|today|last\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day|(?:mon|tues|wednes|thurs|fri|satur|sun)day|the\s+day\s+before\s+yesterday)"
                 r"(?:'s\s+timeline)?" + _Z, re.I)


def timeline_day(text: str, now: float) -> Optional[str]:
    m = _TL.match((text or "").strip())
    if not m:
        return None
    d = m.group("d").lower().replace("last ", "")
    today = datetime.fromtimestamp(now).date()
    if d == "today":
        return today.isoformat()
    if d == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    if d.startswith("the day before"):
        return (today - timedelta(days=2)).isoformat()
    if d in _WEEKDAYS:
        back = (today.weekday() - _WEEKDAYS.index(d)) % 7 or 7
        return (today - timedelta(days=back)).isoformat()
    return None


def timeline(rows: List[Dict[str, Any]], day: str, friendly: Callable[[str], str] = lambda p: p) -> str:
    """Hour by hour from the active_window events of one day: the program and what was open, for each hour there was anything."""
    by_hour: Dict[int, Dict[Tuple[str, str], int]] = {}
    for r in rows:
        if r.get("source") != "active_window":
            continue
        meta = r.get("meta") or {}
        if meta.get("blocked") or meta.get("private") or not meta.get("process"):
            continue
        h = datetime.fromtimestamp(float(r.get("ts") or 0)).hour
        key = (friendly(str(meta.get("process"))), str(meta.get("project") or meta.get("subject") or "")[:30])
        by_hour.setdefault(h, {})
        by_hour[h][key] = by_hour[h].get(key, 0) + 1
    if not by_hour:
        return "I have nothing on that day."
    parts = []
    for h in sorted(by_hour):
        app, what = max(by_hour[h].items(), key=lambda kv: kv[1])[0]
        parts.append(f"{h:02d}h {app}" + (f" ({what})" if what and what.lower() != app.lower() else ""))
    label = datetime.fromisoformat(day).strftime("%A %d %B")
    return f"{label}: " + "; ".join(parts[:14]) + "."


# ----------------------------------------------------------------------------------------------------------------------------- goals
def parse_due(s: str, now: float) -> Optional[float]:
    t = (s or "").strip().lower()
    today = datetime.fromtimestamp(now).date()

    def eod(d):
        return time.mktime(datetime(d.year, d.month, d.day, 23, 59).timetuple())
    if t in ("today", "tonight", "end of the day"):
        return eod(today)
    if t == "tomorrow":
        return eod(today + timedelta(days=1))
    if t in ("next week", "end of the week", "the end of the week"):
        return eod(today + timedelta(days=7 if t == "next week" else (4 - today.weekday()) % 7 or 7))
    m = re.fullmatch(r"(?:in\s+)?(\d+|a|one|two|three|four|five|six|seven|ten)\s+(day|week)s?", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}[m.group(1)]
        return eod(today + timedelta(days=n * (7 if m.group(2) == "week" else 1)))
    t2 = t.replace("next ", "").replace("this ", "")
    if t2 in _WEEKDAYS:
        ahead = (_WEEKDAYS.index(t2) - today.weekday()) % 7 or 7
        return eod(today + timedelta(days=ahead))
    return None


_SET = re.compile(_A + r"(?:my\s+(?:target|aim|deadline)\s+is\s+(?:to\s+)?|(?:set|add)\s+(?:a\s+)?(?:target|deadline)\s*(?:to|:)?\s*|i\s+(?:need|have)\s+to\s+(?=finish|submit|write|complete|prepare|study)(?=.*\bby\b))"
                  r"(?P<what>.+?)(?:\s+by\s+(?P<due>[\w ]{3,20}?))?" + _Z, re.I)
_DONE = re.compile(_A + r"(?:(?:my\s+)?target\s+(?:is\s+)?(?:done|finished|complete)|i\s+(?:finished|completed|did)\s+(?:my\s+)?target(?:\s+(?P<what>.+?))?|mark\s+(?:my\s+)?target\s+(?:as\s+)?done)" + _Z, re.I)
_LIST = re.compile(_A + r"(?:what\s+(?:targets|deadlines)\s+(?:have\s+i\s+set|do\s+i\s+have)|list\s+my\s+(?:targets|deadlines)|how(?:'s|\s+is|\s+are)\s+my\s+(?:targets?|deadlines?)(?:\s+going)?)" + _Z, re.I)


class Goals:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.items: List[Dict[str, Any]] = []
        if path:
            try:
                from src.memory.atomic import load_json
                self.items = list((load_json(path, {}) or {}).get("items") or [])
            except Exception:
                self.items = []

    def save(self):
        if self.path:
            try:
                from src.memory.atomic import write_json
                write_json(self.path, {"items": self.items})
            except Exception:
                pass

    def hear(self, text: str, spent: Callable[[str, float], float]) -> Optional[Tuple[str, Optional[Callable[[], None]]]]:
        t = (text or "").strip()
        now = self.clock()
        if _LIST.match(t):
            live = [g for g in self.items if not g.get("done")]
            if not live:
                return "You haven't set any targets -- say “my target is to finish chapter 3 by Friday”.", None
            return "; ".join(self._line(g, spent, now) for g in live[:5]) + ".", None
        m = _DONE.match(t)
        if m:
            live = [g for g in self.items if not g.get("done")]
            pick = next((g for g in live if m.group("what") and _norm(m.group("what")) in _norm(g["text"])), live[-1] if live else None)
            if pick is None:
                return "You don't have an open target.", None
            pick["done"] = True
            self.save()

            def undo(p=pick):
                p["done"] = False
                self.save()
            return f"Nice -- “{pick['text']}” is done.", undo
        m = _SET.match(t)
        if m and not re.search(r"\b(?:browsing|social|video|youtube)\b", t, re.I):
            what = m.group("what").strip(" .")
            due = parse_due(m.group("due") or "", now) if m.group("due") else None
            if m.group("due") and due is None:
                return f"I didn't get the date -- say it like “by Friday”, “by tomorrow” or “in 3 days”.", None
            g = {"text": what, "due": due, "created": now, "done": False}
            self.items.append(g)
            self.save()

            def undo(g=g):
                if g in self.items:
                    self.items.remove(g)
                self.save()
            return (f"Got it -- “{what}”" + (f", due {datetime.fromtimestamp(due).strftime('%A')}" if due else "") + ". I'll count the time you spend on it."), undo
        return None

    def _line(self, g: Dict[str, Any], spent: Callable[[str, float], float], now: float) -> str:
        s = spent(g["text"], g["created"])
        left = ""
        if g.get("due"):
            d = (datetime.fromtimestamp(g["due"]).date() - datetime.fromtimestamp(now).date()).days
            left = "due today" if d == 0 else "overdue" if d < 0 else "due tomorrow" if d == 1 else f"due in {d} days"
        hrs = f"{s / 3600:.1f} h so far" if s >= 600 else "nothing tracked yet"
        return f"“{g['text']}”: {hrs}" + (f", {left}" if left else "")

    def briefing_part(self, kind: str, now: float, spent: Callable[[str, float], float]) -> Optional[str]:
        if kind != "morning":
            return None
        soon = [g for g in self.items if not g.get("done") and g.get("due") and 0 <= g["due"] - now <= 3 * DAY or (not g.get("done") and g.get("due") and g["due"] < now)]
        if not soon:
            return None
        return "Targets: " + "; ".join(self._line(g, spent, now) for g in soon[:3]) + "."


# ----------------------------------------------------------------------------------------------------------------------------- recurring
_EVERY = re.compile(_A + r"remind\s+me\s+to\s+(?P<what>.+?)\s+every\s+(?P<n>\d+|other|two|three|four)?\s*(?P<u>day|week|month|days|weeks|months)s?" + _Z, re.I)


class Recurring:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.items: List[Dict[str, Any]] = []
        if path:
            try:
                from src.memory.atomic import load_json
                self.items = list((load_json(path, {}) or {}).get("items") or [])
            except Exception:
                self.items = []

    def save(self):
        if self.path:
            try:
                from src.memory.atomic import write_json
                write_json(self.path, {"items": self.items})
            except Exception:
                pass

    def hear(self, text: str) -> Optional[Tuple[str, Callable[[], None]]]:
        m = _EVERY.match((text or "").strip())
        if not m:
            return None
        n = m.group("n")
        k = {"other": 2, "two": 2, "three": 3, "four": 4}.get((n or "").lower(), int(n) if (n or "").isdigit() else 1)
        unit = {"day": 1, "week": 7, "month": 30}[m.group("u").lower().rstrip("s")]
        item = {"what": m.group("what").strip(" ."), "every_days": k * unit, "next": self.clock() + k * unit * DAY}
        self.items.append(item)
        self.save()

        def undo(i=item):
            if i in self.items:
                self.items.remove(i)
            self.save()
        return f"Okay -- I'll remind you to {item['what']} every {k * unit} days, starting {datetime.fromtimestamp(item['next']).strftime('%A')}.", undo

    def due(self) -> List[str]:
        now, out = self.clock(), []
        for i in self.items:
            if now >= i["next"]:
                out.append(f"Time to {i['what']}.")
                i["next"] = now + i["every_days"] * DAY
        if out:
            self.save()
        return out


# ----------------------------------------------------------------------------------------------------------------------------- window layout
_LAYOUT = re.compile(_A + r"(?:put|snap|move|place|show)\s+(?P<a>[\w .+-]{2,25}?)\s+(?:on|to)\s+the\s+(?P<sa>left|right)(?:\s+(?:and|then)\s+(?P<b>[\w .+-]{2,25}?)\s+(?:on|to)\s+the\s+(?P<sb>left|right))?(?:\s+(?:of\s+the\s+screen|side))?" + _Z, re.I)


def parse_layout(text: str) -> Optional[List[Tuple[str, str]]]:
    m = _LAYOUT.match((text or "").strip())
    if not m:
        return None
    out = [(m.group("a").strip(), m.group("sa").lower())]
    if m.group("b"):
        out.append((m.group("b").strip(), m.group("sb").lower()))
    return out


def layout_rects(side: str, width: int, height: int) -> Tuple[int, int, int, int]:
    half = width // 2
    return (0, 0, half, height) if side == "left" else (half, 0, width - half, height)


class Windows:
    """Moving windows: `find(process_name)` -> handle or None; `move(handle, x, y, w, h)`; `size()` -> (w, h). Real on Windows through user32."""

    def __init__(self, find=None, move=None, size=None):
        self.find, self.move, self.size = find or _find_window, move or _move_window, size or _screen_size


def _screen_size() -> Tuple[int, int]:
    try:
        import ctypes
        u = ctypes.windll.user32
        r = ctypes.wintypes.RECT() if hasattr(ctypes, "wintypes") else None
        SPI_GETWORKAREA = 0x30
        from ctypes import wintypes
        rect = wintypes.RECT()
        u.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
        return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        return 1920, 1040


def _find_window(proc: str):
    try:
        import ctypes
        from ctypes import wintypes
        import psutil
        u = ctypes.windll.user32
        want = re.sub(r"\.exe$", "", proc.lower().strip())
        found = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, _l):
            if not u.IsWindowVisible(hwnd) or not u.GetWindowTextLengthW(hwnd):
                return True
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                name = psutil.Process(pid.value).name().lower().replace(".exe", "")
            except Exception:
                return True
            if want in name or name in want:
                found.append(hwnd)
                return False
            return True
        u.EnumWindows(EnumProc(cb), 0)
        return found[0] if found else None
    except Exception:
        return None


def _move_window(hwnd, x, y, w, h) -> bool:
    try:
        import ctypes
        u = ctypes.windll.user32
        u.ShowWindow(hwnd, 9)                                     # restore, if maximised
        return bool(u.MoveWindow(hwnd, x, y, w, h, True))
    except Exception:
        return False


def do_layout(win: Windows, plan: List[Tuple[str, str]], aliases: Dict[str, str]) -> str:
    w, h = win.size()
    done, missing = [], []
    for name, side in plan:
        proc = aliases.get(name.lower(), name)
        hwnd = win.find(proc)
        if hwnd is None:
            missing.append(name)
            continue
        x, y, ww, hh = layout_rects(side, w, h)
        if win.move(hwnd, x, y, ww, hh):
            done.append(f"{name} on the {side}")
    if not done:
        return "I couldn't find " + " or ".join(missing) + " open." if missing else "I couldn't move the windows."
    return "Done: " + ", ".join(done) + "." + (f" I couldn't find {', '.join(missing)} open." if missing else "")


# ----------------------------------------------------------------------------------------------------------------------------- screenshots
_SHOT = re.compile(_A + r"(?:take\s+a\s+)?screenshot(?:\s+(?:of\s+)?(?:this|the\s+screen))?(?:\s+and\s+(?:remember|save|keep|call)\s+(?:it\s+)?(?:as|called)\s+(?P<n>[\w .'-]{2,40}?))?" + _Z, re.I)


def parse_shot(text: str) -> Optional[str]:
    """'' for a plain screenshot, the name for 'screenshot this and remember it as the TPU diagram', None when it is not one."""
    m = _SHOT.match((text or "").strip())
    if not m:
        return None
    return (m.group("n") or "").strip()


def take_shot(name: str, folder: str, grab: Optional[Callable[[], Any]] = None, now: Optional[float] = None) -> Tuple[str, str]:
    """(say, path). The picture goes to the folder named by the time (or by `name`); never over an existing file."""
    now = time.time() if now is None else now
    os.makedirs(folder, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().replace(" ", "_") if name else time.strftime("screen-%Y%m%d-%H%M%S", time.localtime(now))
    path = os.path.join(folder, base + ".png")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"{base}-{n}.png")
    try:
        if grab is None:
            from PIL import ImageGrab
            grab = ImageGrab.grab
        grab().save(path)
    except Exception:
        return "I couldn't take the screenshot.", ""
    return (f"Saved {os.path.basename(path)} in {os.path.basename(folder)}." + (f" I'll remember it as “{name}”." if name else "")), path
