"""
src/daemon/doing.py

Things ASH does for you on this computer, by rule, with a yes before anything that moves, writes over or runs -- none of it needs the internet:

    calendar writing   "put dinner with Omar on Friday at 8" -> an event in ASH's own calendar file (a standard .ics the calendar reader already reads), after a
                       check for what is already at that time
    downloads          "tidy my downloads" -> the files sorted into Images / Documents / Archives / Installers / Video / Audio folders (a plan first, a yes to do it,
                       "undo that" puts every one back); "find duplicate downloads" -> same-size, same-content files; nothing is ever deleted
    scripts            "my script back up my project is C:\\tools\\backup.bat" -> "run back up my project" (asks, runs it, says how it ended, keeps a log)
    research write-up  "write up what you have on NPUs" -> a markdown file in Documents/ASH from your notes, terms, screen memory, files and (if you are online) Wikipedia
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ----------------------------------------------------------------------------------------------------------------------------- calendar writing
_EVENT = re.compile(_A + r"(?:put|add|schedule|book|create)\s+(?:an?\s+)?(?:event\s+|meeting\s+)?(?P<title>.+?)\s+(?:(?:on|for)\s+|to\s+my\s+calendar\s+(?:on\s+|for\s+)?)?"
                    r"(?P<day>today|tomorrow|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\s+(?:at\s+)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?"
                    r"(?:\s+(?:for\s+)?(?P<dur>(?:an?\s+hour|half\s+an\s+hour|\d+\s*(?:hours?|hrs?|minutes?|mins?)|one\s+hour|two\s+hours)))?(?:\s+to\s+(?:my\s+)?calendar)?" + _Z, re.I)


def parse_event(text: str, now: float) -> Optional[Dict[str, Any]]:
    m = _EVENT.match((text or "").strip())
    if not m:
        return None
    title = re.sub(r"\s+to\s+my\s+calendar$", "", m.group("title").strip(" ."), flags=re.I)
    title = re.sub(r"^(?:a|an|the)\s+", "", title, flags=re.I)
    if not title or len(title.split()) > 10:
        return None
    today = datetime.fromtimestamp(now).date()
    d = m.group("day").lower().replace("next ", "")
    if d == "today":
        day = today
    elif d == "tomorrow":
        day = today + timedelta(days=1)
    else:
        ahead = (_WEEKDAYS.index(d) - today.weekday()) % 7 or 7
        day = today + timedelta(days=ahead)
    h, mi = int(m.group("h")), int(m.group("m") or 0)
    ap = (m.group("ap") or "").lower().replace(".", "")
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and h < 12 and (re.search(r"\b(?:dinner|supper|tonight|evening|party|movie|film|drinks|date)\b", title, re.I) or 1 <= h <= 7):
        h += 12                                          # "dinner at 8" is the evening; "meeting at 9" is the morning
    if h > 23 or mi > 59:
        return None
    dur = (m.group("dur") or "").lower()
    mins = 60
    if "half" in dur:
        mins = 30
    elif dur:
        n = re.search(r"\d+", dur)
        v = int(n.group()) if n else {"two": 2}.get(dur.split()[0], 1)
        mins = v * (60 if re.search(r"hour|hr", dur) else 1)
    start = datetime(day.year, day.month, day.day, h, mi)
    return {"title": title[:80], "start": start, "end": start + timedelta(minutes=min(mins, 12 * 60))}


def conflicts(existing: List[Any], start: datetime, end: datetime) -> List[Any]:
    return [e for e in existing if not getattr(e, "allday", False) and e.start < end and (e.end or e.start + timedelta(minutes=30)) > start]


def ics_event(title: str, start: datetime, end: datetime) -> str:
    fmt = "%Y%m%dT%H%M%S"
    esc = lambda s: s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;")          # noqa: E731
    return ("BEGIN:VEVENT\r\nUID:" + uuid.uuid4().hex + "@ash\r\nDTSTAMP:" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "\r\nDTSTART:" + start.strftime(fmt)
            + "\r\nDTEND:" + end.strftime(fmt) + "\r\nSUMMARY:" + esc(title) + "\r\nEND:VEVENT\r\n")


def append_event(path: str, title: str, start: datetime, end: datetime) -> Callable[[], None]:
    """Add the event to ASH's own .ics (made if it isn't there); returns the undo that takes exactly that event back out."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    block = ics_event(title, start, end)
    text = open(path, encoding="utf-8", newline="").read() if os.path.exists(path) else "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//ASH//EN\r\nEND:VCALENDAR\r\n"
    i = text.rfind("END:VCALENDAR")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text[:i] + block + text[i:])

    def undo():
        t = open(path, encoding="utf-8", newline="").read()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(t.replace(block, "", 1))
    return undo


# ----------------------------------------------------------------------------------------------------------------------------- downloads
KINDS = {"Images": ".jpg .jpeg .png .gif .bmp .webp .svg .heic .tif .tiff", "Documents": ".pdf .doc .docx .txt .md .rtf .odt .xls .xlsx .csv .ppt .pptx",
         "Archives": ".zip .rar .7z .tar .gz .tgz .bz2", "Installers": ".exe .msi .msix .apk .dmg .iso", "Video": ".mp4 .mkv .avi .mov .webm .wmv",
         "Audio": ".mp3 .wav .flac .m4a .ogg .aac"}
_EXT = {e: k for k, v in KINDS.items() for e in v.split()}
PARTIAL = (".crdownload", ".part", ".partial", ".download", ".tmp", ".opdownload")
_TIDY = re.compile(_A + r"(?:tidy|clean\s+up|sort|organi[sz]e)\s+(?:up\s+)?(?:my\s+)?downloads(?:\s+folder)?" + _Z, re.I)
_DUPES = re.compile(_A + r"(?:find|show|any)\s+(?:me\s+)?(?:the\s+)?duplicate(?:s|d)?\s+(?:downloads|files)(?:\s+in\s+(?:my\s+)?downloads)?" + _Z, re.I)


def asks_tidy(t: str) -> bool:
    return bool(_TIDY.match((t or "").strip()))


def asks_dupes(t: str) -> bool:
    return bool(_DUPES.match((t or "").strip()))


def tidy_plan(folder: str, now: float, min_age_s: float = 86400.0) -> List[Tuple[str, str]]:
    """[(from, to)] for the loose files in `folder` older than a day, by kind. Folders, partial downloads and unknown kinds stay where they are."""
    plan = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for n in names:
        p = os.path.join(folder, n)
        if not os.path.isfile(p) or n.lower().endswith(PARTIAL) or n.startswith("."):
            continue
        kind = _EXT.get(os.path.splitext(n)[1].lower())
        if not kind or now - os.path.getmtime(p) < min_age_s:
            continue
        plan.append((p, os.path.join(folder, kind, n)))
    return plan


def tidy_do(plan: List[Tuple[str, str]]) -> Tuple[int, Callable[[], int]]:
    done: List[Tuple[str, str]] = []
    for src, dst in plan:
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                continue                                       # never over a file that is there
            shutil.move(src, dst)
            done.append((src, dst))
        except OSError:
            continue

    def undo() -> int:
        n = 0
        for src, dst in reversed(done):
            try:
                if os.path.exists(dst) and not os.path.exists(src):
                    shutil.move(dst, src)
                    n += 1
            except OSError:
                pass
        return n
    return len(done), undo


def tidy_say(plan: List[Tuple[str, str]]) -> str:
    if not plan:
        return "Your Downloads are already tidy -- nothing loose older than a day that I know the kind of."
    by: Dict[str, int] = {}
    for _s, d in plan:
        k = os.path.basename(os.path.dirname(d))
        by[k] = by.get(k, 0) + 1
    return "I'd move " + ", ".join(f"{n} to {k}" for k, n in sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))) + ". Nothing is deleted, and \u201cundo that\u201d puts them back. Go ahead?"


def find_dupes(folder: str) -> List[List[str]]:
    """Groups of files with the same size and the same content (hashed only when sizes match)."""
    by_size: Dict[int, List[str]] = {}
    try:
        for n in os.listdir(folder):
            p = os.path.join(folder, n)
            if os.path.isfile(p) and not n.lower().endswith(PARTIAL):
                by_size.setdefault(os.path.getsize(p), []).append(p)
    except OSError:
        return []
    groups = []
    for size, paths in by_size.items():
        if len(paths) < 2 or size == 0:
            continue
        by_hash: Dict[str, List[str]] = {}
        for p in paths:
            h = hashlib.sha1()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            by_hash.setdefault(h.hexdigest(), []).append(p)
        groups += [sorted(g) for g in by_hash.values() if len(g) > 1]
    return groups


def dupes_say(groups: List[List[str]]) -> str:
    if not groups:
        return "No duplicates in your Downloads."
    s = f"{len(groups)} duplicate set{'s' if len(groups) != 1 else ''}: " + "; ".join(" = ".join(os.path.basename(p) for p in g[:3]) for g in groups[:4]) + "."
    return s + " I don't delete anything -- say \u201cmove the duplicates aside\u201d and I'll put the extra copies in a _duplicates folder."


def move_dupes_aside(groups: List[List[str]], folder: str) -> Tuple[int, Callable[[], int]]:
    plan = []
    for g in groups:
        for extra in g[1:]:
            plan.append((extra, os.path.join(folder, "_duplicates", os.path.basename(extra))))
    return tidy_do(plan)


# ----------------------------------------------------------------------------------------------------------------------------- scripts
_REG = re.compile(_A + r"my\s+script\s+(?P<n>[\w' -]{3,40}?)\s+is\s+[\"']?(?P<p>[A-Za-z]:[\\/][^\"']+?\.(?:bat|cmd|ps1|py|sh))[\"']?" + _Z, re.I)
_RUN = re.compile(_A + r"(?:run|start|launch|do)\s+(?:my\s+)?(?:script\s+)?(?P<n>[\w' -]{3,40}?)(?:\s+script)?" + _Z, re.I)
_LAST = re.compile(_A + r"(?:how\s+did|what\s+happened\s+(?:with|to))\s+(?:that|the\s+last)\s+script(?:\s+(?:go|run))?" + _Z, re.I)


class Scripts:
    def __init__(self, path: Optional[str] = None, run: Optional[Callable[[str], Tuple[int, str]]] = None, clock: Callable[[], float] = time.time):
        self.path, self._run, self.clock = path, run or _run_script, clock
        self.items: Dict[str, str] = {}
        self.last: Optional[Dict[str, Any]] = None
        if path:
            try:
                from src.memory.atomic import load_json
                d = load_json(path, {}) or {}
                self.items = dict(d.get("items") or {})
                self.last = d.get("last")
            except Exception:
                pass

    def save(self):
        if self.path:
            try:
                from src.memory.atomic import write_json
                write_json(self.path, {"items": self.items, "last": self.last})
            except Exception:
                pass

    def register(self, text: str) -> Optional[Tuple[str, Callable[[], None]]]:
        m = _REG.match((text or "").strip())
        if not m:
            return None
        name, path = m.group("n").strip().lower(), m.group("p").strip()
        prev = self.items.get(name)
        self.items[name] = path
        self.save()

        def undo():
            if prev is None:
                self.items.pop(name, None)
            else:
                self.items[name] = prev
            self.save()
        return f"Okay -- \u201c{name}\u201d runs {os.path.basename(path)}. Say \u201crun {name}\u201d and I'll ask before I do.", undo

    def which(self, text: str) -> Optional[str]:
        m = _RUN.match((text or "").strip())
        if not m:
            return None
        n = m.group("n").strip().lower()
        return n if n in self.items else None

    def run(self, name: str) -> str:
        path = self.items[name]
        if not os.path.exists(path):
            return f"I can't find {path} -- has it moved?"
        t0 = self.clock()
        code, tail = self._run(path)
        self.last = {"name": name, "code": code, "tail": tail[-200:], "t": t0, "took": self.clock() - t0}
        self.save()
        return (f"{name} finished" + (f" in {self.last['took']:.0f} s" if self.last["took"] >= 2 else "") + ("." if code == 0 else f" with an error (code {code}).")
                + (f" Last line: {tail.strip().splitlines()[-1][:100]}" if tail.strip() else ""))

    def last_say(self, text: str) -> Optional[str]:
        if not _LAST.match((text or "").strip()):
            return None
        if not self.last:
            return "I haven't run a script for you yet."
        l = self.last
        return f"{l['name']}: " + ("it worked." if l["code"] == 0 else f"it ended with code {l['code']}.") + (f" Last line: {l['tail'].strip().splitlines()[-1][:100]}" if l["tail"].strip() else "")


def _run_script(path: str, timeout: float = 600.0) -> Tuple[int, str]:
    ext = os.path.splitext(path)[1].lower()
    cmd = {".ps1": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path], ".py": ["python", path], ".sh": ["sh", path]}.get(ext, ["cmd", "/c", path])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(path) or None, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "it ran longer than ten minutes and was stopped"
    except Exception as e:
        return 1, str(e)


# ----------------------------------------------------------------------------------------------------------------------------- research write-up
_WRITEUP = re.compile(_A + r"(?:write\s+up|compile|put\s+together|make)\s+(?:a\s+)?(?:summary|document|doc|write-?up|report|page)?\s*(?:of\s+|on\s+|about\s+)?(?:what\s+you\s+have\s+(?:on|about)\s+)?(?P<q>[\w .'&+-]{2,40}?)" + _Z, re.I)
_WRITEUP2 = re.compile(_A + r"write\s+up\s+(?:what\s+you\s+(?:have|know)\s+(?:on|about)\s+)?(?P<q>[\w .'&+-]{2,40}?)" + _Z, re.I)


def parse_writeup(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = _WRITEUP2.match(t) or re.match(_A + r"(?:compile|put\s+together)\s+(?:a\s+)?(?:summary|document|report)\s+(?:of|on|about)\s+(?P<q>[\w .'&+-]{2,40}?)" + _Z, t, re.I)
    if not m:
        return None
    q = m.group("q").strip()
    return None if re.fullmatch(r"(?:it|that|this|everything|my\s+day|the\s+day)", q, re.I) else q


def compile_research(topic: str, sections: Dict[str, List[str]], now: float) -> str:
    lines = [f"# {topic}", "", f"_Compiled by ASH on {datetime.fromtimestamp(now).strftime('%A %d %B %Y, %H:%M')} from what is on this computer._", ""]
    for title, items in sections.items():
        if items:
            lines += [f"## {title}", ""] + [f"- {x}" for x in items[:12]] + [""]
    if len(lines) <= 4:
        lines.append("Nothing found yet.")
    return "\n".join(lines).rstrip() + "\n"


def write_doc(folder: str, topic: str, text: str) -> str:
    os.makedirs(folder, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", topic).strip().replace(" ", "-") or "research"
    path = os.path.join(folder, f"research-{base}.md")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"research-{base}-{n}.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path
