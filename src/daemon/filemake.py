"""
src/daemon/filemake.py

"Create a new empty desktop text file called notes.txt" -- done, by rule, with the file tool. There was no route for it: the request
went to the model, which said "Yes, I created a new empty text file on your desktop" with nothing created, and, asked where it was,
"in the current active window".

Only new, empty files, only in the user's Desktop, Documents or Downloads, and never over a file that is already there.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

_PLACES = {"desktop": "Desktop", "documents": "Documents", "downloads": "Downloads"}
_KINDS = {"text": ".txt", "txt": ".txt", "markdown": ".md", "md": ".md", "python": ".py", "json": ".json", "csv": ".csv"}
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_ASK = re.compile(
    r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|would\s+you)[,\s]+)*(?:create|make|add|generate|touch)\s+(?:me\s+)?"
    r"(?:a\s+|an\s+)?(?:new\s+)?(?:(?:empty|blank)\s+)?(?:(?P<p1>desktop|documents|downloads)\s+)?(?:(?P<k>text|txt|markdown|md|python|json|csv)\s+)?"
    r"file\b(?P<rest>.*)$", re.I)
_PLACE = re.compile(r"\b(?:on|in|at|into|to)\s+(?:my\s+|the\s+)?(?P<p>desktop|documents|downloads)\b", re.I)
_NAME = re.compile(r"\b(?:called|named|titled)\s+[\"']?(?P<n>[\w][\w .\-\\/]{0,60}?)[\"']?\s*(?:[.!?]*\s*$|\s+(?:on|in|at)\b)", re.I)


def parse(text: str) -> Optional[dict]:
    m = _ASK.match(text or "")
    if not m:
        return None
    rest = m.group("rest") or ""
    pm = _PLACE.search(rest)
    place = (m.group("p1") or (pm.group("p") if pm else "") or "desktop").lower()
    nm = _NAME.search(rest)
    name = _BAD.sub("", nm.group("n")).strip(" .") if nm else ""
    kind = (m.group("k") or "").lower()
    ext = _KINDS.get(kind, ".txt") if kind else ".txt"
    if name and re.search(r"\.[A-Za-z0-9]{1,5}$", name):
        ext = ""
    return {"place": place, "name": name or "new file", "ext": ext}


_WHERE = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:and|so|but)\s+)?where\s+(?:did\s+you\s+(?:put|create|make|save)|is|was|'?s)\s+(?:the\s+|that\s+|my\s+)?(?:new\s+)?(?:file|it)"
                    r"(?:\s+you\s+(?:just\s+)?(?:created|made|put))?\b", re.I)
last = {"path": "", "at": 0.0}


def where(text: str, now: float) -> Optional[str]:
    """"Where did you create the file?" -- answered from what was really made (the model said "in the current active window")."""
    if not _WHERE.match(text or "") or not last["path"] or now - last["at"] > 3600:
        return None
    return f"It's at {last['path']}."


def _home() -> str:
    return os.path.expanduser("~")


def make(spec: dict, write: Optional[Callable] = None) -> str:
    """Create it; what to say. `write` is the file tool: (path, content, overwrite) -> {"ok": ...}."""
    if write is None:
        from tools.file_manager import write_file_tool as _w
        write = lambda p, c, o: _w(p, c, overwrite=o)            # noqa: E731
    folder = os.path.join(_home(), _PLACES[spec["place"]])
    if not os.path.isdir(folder):
        folder = _home()
    path = os.path.join(folder, spec["name"] + spec["ext"])
    where = os.path.basename(folder)
    if os.path.exists(path):
        return f"There is already a file called {os.path.basename(path)} in your {where} folder -- I won't replace it. Give me another name."
    res = write(path, "", False) or {}
    if not res.get("ok") or not os.path.exists(path):
        return f"I couldn't create it: {res.get('error') or 'the file is not there afterwards'}."
    import time as _t
    last["path"], last["at"] = path, _t.time()
    return f"Done -- {os.path.basename(path)} is in your {where} folder, empty."
