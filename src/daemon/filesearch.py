"""
src/daemon/filesearch.py

"Find the contract I was editing last week" -- your own files, searched on this computer, no model, no network.

    find the contract I was editing last week        by words in the name and the text, narrowed by when it changed
    where's my resume? / find my tax return pdf      a kind of file narrows it too (pdf, word, spreadsheet, slides, photo)
    search my files for invoice                      the same, said plainly
    what did the lease say about pets?               finds the lease, then reads back the sentence about pets
    open it / yes                                    opens the best match (through the desktop executor)

What is searched: the folders in `filesearch.folders` (Documents, Desktop and Downloads by default), not hidden
folders, not program folders (node_modules, .git, virtual environments...), at most `max_files` files. Text is read
from plain text, Markdown, code, CSV, Word (.docx) and PDF files up to a few MB, the first few thousand characters.
If the sentence encoder is given (the one ASH already loads), meaning counts as well as words: "contract" finds
"agreement". The index stays in memory and in one file in state/; nothing leaves this computer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ash.daemon.filesearch")

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".py", ".js", ".ts", ".html", ".htm", ".css", ".ini",
            ".yaml", ".yml", ".toml", ".xml", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rtf", ".tex"}
KINDS = {
    "pdf": {".pdf"}, "word": {".docx", ".doc", ".odt", ".rtf"}, "document": {".docx", ".doc", ".odt", ".pdf", ".txt", ".md", ".rtf"},
    "spreadsheet": {".xlsx", ".xls", ".csv", ".ods"}, "excel": {".xlsx", ".xls"}, "slides": {".pptx", ".ppt", ".odp", ".key"},
    "presentation": {".pptx", ".ppt", ".odp", ".key"}, "photo": {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"},
    "picture": {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"}, "image": {".jpg", ".jpeg", ".png", ".heic", ".webp", ".gif"},
    "video": {".mp4", ".mov", ".mkv", ".avi"}, "text": {".txt", ".md"}, "code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs"},
    "zip": {".zip", ".rar", ".7z"},
}
_KIND_WORDS = {"pdf": "pdf", "pdfs": "pdf", "word": "word", "doc": "word", "docx": "word", "document": "document",
               "documents": "document", "spreadsheet": "spreadsheet", "sheet": "spreadsheet", "excel": "excel",
               "slides": "slides", "slideshow": "slides", "deck": "slides", "presentation": "presentation",
               "powerpoint": "slides", "photo": "photo", "photos": "photo", "picture": "picture", "pictures": "picture",
               "image": "image", "screenshot": "image", "video": "video", "zip": "zip", "archive": "zip", "code": "code",
               "script": "code", "notes": "text"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "venv", ".venv", "env", "site-packages", "appdata", "$recycle.bin",
             "windows", "program files", "program files (x86)", ".cache", "cache", "temp", "tmp", "dist", "build"}
_STOP = frozenset("""a an the my me i you your it its this that these those of in on at to for from with about and or is are was
    were be been find search look locate where whats what's which file files folder doc document documents thing stuff called named
    was were i'm im edit editing edited working worked wrote write writing made make saved save opened open last this past ago
    week weeks month months year today yesterday recently recent latest newest new old some any please can could get show
    there here pc computer ash did do does say said says tell told""".split())

_FIND = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you)[,\s]+)*"
                   r"(?:find|search\s+(?:my\s+)?(?:files|computer|documents|pc)\s+for|search\s+for|look\s+for|locate|where'?s|where\s+is|where\s+did\s+i\s+(?:put|save))\s+"
                   r"(?P<q>.{2,120}?)\s*[?.!]*\s*$", re.I)
_ABOUT = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?what\s+(?:did|does)\s+(?:the|my|that)\s+(?P<doc>.{2,60}?)\s+say\s+(?:about|on)\s+"
                    r"(?P<topic>.{2,60}?)\s*[?.!]*\s*$", re.I)
_FILEISH = re.compile(r"\b(?:file|files|document|doc|docx|pdf|spreadsheet|sheet|slides|presentation|deck|folder|photo|picture|"
                      r"screenshot|contract|invoice|resume|cv|report|essay|notes|receipt|lease|statement|form|letter|"
                      r"agreement|proposal|thesis|draft|scan|ticket|manual|slideshow|backup|zip)\b", re.I)


def words(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", (text or "").lower().replace("_", " ").replace("-", " ")):
        if w in _STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def when_range(text: str, now: float) -> Optional[Tuple[float, float]]:
    """(from, to) epoch seconds for "today", "yesterday", "last week", "this month", "recently"..., or None."""
    t = (text or "").lower()
    d = datetime.fromtimestamp(now)
    day0 = datetime(d.year, d.month, d.day)
    if re.search(r"\btoday\b", t):
        return day0.timestamp(), now
    if re.search(r"\byesterday\b", t):
        return (day0 - timedelta(days=1)).timestamp(), day0.timestamp()
    if re.search(r"\bthis\s+week\b", t):
        return (day0 - timedelta(days=day0.weekday())).timestamp(), now
    if re.search(r"\blast\s+week\b|\ba\s+week\s+ago\b", t):
        start = day0 - timedelta(days=day0.weekday() + 7)
        return start.timestamp(), (start + timedelta(days=7)).timestamp()
    if re.search(r"\bthis\s+month\b", t):
        return datetime(d.year, d.month, 1).timestamp(), now
    if re.search(r"\blast\s+month\b", t):
        first = datetime(d.year, d.month, 1)
        prev = (first - timedelta(days=1)).replace(day=1)
        return prev.timestamp(), first.timestamp()
    if re.search(r"\b(?:recently|lately|latest|newest|just\s+(?:made|saved|downloaded))\b", t):
        return now - 3 * 86400, now
    return None


def kind_of(text: str) -> Optional[str]:
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in _KIND_WORDS:
            return _KIND_WORDS[w]
    return None


def _read_text(path: str, limit: int) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in TEXT_EXT:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read(limit)
        if ext == ".docx":
            import docx                                                     # type: ignore
            return "\n".join(p.text for p in docx.Document(path).paragraphs)[:limit]
        if ext == ".pdf":
            from pypdf import PdfReader                                     # type: ignore
            out = []
            for page in PdfReader(path).pages[:6]:
                out.append(page.extract_text() or "")
                if sum(len(x) for x in out) >= limit:
                    break
            return "\n".join(out)[:limit]
    except Exception:
        logger.debug("could not read %s", path, exc_info=True)
    return ""


@dataclass
class Entry:
    path: str
    name: str
    ext: str
    mtime: float
    size: int
    words: List[str] = field(default_factory=list)       # from the name
    text: str = ""                                       # the first few thousand characters, for reading back
    vec: Optional[List[float]] = None


@dataclass
class Hit:
    entry: Entry
    score: float
    why: str = ""


class FileSearch:
    def __init__(self, folders: Optional[Sequence[str]] = None, index_path: Optional[str] = None,
                 embed: Optional[Callable[[List[str]], Any]] = None, clock: Callable[[], float] = time.time,
                 max_files: int = 20000, max_text_files: int = 3000, max_bytes: int = 5 * 1024 * 1024,
                 text_chars: int = 4000, max_depth: int = 8):
        home = os.path.expanduser("~")
        self.folders = [os.path.expanduser(f) for f in (folders if folders is not None else
                                                        [os.path.join(home, d) for d in ("Documents", "Desktop", "Downloads")])]
        self.index_path = index_path
        self.embed = embed
        self.clock = clock
        self.max_files, self.max_text_files, self.max_bytes = max_files, max_text_files, max_bytes
        self.text_chars, self.max_depth = text_chars, max_depth
        self.entries: List[Entry] = []
        self.built_at = 0.0
        self._lock = threading.RLock()
        self._building = False
        self.pending: Optional[Tuple[str, float]] = None          # a path offered to open
        self.last_path: Optional[Tuple[str, float]] = None        # the last file found, for "send it to my phone"
        self.stats = {"searches": 0, "found": 0, "opened": 0, "files": 0}
        self._load()

    # -- the index ------------------------------------------------------------------------------------------
    def _load(self):
        if not self.index_path or not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.entries = [Entry(**e) for e in raw.get("entries", [])]
            self.built_at = float(raw.get("built_at", 0.0))
            self.stats["files"] = len(self.entries)
        except Exception:
            self.entries = []

    def _save(self):
        if not self.index_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"built_at": self.built_at, "entries": [asdict(e) for e in self.entries]}, fh)
        os.replace(tmp, self.index_path)

    def _walk(self):
        n = 0
        for root in self.folders:
            if not os.path.isdir(root):
                continue
            base_depth = root.rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d.lower() not in SKIP_DIRS]
                if dirpath.count(os.sep) - base_depth >= self.max_depth:
                    dirnames[:] = []
                for f in filenames:
                    if f.startswith((".", "~$")):
                        continue
                    yield os.path.join(dirpath, f)
                    n += 1
                    if n >= self.max_files:
                        return

    def build(self) -> int:
        """(Re)build the index. Files that did not change keep what was read last time."""
        old = {e.path: e for e in self.entries}
        out: List[Entry] = []
        texts = 0
        for path in self._walk():
            try:
                st = os.stat(path)
            except OSError:
                continue
            prev = old.get(path)
            if prev is not None and abs(prev.mtime - st.st_mtime) < 1 and prev.size == st.st_size:
                out.append(prev)
                continue
            name = os.path.basename(path)
            ext = os.path.splitext(name)[1].lower()
            e = Entry(path=path, name=name, ext=ext, mtime=st.st_mtime, size=st.st_size, words=words(os.path.splitext(name)[0]))
            if texts < self.max_text_files and st.st_size <= self.max_bytes and (ext in TEXT_EXT or ext in (".docx", ".pdf")):
                e.text = _read_text(path, self.text_chars)
                texts += 1
            if self.embed is not None:
                try:
                    v = self.embed([f"{os.path.splitext(name)[0].replace('_', ' ')}. {e.text[:600]}"])[0]
                    e.vec = [round(float(x), 4) for x in v]
                except Exception:
                    e.vec = None
            out.append(e)
        with self._lock:
            self.entries = out
            self.built_at = self.clock()
            self.stats["files"] = len(out)
        try:
            self._save()
        except Exception:
            logger.exception("Could not save the file index")
        return len(out)

    def refresh(self, now: Optional[float] = None, every_s: float = 3600.0, background: bool = True):
        now = self.clock() if now is None else now
        if self._building or now - self.built_at < every_s:
            return
        self._building = True

        def run():
            try:
                self.build()
            except Exception:
                logger.exception("File index failed")
            finally:
                self._building = False
        if background:
            threading.Thread(target=run, daemon=True, name="ash-file-index").start()
        else:
            run()

    # -- searching --------------------------------------------------------------------------------------------
    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
        t = (text or "").strip()
        m = _ABOUT.match(t)
        if m and _FILEISH.search(m.group("doc")):
            return "about", {"doc": m.group("doc"), "topic": m.group("topic")}
        m = _FIND.match(t)
        if m:
            q = m.group("q")
            plain = re.match(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?search\s+(?:my\s+)?(?:files|computer|documents|pc)\s+for\b", t, re.I)
            if plain or _FILEISH.search(q) or re.search(r"\b(?:i\s+(?:was|were)\s+(?:editing|working\s+on|writing)|i\s+(?:saved|downloaded|made|wrote))\b", q, re.I):
                return "find", {"q": q}
        return None

    def search(self, query: str, now: Optional[float] = None, n: int = 3) -> List[Hit]:
        now = self.clock() if now is None else now
        want = set(words(query))
        rng = when_range(query, now)
        kind = kind_of(query)
        exts = KINDS.get(kind or "", set())
        content = want - {k for k in _KIND_WORDS}
        qv = None
        if self.embed is not None and content:
            try:
                import numpy as np
                qv = np.asarray(self.embed([" ".join(sorted(content))])[0], dtype=float)
                qv = qv / (np.linalg.norm(qv) + 1e-9)
            except Exception:
                qv = None
        hits: List[Hit] = []
        with self._lock:
            entries = list(self.entries)
        for e in entries:
            if rng is not None and not (rng[0] <= e.mtime <= rng[1]):
                continue
            if exts and e.ext not in exts:
                continue
            name_hit = len(content & set(e.words))
            text_words = set(words(e.text[:self.text_chars])) if e.text else set()
            text_hit = len(content & text_words)
            score = 1.0 * name_hit + 0.4 * text_hit
            why = "name" if name_hit else ("text" if text_hit else "")
            if qv is not None and e.vec:
                import numpy as np
                v = np.asarray(e.vec, dtype=float)
                cos = float(v @ qv / (np.linalg.norm(v) + 1e-9))
                if cos > 0.35:
                    score += 1.5 * (cos - 0.35) / 0.65
                    why = why or "meaning"
            if not content:
                score = 0.5                                     # only a kind and/or a time: all of those count
            if score <= 0:
                continue
            score += 0.15 * max(0.0, 1.0 - (now - e.mtime) / (30 * 86400))     # newer breaks ties
            hits.append(Hit(e, score, why))
        hits.sort(key=lambda h: -h.score)
        return hits[:n]

    def _say_when(self, t: float, now: float) -> str:
        d, today = datetime.fromtimestamp(t).date(), datetime.fromtimestamp(now).date()
        days = (today - d).days
        if days <= 0:
            return "today"
        if days == 1:
            return "yesterday"
        if days < 7:
            return "on " + datetime.fromtimestamp(t).strftime("%A")
        return "on " + datetime.fromtimestamp(t).strftime("%d %B").lstrip("0")

    def _where(self, path: str) -> str:
        home = os.path.expanduser("~")
        folder = os.path.dirname(path)
        rel = os.path.relpath(folder, home) if folder.lower().startswith(home.lower()) else folder
        return "your home folder" if rel in (".", "") else rel.replace(os.sep, " > ")

    def find(self, query: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        self.stats["searches"] += 1
        self.pending = None
        if not self.entries:
            return ("I'm still making a list of your files -- ask me again in a minute." if self._building
                    else "I don't have any of your files indexed. Check filesearch.folders in the settings.")
        hits = self.search(query, now)
        if not hits:
            return "I couldn't find a file like that in " + ", ".join(os.path.basename(f) for f in self.folders) + "."
        self.stats["found"] += 1
        top = hits[0].entry
        self.pending = (top.path, now)
        self.last_path = (top.path, now)
        line = f"I found {top.name} in {self._where(top.path)}, changed {self._say_when(top.mtime, now)}."
        others = [h.entry.name for h in hits[1:] if h.score >= 0.6 * hits[0].score]
        if others:
            line += " Also: " + ", ".join(others) + "."
        return line + " Want me to open it?"

    def about(self, doc: str, topic: str, now: Optional[float] = None) -> str:
        """Find the document, then read back the sentence that says most about the topic -- no model."""
        now = self.clock() if now is None else now
        hits = [h for h in self.search(doc, now, n=5) if h.entry.text]
        if not hits:
            return f"I couldn't find a {doc.strip()} I can read."
        e = hits[0].entry
        want = set(words(topic))
        best, best_n = "", 0
        for sent in re.split(r"(?<=[.!?])\s+|\n+", e.text):
            s = sent.strip()
            if not 8 <= len(s) <= 400:
                continue
            k = len(want & set(words(s)))
            if k > best_n:
                best, best_n = s, k
        self.pending = (e.path, now)
        self.last_path = (e.path, now)
        if not best:
            return f"{e.name} doesn't seem to say anything about {topic.strip()} -- at least not in the part I read. Want me to open it?"
        return f"In {e.name}: “{best}” Want me to open it?"

    YES = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:yes|yeah|yep|sure|ok(?:ay)?|open\s+it|please(?:\s+do)?|go\s+ahead)\b[\s.!]*$", re.I)

    def reply(self, text: str, open_fn: Callable[[str], bool], now: Optional[float] = None, window_s: float = 90.0) -> Optional[str]:
        if self.pending is None or not self.YES.match(text or ""):
            return None
        now = self.clock() if now is None else now
        path, at = self.pending
        self.pending = None
        if now - at > window_s:
            return None
        ok = False
        try:
            ok = bool(open_fn(path))
        except Exception:
            ok = False
        if ok:
            self.stats["opened"] += 1
            return f"Opening {os.path.basename(path)}."
        return f"I couldn't open {os.path.basename(path)}."

    def status(self) -> Dict[str, Any]:
        return {**self.stats, "indexing": self._building, "folders": len(self.folders)}
