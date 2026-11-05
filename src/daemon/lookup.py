"""
src/daemon/lookup.py

Looking something up before asking about it.

Wondering used to ask "What does ZXQ stand for?" with a blank face, even when
ZXQ was spelled out in a README on the user's disk, or defined in something
they said last month. A person would look first, and then ask to check:
"I think ZXQ means Zero Extra Queue -- is that right?". That is easier to
answer (a yes is enough), and it shows the work.

Three places, in this order, all optional:

    memory   episodic memory: a past sentence that defines the term
    files    text files under configured folders: a definition, an acronym
             spelled out next to it ("Zero Extra Queue (ZXQ)"), or, for a file
             name ("brain.py"), what the file says about itself (a module
             docstring, a README's first line)
    web      a Wikipedia summary. Only the term is sent, only for acronyms and
             quoted terms (never a project codename or a file name), with a
             short timeout. Off when the daemon runs air-gapped.

The result is a guess, never a fact: it is put to the user to confirm, and
only what they confirm is stored. Rule-based; no model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.lookup")

TEXT_EXT = frozenset(".md .txt .rst .py .js .ts .tsx .jsx .json .yaml .yml .toml .ini .cfg .html .htm "
                     ".tex .csv .java .c .cc .cpp .h .hpp .go .rs .rb .php .sh .ps1 .bat .sql .kt .swift "
                     ".org .adoc".split())
SKIP_DIRS = frozenset(".git .hg .svn node_modules __pycache__ .venv venv env .tox .mypy_cache .idea "
                      ".vscode dist build state logs .claude site-packages AppData".split())


@dataclass
class Finding:
    text: str                 # the definition, one clause
    source: str               # memory | files | web
    where: str = ""           # a file name, "wikipedia", ""
    confidence: float = 0.5


@dataclass
class LookupConfig:
    enabled: bool = True
    memory: bool = True
    files: bool = True
    web: bool = False
    roots: Tuple[str, ...] = ()
    max_files: int = 3000
    max_bytes: int = 1_000_000          # per file
    budget_s: float = 1.5               # wall time for the file search
    web_timeout_s: float = 3.0
    min_confidence: float = 0.45

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LookupConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k):
                continue
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, tuple):
                    setattr(c, k, tuple(str(x) for x in (v or ())))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                else:
                    setattr(c, k, float(v))
            except (TypeError, ValueError):
                pass
        return c


def _clean(s: str, n: int = 140) -> str:
    s = re.sub(r"[`*_#>\[\]|]+", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip(" \t\"'.,;:-")
    if len(s) > n:
        s = s[:n].rsplit(" ", 1)[0]
    return s


def _initials_match(words: List[str], acro: str) -> bool:
    """Do these words spell the acronym? Small words may be skipped
    ("Bureau of Labor Statistics" -> BLS), and a word may give two letters
    of a camel-cased acronym it starts ("JavaScript Object Notation")."""
    letters = acro.lower()
    sig = [w for w in words if w]
    if not sig:
        return False
    skip = {"of", "and", "the", "for", "a", "an", "to", "in", "on", "&"}

    def walk(i: int, j: int) -> bool:
        if j == len(letters):
            return i == len(sig) or all(w.lower() in skip for w in sig[i:])
        if i == len(sig):
            return False
        w = sig[i]
        if w.lower() in skip and walk(i + 1, j):
            return True
        # What one word can spell: its first letter, then any of its internal
        # capitals ("JavaScript" -> J, JS). A word that starts lowercase and
        # carries a capital spells by the capital ("eXtensible" -> X).
        caps = "".join(c.lower() for c in w[1:] if c.isupper())
        spells = {w[:1].lower() + caps[:n] for n in range(len(caps) + 1)}
        if w[:1].islower() and caps:
            spells |= {caps[:n] for n in range(1, len(caps) + 1)}
        for sp in sorted(spells, key=len, reverse=True):
            if sp and letters.startswith(sp, j) and walk(i + 1, j + len(sp)):
                return True
        return False

    return walk(0, 0)


def definitions_in(text: str, term: str) -> List[Tuple[str, float]]:
    """Every way `text` defines `term`, with how sure the form is."""
    if not text or not term:
        return []
    t = re.escape(term)
    acro = bool(re.fullmatch(r"[A-Z0-9]{2,8}", term))
    flags = 0 if acro else re.I
    out: List[Tuple[str, float]] = []
    # "ZXQ means / stands for / is short for ..."
    for m in re.finditer(r"(?<![\w-])" + t + r"(?![\w-])['\"]?\s+(?:means|stands for|is short for|refers to|"
                         r"is an? acronym for|is an abbreviation for)\s+([^.\n;!?]{3,140})", text, flags):
        out.append((_clean(m.group(1)), 0.8))
    # "Zero Extra Queue (ZXQ)" -- the acronym spelled out before it
    if acro:
        for m in re.finditer(r"((?:[A-Za-z][\w\-']*\s+){1,8}?)\(\s*" + t + r"\s*\)", text):
            words = m.group(1).split()
            for k in range(1, len(words) + 1):         # the fewest words that spell it
                cand = words[-k:]
                if _initials_match(cand, term):
                    out.append((" ".join(cand), 0.85))
                    break
        # "ZXQ (Zero Extra Queue)"
        for m in re.finditer(r"(?<![\w-])" + t + r"\s*\(\s*([A-Za-z][^()]{3,80})\)", text):
            if _initials_match(m.group(1).split(), term):
                out.append((_clean(m.group(1)), 0.85))
    # "ZXQ: a queue that ..." / "ZXQ - a queue ..." at the start of a line
    for m in re.finditer(r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?" + t + r"(?:\*\*)?\s*(?::|--|—|-)\s+([^\n]{3,140})",
                         text, flags):
        out.append((_clean(m.group(1)), 0.65))
    # "ZXQ is a ..." -- weakest: many sentences start like a definition
    for m in re.finditer(r"(?<![\w-])" + t + r"\s+(?:is|are)\s+((?:an?|the)\s+[^.\n;!?]{3,120})", text, flags):
        out.append((_clean(m.group(1)), 0.5))
    return [(d, c) for d, c in out if d and len(d) >= 3 and d.lower() != term.lower()]


def self_description(path: str, text: str) -> Optional[str]:
    """What a file says about itself: a Python module docstring, a README's
    first heading or paragraph, a leading comment."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        m = re.match(r'\s*(?:#[^\n]*\n\s*)*(?:[rRuU]?)("""|\'\'\')(.*?)\1', text, re.S)
        if m:
            lines = [ln.strip() for ln in m.group(2).strip().splitlines() if ln.strip()]
            base = os.path.basename(path)
            lines = [ln for ln in lines if not ln.endswith(base) and not re.fullmatch(r"[-=~]{3,}", ln)]
            para = []
            for ln in lines:
                if not ln and para:
                    break
                para.append(ln)
                if ln.endswith("."):
                    break
            s = _clean(" ".join(para), 140)
            return s or None
    if ext in (".md", ".rst", ".txt", ".org", ".adoc"):
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or re.fullmatch(r"[-=~#*]{3,}", ln):
                continue
            s = _clean(ln, 140)
            if len(s) > 3:
                return s
    m = re.match(r"\s*(?://|#|--|;)\s*([^\n]{8,140})", text)
    if m and not m.group(1).startswith("!"):
        return _clean(m.group(1))
    return None


def _wiki_fetch(url: str, timeout: float) -> Optional[Dict[str, Any]]:
    """GET a JSON summary, never blocking the caller longer than `timeout`.

    urlopen's timeout covers the socket but not name resolution, which on an
    offline machine can hang far longer; the ambient loop calls this, so the
    request runs on its own thread and is abandoned if it is late."""
    box: Dict[str, Any] = {}

    def run():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ASH-local-assistant/1.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:     # noqa: S310 -- fixed https host
                box["v"] = (json.loads(r.read(200_000).decode("utf-8", "replace"))
                            if r.status == 200 else None)
        except Exception as e:                                         # reported below
            box["e"] = e

    th = threading.Thread(target=run, daemon=True, name="ash-lookup")
    th.start()
    th.join(timeout + 0.5)
    if th.is_alive():
        raise TimeoutError("web lookup took too long")
    if "e" in box:
        raise box["e"]
    return box.get("v")


class Lookup:
    """Finds a likely meaning for a term. Every source is a plain function, so
    tests can hand in their own memory, folder and web."""

    def __init__(self, cfg: Optional[LookupConfig] = None, episodic=None,
                 fetch: Optional[Callable[[str, float], Optional[Dict[str, Any]]]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg or LookupConfig()
        self.episodic = episodic
        self.fetch = fetch or _wiki_fetch
        self.clock = clock
        self.stats = {"looked": 0, "found_memory": 0, "found_files": 0, "found_web": 0, "none": 0,
                      "web_errors": 0}

    # -- sources -------------------------------------------------------------
    def from_memory(self, term: str) -> List[Finding]:
        if not self.cfg.memory or self.episodic is None:
            return []
        try:
            eps = self.episodic.retrieve(term, top_k=8)
        except Exception:
            logger.exception("episodic lookup failed")
            return []
        out = []
        for ep in eps or []:
            for d, c in definitions_in(str(getattr(ep, "summary", "") or ""), term):
                out.append(Finding(d, "memory", "", c))
        return out

    def _walk(self) -> Iterable[str]:
        n = 0
        for root in self.cfg.roots:
            root = os.path.expanduser(root)
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    yield os.path.join(dirpath, fn)
                    n += 1
                    if n >= self.cfg.max_files:
                        return

    def from_files(self, term: str, filename: bool = False) -> List[Finding]:
        if not self.cfg.files or not self.cfg.roots:
            return []
        deadline = self.clock() + self.cfg.budget_s
        out: List[Finding] = []
        needle = term if re.fullmatch(r"[A-Z0-9]{2,8}", term) else term.lower()
        for path in self._walk():
            if self.clock() > deadline:
                break
            base = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()
            if ext not in TEXT_EXT:
                continue
            try:
                if os.path.getsize(path) > self.cfg.max_bytes:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if filename:
                if base.lower() == term.lower():
                    d = self_description(path, text)
                    if d:
                        out.append(Finding(d, "files", base, 0.7))
                continue
            hay = text if needle == term else text.lower()
            if needle not in hay:
                continue
            for d, c in definitions_in(text, term):
                out.append(Finding(d, "files", base, c))
        return out

    def from_web(self, term: str) -> List[Finding]:
        if not self.cfg.web:
            return []
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(term.replace(" ", "_"), safe=""))
        try:
            d = self.fetch(url, self.cfg.web_timeout_s)
        except Exception as e:
            self.stats["web_errors"] += 1
            logger.info("web lookup failed: %s", type(e).__name__)
            return []
        if not isinstance(d, dict) or d.get("type") in ("disambiguation", "no-extract") or not d.get("extract"):
            return []
        first = re.split(r"(?<=[.!?])\s", str(d["extract"]).strip(), maxsplit=1)[0]
        m = re.search(r"\b(?:is|was|are|refers to|stands for)\s+(.*)$", first)
        text = _clean(m.group(1) if m else first, 140)
        return [Finding(text, "web", "wikipedia", 0.6)] if text else []

    # -- the decision ----------------------------------------------------------
    def find(self, term: str, how: str = "") -> Optional[Finding]:
        """The best guess, or None. `how` is how the term came up: acronym,
        quoted, named (a project codename), activity (often a file name)."""
        if not self.cfg.enabled or not term or not term.strip():
            return None
        term = term.strip()
        self.stats["looked"] += 1
        is_file = bool(re.fullmatch(r"[\w\-. ]+\.[A-Za-z0-9]{1,5}", term))
        found: List[Finding] = []
        found += self.from_memory(term)
        found += self.from_files(term, filename=is_file)
        if not is_file and how in ("acronym", "quoted") and not any(f.confidence >= 0.8 for f in found):
            found += self.from_web(term)
        found = [f for f in found if f.confidence >= self.cfg.min_confidence]
        if not found:
            self.stats["none"] += 1
            return None
        rank = {"memory": 2, "files": 1, "web": 0}
        best = max(found, key=lambda f: (f.confidence, rank.get(f.source, 0)))
        self.stats["found_" + best.source] += 1
        return best

    def confirm_question(self, term: str, f: Finding) -> str:
        if f.source == "web":
            return f"Is {term} {f.text}? That's what Wikipedia says."
        if f.source == "files" and f.where and f.where.lower() == term.lower():
            return f"I had a look at {term}: it says it's {f.text}. Is that what it's for?"
        if f.source == "files" and f.where:
            return f"I think {term} means {f.text} -- I saw it in {f.where}. Is that right?"
        return f"I think {term} means {f.text}. Is that right?"

    def for_wonder(self, w) -> Optional[Dict[str, Any]]:
        """The hook the Wonderer calls before asking a term or activity
        question. Returns {"guess", "source", "where", "question"} or None."""
        term = str((w.data or {}).get("term") or w.subject or "")
        f = self.find(term, str((w.data or {}).get("how", "")))
        if f is None:
            return None
        return {"guess": f.text, "source": f.source, "where": f.where,
                "question": self.confirm_question(term, f)}

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "memory": self.cfg.memory, "files": self.cfg.files,
                "web": self.cfg.web, "roots": list(self.cfg.roots), **self.stats}
