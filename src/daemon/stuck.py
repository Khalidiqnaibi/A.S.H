"""
src/daemon/stuck.py

Help when stuck, without the model.

The flow detector already notices the same error coming back and offers
help ("That error keeps coming back (KeyError: 'id') -- want me to take a
look?"). Said yes to, this does two concrete things before anything else:

  1. looks for that error in the user's own files -- notes, logs, old
     scripts under the lookup folders -- where it may have been met, and
     solved, before; the files and lines are said back
  2. starts a background look-up of it (research.py), told when it is ready

Only the error's own words are searched for, and only in the folders set for
look-ups (wondering.lookup.roots). The web look-up sends the error's key
line, nothing else, and only if research is on.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable, Iterable, List, Optional, Tuple

TEXT_EXT = frozenset(".md .txt .rst .py .js .ts .tsx .jsx .json .yaml .yml .toml .ini .cfg .log .html .htm "
                     ".java .c .cpp .h .cs .go .rs .rb .php .sh .ps1 .bat .sql .ipynb".split())
SKIP_DIRS = frozenset(".git .hg .svn node_modules __pycache__ .venv venv env .tox .mypy_cache .idea "
                      "dist build site-packages .cache".split())


def needle(error: str) -> Optional[str]:
    """The part of an error worth searching for: "KeyError: 'start work'"
    stays as it is; a long message is cut to its distinctive start; paths,
    line numbers and addresses that change every time are dropped."""
    t = (error or "").strip().splitlines()[-1] if (error or "").strip() else ""
    t = re.sub(r"(?i)^\s*traceback[^:]*:\s*", "", t)
    t = re.sub(r"0x[0-9a-f]+", "", t, flags=re.I)
    t = re.sub(r"(?:[A-Za-z]:)?[\\/][^\s'\"]+", "", t)          # paths
    t = re.sub(r"\bline \d+\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" :,")
    if len(t) < 6:
        return None
    return t[:60].strip()


def find_in_files(error: str, roots: Iterable[str], budget_s: float = 3.0, max_files: int = 4000,
                  max_bytes: int = 1_000_000, limit: int = 3,
                  clock: Callable[[], float] = time.monotonic) -> List[Tuple[str, int, str]]:
    """(path, line number, the line) where the error's words appear, newest
    file first, at most `limit`."""
    n = needle(error)
    if not n:
        return []
    low = n.lower()
    deadline = clock() + budget_s
    hits: List[Tuple[float, str, int, str]] = []
    seen = 0
    for root in roots:
        root = os.path.expanduser(str(root))
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if clock() > deadline or seen >= max_files:
                    break
                if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                    continue
                seen += 1
                path = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(path) > max_bytes:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if low in line.lower():
                                hits.append((os.path.getmtime(path), path, i, line.strip()[:120]))
                                break
                except OSError:
                    continue
    hits.sort(key=lambda h: -h[0])
    return [(p, i, line) for _, p, i, line in hits[:limit]]


def describe(error: str, hits: List[Tuple[str, int, str]], research_started: bool) -> str:
    n = needle(error) or error
    if hits:
        where = "; ".join(f"{os.path.basename(p)} line {i}" for p, i, _ in hits)
        first = f"I found “{n[:50]}” in your files before: {where}."
    else:
        first = f"I didn't find “{n[:50]}” anywhere in your files."
    if research_started:
        return first + " I'm also looking it up -- I'll tell you what I find."
    return first


def research_topic(error: str) -> Optional[str]:
    n = needle(error)
    return n if n else None


def tidy_hits(hits: List[Tuple[str, int, str]], home: Optional[str] = None) -> List[str]:
    home = home or os.path.expanduser("~")
    out = []
    for p, i, line in hits:
        shown = "~" + p[len(home):] if p.startswith(home) else p
        out.append(f"{shown}:{i}  {line}")
    return out


__all__ = ["needle", "find_in_files", "describe", "research_topic", "tidy_hits"]
