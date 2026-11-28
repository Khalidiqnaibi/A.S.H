"""
src/daemon/pipeline.py

Things ASH can already do, chained like a Unix pipe: what one stage produces feeds the next.

    "Find the lease, summarize it, and tell me what it says about pets."          find files | read | summarize | pick out "pets" | say
    "Search Google for pygame sprites, read the first result and save it as a note."  search | read page | note
    research NPU | summarize | note                                               the same, with a pipe, for whoever wants to type it

No syntax to learn: stages are found by what they do ("find ...", "summarize it", "save it as a note"), joined by commas, "then", "and then", "and", or a "|". Each stage declares what it
takes and what it gives -- text, files, page, none -- so a stage that gives files can feed one that wants text (the first file is read: an adapter), and a pipe that cannot be
joined is said so by stage, before anything runs.

    sequential     the default: each stage's output is the next stage's input
    parallel       "... at the same time as ...", "... meanwhile ...": independent pipelines run together and answer in one line
    background     "... in the background": started at once, spoken when done, listed by "what are you doing?", stopped by "stop that"
    confirmed once the plan is read back before the FIRST stage that changes something (a note, a copy, a remembered term); read-only pipes just run

Every real thing a stage reaches is behind `Ctx`, so tests use a fake one and nothing is opened, searched, copied or written.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------------------------------------- stages
TEXT, FILES, PAGE, NONE = "text", "files", "page", "none"


@dataclass
class Stage:
    name: str
    takes: str
    gives: str
    effect: bool = False                 # changes something: a note, a copy, a term
    arg: str = ""                        # the stage's own argument ("pygame sprites", "pets")
    say: str = ""                        # how it is read back


# (name, takes, gives, effect, regex with a named group `a` for its argument)
_DEFS: List[Tuple[str, str, str, bool, str]] = [
    ("find_files", NONE, FILES, False, r"(?:find|locate|look\s+for|dig\s+up|pull\s+up|get)\s+(?:me\s+)?(?:the\s+|my\s+|that\s+)?(?P<a>[\w .'-]{2,40}?)(?:\s+(?:file|document|doc|pdf))?"),
    ("search_web", NONE, PAGE, False, r"(?:search|google|look\s+up|research)(?:\s+google)?(?:\s+for)?\s+(?P<a>[\w .'+#-]{2,60}?)"),
    ("memory", NONE, TEXT, False, r"(?:search|check|look\s+through)\s+(?:my\s+|your\s+)?memory\s+(?:for|about)\s+(?P<a>[\w .'-]{2,40}?)"),
    ("read_file", FILES, TEXT, False, r"(?:read|open|load)\s+(?:it|that|them|the\s+file|the\s+first(?:\s+one)?)"),
    ("read_page", PAGE, TEXT, False, r"(?:read|get|grab)\s+(?:the\s+)?(?:first\s+result|page|article|result|text|it|that)(?:\s+(?:from|of)\s+the\s+page)?"),
    ("summarize", TEXT, TEXT, False, r"(?:summari[sz]e|sum\s+up|give\s+me\s+the\s+gist\s+of|shorten|boil\s+down)(?:\s+(?:it|that|them|this))?"),
    ("about", TEXT, TEXT, False, r"(?:tell\s+me|what\s+does\s+(?:it|that|the\s+\w+)\s+say|show\s+me|pick\s+out|find)\s+(?:what\s+it\s+says\s+)?(?:about|on)\s+(?P<a>[\w .'-]{2,40}?)"),
    ("note", TEXT, NONE, True, r"(?:save|keep|put|write|store)\s+(?:it|that|this|them)\s+(?:as|in|into|to)\s+(?:a\s+|my\s+)?note|(?:make\s+a\s+)?note\s+(?:of\s+)?(?:it|that)|note\s+(?:it|that)\s+down"),
    ("remember_as", TEXT, NONE, True, r"(?:remember|keep|learn)\s+(?:it|that)\s+as\s+(?:what\s+)?(?:the\s+)?(?P<a>[\w .'-]{2,40}?)"),
    ("copy", TEXT, NONE, True, r"copy\s+(?:it|that|this)(?:\s+to\s+(?:the\s+)?clipboard)?"),
    ("say", TEXT, NONE, False, r"(?:tell|say|read)\s+(?:it|that|them)(?:\s+(?:out|to\s+me|back))?|tell\s+me|let\s+me\s+know|(?:and\s+)?show\s+me\s+(?:it|that)"),
]
_RX = [(n, t, g, e, re.compile(r"^\s*(?:(?:and|then|and\s+then|next|finally|also)\s+)*(?:" + p + r")\s*[.,;!?]*\s*$", re.I)) for n, t, g, e, p in _DEFS]
_SPLIT = re.compile(r"\s*(?:\|+|;|,\s*(?:and\s+)?(?:then\s+)?|\s+and\s+then\s+|\s+then\s+|\s+and\s+)\s*", re.I)
_PARALLEL = re.compile(r"\s*(?:,\s*)?(?:meanwhile|at\s+the\s+same\s+time(?:\s+as\s+that)?|while\s+you(?:'re|\s+are)\s+at\s+it|and\s+also\s+in\s+parallel|in\s+parallel(?:\s+with)?)\s*,?\s*", re.I)
_BACKGROUND = re.compile(r"\s*(?:,\s*)?(?:in\s+the\s+background|in\s+background|and\s+tell\s+me\s+when\s+(?:it's|it\s+is|you're|you\s+are)\s+(?:done|finished|ready))\s*[.!?]*\s*$", re.I)
_LEAD = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?", re.I)


def match_stage(clause: str) -> Optional[Stage]:
    c = (clause or "").strip()
    for name, takes, gives, effect, rx in _RX:
        m = rx.match(c)
        if m:
            arg = (m.groupdict().get("a") or "").strip(" .,'")
            if name in ("find_files", "search_web", "memory", "about", "remember_as") and not arg:
                continue
            if name == "find_files" and re.fullmatch(r"(?:it|that|this|them|something|anything|me)", arg, re.I):
                continue
            return Stage(name, takes, gives, effect, arg, _say(name, arg))
    return None


def _say(name: str, arg: str) -> str:
    return {"find_files": f"find {arg}", "search_web": f"search for {arg}", "memory": f"search my memory for {arg}", "read_file": "read it", "read_page": "read the page",
            "summarize": "summarize it", "about": f"pick out what it says about {arg}", "note": "keep it as a note", "remember_as": f"remember it as {arg}", "copy": "copy it",
            "say": "tell you"}.get(name, name)


def parse(text: str) -> Optional[Dict[str, Any]]:
    """{"parallel": [[Stage...], ...], "background": bool} for a sentence that is a pipe of two or more known stages; None otherwise (a single request is not a pipe)."""
    t = _LEAD.sub("", (text or "").strip())
    if not t or len(t) > 400:
        return None
    background = bool(_BACKGROUND.search(t))
    t = _BACKGROUND.sub("", t)
    groups = [g for g in _PARALLEL.split(t) if g and g.strip()]
    out: List[List[Stage]] = []
    for g in groups:
        pieces = [p for p in _SPLIT.split(g) if p and p.strip()]
        stages: List[Stage] = []
        for p in pieces:
            st = match_stage(p)
            if st is not None:
                stages.append(st)
            elif stages and stages[-1].arg:
                # a fragment that is not a stage: part of the previous argument ("pygame and sprites" split at "and")
                stages[-1].arg = (stages[-1].arg + " " + p).strip()
                stages[-1].say = _say(stages[-1].name, stages[-1].arg)
            else:
                return None
        if stages:
            out.append(stages)
    total = sum(len(g) for g in out)
    if total < 2 and not (len(out) >= 2):
        return None
    return {"parallel": out, "background": background}


def check_types(stages: List[Stage]) -> Optional[str]:
    """Why these stages cannot be joined, by stage -- before anything runs. Adapters: files -> text (the first file is read), page -> text (the page is read)."""
    have = NONE
    for i, st in enumerate(stages, 1):
        if st.takes != NONE and have == NONE and i == 1:
            return f"{st.say} needs something to work on -- start with a find, a search or your memory."
        if st.takes != NONE and have != st.takes:
            if have == FILES and st.takes == TEXT:
                pass
            elif have == PAGE and st.takes == TEXT:
                pass
            elif have == NONE:
                return f"stage {i} ({st.say}) has nothing to work on: the stage before it gives nothing."
            else:
                return f"stage {i} ({st.say}) wants {st.takes}, but the stage before it gives {have}."
        have = st.gives if st.gives != NONE else have if st.name in ("say",) else st.gives
    return None


def plan_line(par: List[List[Stage]], background: bool) -> str:
    parts = [" → ".join(s.say for s in g) for g in par]
    return ("; at the same time ".join(parts)) + (" (in the background)" if background else "")


def needs_yes(par: List[List[Stage]]) -> bool:
    return any(s.effect for g in par for s in g)


# ---------------------------------------------------------------------------------------------------------------------------- extractive summary
_STOP = set("the a an and or but of to in on at for with from by is are was were be been it its this that these those as at not no so if then than which who whom what when where why how "
            "i you he she we they me him her them my your our their his can could will would should may might do does did have has had also very just about into over under again".split())


def sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(])", t) if len(s.strip()) >= 20]


def summarize(text: str, n: int = 3) -> str:
    """The n sentences that carry the most of the text's own words, kept in their order. No model, no network."""
    ss = sentences(text)
    if len(ss) <= n:
        return " ".join(ss) if ss else (text or "").strip()[:300]
    freq: Dict[str, int] = {}
    for s in ss:
        for w in set(re.findall(r"[a-z]{3,}", s.lower())):
            if w not in _STOP:
                freq[w] = freq.get(w, 0) + 1
    def score(s: str) -> float:
        ws = [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in _STOP]
        return sum(freq.get(w, 0) for w in set(ws)) / (len(ws) ** 0.5 + 1)
    top = sorted(range(len(ss)), key=lambda i: -score(ss[i]))[:n]
    return " ".join(ss[i] for i in sorted(top))


def about(text: str, topic: str, n: int = 2) -> str:
    words = [w for w in re.findall(r"[a-z]{3,}", topic.lower()) if w not in _STOP] or re.findall(r"[a-z]+", topic.lower())
    hits = [s for s in sentences(text) if any(w in s.lower() for w in words)]
    return " ".join(hits[:n]) if hits else ""


# ---------------------------------------------------------------------------------------------------------------------------- running
class Ctx:
    """What a stage reaches. A fake in tests; built by the runtime for real."""

    def find_files(self, q: str) -> List[str]: ...
    def read_file(self, path: str) -> str: ...
    def search_open(self, q: str) -> bool: ...
    def read_page(self) -> str: ...
    def memory(self, q: str) -> str: ...
    def note(self, text: str) -> bool: ...
    def remember(self, name: str, text: str) -> bool: ...
    def copy(self, text: str) -> bool: ...


@dataclass
class Run:
    id: int
    text: str
    stages: List[Stage]
    state: str = "running"                  # running | done | failed | cancelled
    at: int = 0                             # index of the stage in progress
    result: str = ""
    started: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    value: Any = None


def run_stages(ctx: Ctx, run: Run, pause: Callable[[float], None] = time.sleep) -> str:
    """Runs one pipe; returns what to say. A stage that fails ends it with the stage named."""
    val: Any = None
    kind = NONE
    last_say = ""
    for i, st in enumerate(run.stages):
        if run.cancel.is_set():
            run.state = "cancelled"
            return "Stopped."
        run.at = i
        try:
            # adapters
            if st.takes == TEXT and kind == FILES:
                paths = list(val or [])
                if not paths:
                    return _fail(run, i, st, "no file was found")
                val, kind = ctx.read_file(paths[0]) or "", TEXT
                if not val:
                    return _fail(run, i, st, f"I couldn't read {paths[0]}")
            if st.takes == TEXT and kind == PAGE:
                pause(1.5)
                val, kind = ctx.read_page() or "", TEXT
                if not val:
                    return _fail(run, i, st, "I couldn't read the page")
            n = st.name
            if n == "find_files":
                val, kind = ctx.find_files(st.arg), FILES
                if not val:
                    return _fail(run, i, st, f"I couldn't find {st.arg}")
            elif n == "search_web":
                if not ctx.search_open(st.arg):
                    return _fail(run, i, st, "I couldn't open the search")
                val, kind = True, PAGE
            elif n == "memory":
                val, kind = ctx.memory(st.arg) or "", TEXT
            elif n == "read_file":
                if kind != FILES:
                    pass                                   # already text
            elif n == "read_page":
                pause(1.5)
                val, kind = ctx.read_page() or "", TEXT
                if not val:
                    return _fail(run, i, st, "I couldn't read the page")
            elif n == "summarize":
                val = summarize(str(val))
            elif n == "about":
                got = about(str(val), st.arg)
                if not got:
                    return _fail(run, i, st, f"nothing in it about {st.arg}")
                val = got
            elif n == "note":
                if not ctx.note(str(val)):
                    return _fail(run, i, st, "the note wasn't kept")
                kind, last_say = NONE, "Kept as a note."
            elif n == "remember_as":
                if not ctx.remember(st.arg, str(val)):
                    return _fail(run, i, st, "I couldn't remember it")
                kind, last_say = NONE, f"Remembered as {st.arg}."
            elif n == "copy":
                if not ctx.copy(str(val)):
                    return _fail(run, i, st, "the clipboard wasn't reachable")
                kind, last_say = NONE, "Copied."
            elif n == "say":
                last_say = str(val)[:320]
        except Exception as e:                                # a stage that raises ends the pipe, named
            return _fail(run, i, st, f"{type(e).__name__}")
    run.state = "done"
    run.value = val
    if last_say:
        return last_say if len(run.stages) == 1 or run.stages[-1].name == "say" else (last_say + (" " + str(val)[:200] if isinstance(val, str) and run.stages[-1].name in ("note", "remember_as", "copy") and False else ""))
    return (str(val)[:320] if isinstance(val, str) and val else "Done.")


def _fail(run: Run, i: int, st: Stage, why: str) -> str:
    run.state = "failed"
    return f"Stopped at stage {i + 1} ({st.say}): {why}."


class Runs:
    """The pipes in flight: listed by "what are you doing?", stopped by "stop that"."""

    def __init__(self):
        self.items: List[Run] = []
        self._n = 0
        self._lock = threading.Lock()

    def new(self, text: str, stages: List[Stage]) -> Run:
        with self._lock:
            self._n += 1
            r = Run(self._n, text, stages)
            self.items.append(r)
            self.items = self.items[-20:]
            return r

    def active(self) -> List[Run]:
        return [r for r in self.items if r.state == "running"]

    def status(self) -> str:
        a = self.active()
        if not a:
            return "Nothing running right now."
        return "; ".join(f"{' → '.join(s.say for s in r.stages)} (stage {r.at + 1} of {len(r.stages)}, {int(time.time() - r.started)} s)" for r in a) + "."

    def stop(self) -> str:
        a = self.active()
        if not a:
            return ""
        a[-1].cancel.set()
        return "Stopping " + " → ".join(s.say for s in a[-1].stages) + "."


_STATUS = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:what\s+are\s+you\s+(?:doing|working\s+on|running)(?:\s+right\s+now)?|what'?s\s+(?:running|in\s+the\s+background)|how'?s\s+(?:the\s+|that\s+)?(?:research|search|pipeline|job)\s*(?:going)?)\W*$", re.I)
_STOP_THAT = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:stop|cancel|abort|kill)\s+(?:that|it|the\s+(?:research|search|pipeline|job))(?:\s+one)?\W*$", re.I)


def asks_status(t: str) -> bool:
    return bool(_STATUS.match((t or "").strip()))


def asks_stop(t: str) -> bool:
    return bool(_STOP_THAT.match((t or "").strip()))


# ---------------------------------------------------------------------------------------------------------------------------- the page in front
def html_to_text(html: str, limit: int = 20000) -> str:
    """The readable text of a page: scripts, styles and tags gone, entities mended, white space squeezed."""
    import html as _h
    t = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html or "")
    t = re.sub(r"(?is)<!--.*?-->", " ", t)
    t = re.sub(r"(?i)</?(?:p|div|br|li|h[1-6]|tr|section|article)\b[^>]*>", "\n", t)
    t = _h.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    return t[:limit]


def fetch_page(url: str, timeout: float = 6.0) -> str:
    """The page at `url` as text; raises with no connection (the stage then says it couldn't read it)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(1_500_000)
        enc = r.headers.get_content_charset() or "utf-8"
    return html_to_text(raw.decode(enc, "ignore"))


def front_page_text(current: Optional[Dict[str, Any]], fetch: Callable[[str], str] = fetch_page) -> str:
    """The text of the tab in front (the browser-tabs sensor's current tab), "" when there is none or it can't be fetched."""
    url = (current or {}).get("url") or ""
    if not url:
        return ""
    try:
        return fetch(url) or ""
    except Exception:
        return ""
