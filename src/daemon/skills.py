"""
src/daemon/skills.py

Teaching ASH a concept it has never seen, out of tools it already has -- and having it use them the way it was taught.

    "To search Google for X means: open a new tab, go to google.com, type X, press enter, click the first result, read the title, save it as a note."

That one sentence is a *skill*: a name, slots (X), and steps. Each step is one of a small fixed set of actions ASH already has -- open a page, press a key,
type, click what is written on the screen, click the first result, read the title or the screen, wait, keep a note, say it, copy it, remember it as a
term, or run another skill. Nothing in a skill is code and nothing in it is decided by the language model: it is data, read back and run by rules.

    what makes it more than a saved plan
      slots         one lesson works for any X
      results       a step can leave a value ("read the title") that later steps use ("save it as a note", "remember it as <X>")
      checks        after a step that should change the screen, the window's title is looked at; if nothing changed the run stops and says which step failed
      composition   a step can run another skill ("do research X"), up to three deep
      supervision   the first runs are read back and need a yes; a skill that has worked twice by itself is run straight away
      showing       "watch me" ... "that's how you open my notes": the windows and addresses that opened in between become the steps (no keystrokes are
                    recorded, and no window titles)
      limits        no step can send, pay, buy, delete, submit, download, install or type a secret: a lesson with one is refused when it is TAUGHT

The tools are reached through `Hands` (a thing with open_url, keys, type_text, click_text, click_first_result, title, read_screen, copy, note, remember, say),
so tests never touch the mouse, keyboard, browser or files of this machine.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.skills")

MAX_STEPS = 12
MAX_DEPTH = 3
MAX_EACH = 5
TRUST_AFTER = 2

SITES = {"google": "google.com", "youtube": "youtube.com", "wikipedia": "wikipedia.org", "github": "github.com", "bing": "bing.com",
         "duckduckgo": "duckduckgo.com", "reddit": "reddit.com", "gmail": "mail.google.com", "maps": "maps.google.com", "gemini": "gemini.google.com",
         "chatgpt": "chatgpt.com", "claude": "claude.ai", "stackoverflow": "stackoverflow.com", "stack overflow": "stackoverflow.com", "twitter": "x.com",
         "amazon": "amazon.com", "netflix": "netflix.com", "spotify": "open.spotify.com", "arxiv": "arxiv.org", "scholar": "scholar.google.com"}
ENGINES = {"google": "https://www.google.com/search?q=", "youtube": "https://www.youtube.com/results?search_query=", "bing": "https://www.bing.com/search?q=",
           "duckduckgo": "https://duckduckgo.com/?q=", "wikipedia": "https://en.wikipedia.org/w/index.php?search="}

# What a skill may never do. Checked when it is taught and again when it runs.
_FORBIDDEN_TARGET = re.compile(r"\b(?:send|pay|buy|purchase|order|checkout|check\s+out|delete|remove|erase|submit|confirm|accept|agree|sign\s+in|log\s*in|"
                               r"sign\s+up|transfer|post|publish|download|install|uninstall|unsubscribe|donate|subscribe|place\s+order)\b", re.I)
_SECRET = re.compile(r"\b(?:password|passcode|pin\s+code|cvv|card\s+number|credit\s+card|ssn|social\s+security|secret\s+key|api\s+key|token)\b|\d{12,}", re.I)
_SLOT_WORDS = {"x", "y", "z", "something", "anything"}
_IT = re.compile(r"^(?:it|that|this|the\s+thing|the\s+topic|the\s+result|the\s+answer|them)$", re.I)

_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}
_KEYS = {"enter": ("enter",), "return": ("enter",), "escape": ("esc",), "esc": ("esc",), "tab": ("tab",), "space": ("space",), "down": ("down",), "up": ("up",),
         "page down": ("pagedown",), "page up": ("pageup",), "backspace": ("backspace",), "control l": ("ctrl", "l"), "control f": ("ctrl", "f"),
         "control t": ("ctrl", "t"), "control w": ("ctrl", "w"), "control a": ("ctrl", "a"), "control c": ("ctrl", "c"), "control v": ("ctrl", "v")}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


def _n(t: str) -> str:
    return " ".join(_norm(t).split())


# ------------------------------------------------------------------------------------------------------------------------------ reading a lesson
class LessonError(ValueError):
    """A lesson ASH cannot take, with what to say about it."""


def _site(s: str) -> Optional[str]:
    t = re.sub(r"\s+dot\s+", ".", (s or "").strip().lower()).strip(" .")
    t = re.sub(r"^(?:the\s+)?(?:website|site|page)\s+", "", t)
    if t in SITES:
        return SITES[t]
    if re.fullmatch(r"(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+", t):
        return t
    return None


def _var(arg: str, slots: List[str]) -> Any:
    """A step's argument: a slot reference ("$X"), the result ("$it"), or literal text."""
    a = (arg or "").strip().strip("\"'“”").strip()
    low = a.lower()
    for s in slots:
        if low == s.lower():
            return "$" + s
    if _IT.match(low):
        return "$it"
    return a


def parse_step(clause: str, slots: List[str]) -> Optional[Dict[str, Any]]:
    """One spoken clause -> one step, or None when it is not something ASH can do."""
    c = re.sub(r"\s+", " ", (clause or "").strip(" .,;!")).strip()
    c = re.sub(r"^(?:and|then|and\s+then|next|after\s+that|finally)\s+", "", c, flags=re.I)
    low = c.lower()
    if not low:
        return None
    if re.fullmatch(r"(?:open|make|create)\s+(?:a\s+)?new\s+tab", low):
        return {"do": "new_tab"}
    m = re.fullmatch(r"search\s+(?:(?P<e>google|youtube|bing|duckduckgo|wikipedia)\s+)?for\s+(?P<q>.+)", low)
    if m:
        return {"do": "search", "engine": m.group("e") or "google", "q": _var(m.group("q"), slots)}
    m = re.fullmatch(r"(?:go\s+to|open|visit|navigate\s+to|browse\s+to)\s+(?P<s>.+)", low)
    if m:
        site = _site(m.group("s"))
        if site:
            return {"do": "open_url", "url": "https://" + site}
        app = re.sub(r"^(?:the\s+|my\s+)", "", m.group("s").strip())
        if re.fullmatch(r"[a-z][a-z0-9 .+-]{1,30}", app) and not _site(app):
            return {"do": "open_app", "app": app}
        return None
    m = re.fullmatch(r"(?:type|write|enter|put)\s+(?:in\s+)?(?P<t>.+?)(?:\s+(?:in|into)\s+the\s+(?:search\s+)?(?:box|bar|field))?", low)
    if m:
        return {"do": "type", "text": _var(m.group("t"), slots)}
    m = re.fullmatch(r"(?:press|hit|push)\s+(?P<k>.+)", low)
    if m:
        k = re.sub(r"\s*\+\s*", " ", re.sub(r"\bctrl\b", "control", m.group("k").strip().replace("the ", "").replace(" key", "")))
        k = re.sub(r"\s+", " ", k).strip()
        if k in _KEYS:
            return {"do": "press", "keys": list(_KEYS[k])}
        return None
    if re.fullmatch(r"click\s+(?:on\s+)?the\s+first\s+(?:search\s+)?(?:result|link)", low):
        return {"do": "click_first_result"}
    m = re.fullmatch(r"click\s+(?:on\s+)?(?P<t>.+)", low)
    if m:
        return {"do": "click_text", "text": _var(m.group("t"), slots)}
    m = re.fullmatch(r"wait(?:\s+for)?\s+(?:(?P<n>\d+|a|one|two|three|four|five|six|ten)\s+seconds?|the\s+page(?:\s+to\s+load)?|a\s+(?:bit|moment|second))", low)
    if m:
        n = m.group("n")
        return {"do": "wait", "s": float(n) if (n or "").isdigit() else float(_NUMW.get(n or "", 3))}
    m = re.fullmatch(r"(?:read|get|take)\s+(?:the\s+)?(?P<w>page\s+title|title|first\s+result|heading|page|screen|text)(?:\s+(?:as|into)\s+\w+)?", low)
    if m:
        return {"do": "read_screen" if m.group("w") in ("page", "screen", "text") else "read_title"}
    m = re.fullmatch(r"(?:if|when)\s+(?:the\s+)?(?:page|screen)?\s*(?:shows?|says?|has)\s+(?P<t>.+?)(?:\s*,\s*|\s+)(?:then\s+)?(?:stop|tell\s+me)", low)
    if m:
        return {"do": "stop_if_text", "text": m.group("t").strip(" \"'")}
    m = re.fullmatch(r"(?:if|when)\s+(?:the\s+)?(?:page|screen)\s*(?:shows?|says?|has)\s+(?P<t>.+?)\s*,\s*(?:then\s+)?(?:do|run)\s+(?P<n>[a-z][a-z0-9 ]{2,60})", low)
    if m:
        return {"do": "branch", "text": m.group("t").strip(" \"'"), "then": m.group("n").strip()}
    if re.fullmatch(r"(?:save|put|write|keep)\s+(?:it|that|the\s+(?:title|result|answer))\s+(?:in|as|into)\s+(?:a\s+|my\s+)?note", low):
        return {"do": "note"}
    if re.fullmatch(r"(?:tell\s+me(?:\s+(?:it|that|the\s+(?:title|result|answer)))?|say\s+(?:it|that|the\s+(?:title|result|answer))|read\s+(?:it|that)\s+(?:out|to\s+me))", low):
        return {"do": "say"}
    if re.fullmatch(r"copy\s+(?:it|that|the\s+(?:title|result|answer))(?:\s+to\s+(?:the\s+)?clipboard)?", low):
        return {"do": "copy"}
    m = re.fullmatch(r"(?:remember|keep|learn)\s+(?:it|that)\s+as\s+(?:what\s+)?(?:the\s+)?(?P<n>.+?)(?:\s+(?:is|means))?", low)
    if m:
        return {"do": "remember", "name": _var(m.group("n"), slots)}
    m = re.fullmatch(r"(?:do|run|use)\s+(?P<t>[a-z][a-z0-9 ]{2,60})", low)
    if m:
        return {"do": "run_skill", "text": m.group("t").strip()}
    return None


def split_clauses(body: str) -> List[str]:
    t = re.sub(r"\s+dot\s+", ".", body or "", flags=re.I)
    t = re.sub(r"((?:if|when)\s+the\s+(?:page|screen)\s+(?:shows?|says?|has)\s+[^,;]+?),\s*(?:then\s+)?(do|run)\s", lambda m: m.group(1) + "§ " + m.group(2) + " ", t, flags=re.I)
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?(?:then\s+)?|;\s*|\.\s+(?=[A-Z])|\s+and\s+then\s+|\s+then\s+)", t)
    return [p.strip(" .,;").replace("§", ",") for p in parts if p and p.strip(" .,;")]


def check_allowed(steps: List[Dict[str, Any]]) -> None:
    for st in steps:
        for key in ("text", "q", "app", "name"):
            v = st.get(key)
            if isinstance(v, str) and not v.startswith("$"):
                if st["do"] in ("click_text", "type", "open_app", "search") and _FORBIDDEN_TARGET.search(v):
                    raise LessonError(f"I won't learn a step that touches “{v}” -- sending, paying, buying, deleting, submitting, downloading and "
                                      "installing are never part of something I do by myself.")
                if _SECRET.search(v) and st["do"] in ("type", "search", "click_text"):
                    raise LessonError("I won't learn a step that types something like a password or a card number.")


_HEAD = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|please|now)[,\s]+)*(?:to|when\s+i\s+say|teach\s+you\s+to|learn\s+to|learn\s+how\s+to|here'?s\s+how\s+to|"
                   r"this\s+is\s+how\s+(?:you|to))\s+(?P<phrase>[^:,]{3,70}?)\s*(?:,?\s*(?:means?|is\s+to|you\s+(?:should|do))\s*:?|:)\s*(?P<body>.{6,600})$", re.I)


def _slots_of(phrase: str) -> List[str]:
    out = []
    for w in re.findall(r"[A-Za-z]+", phrase):
        if w.lower() in _SLOT_WORDS and w.upper() not in out:
            out.append(w.upper() if len(w) == 1 else w.lower())
    return out[:2]


def parse_lesson(text: str) -> Optional[Dict[str, Any]]:
    """The skill taught in a sentence, or None when the sentence is not a lesson. LessonError (with what to say) when it is, but cannot be taken."""
    m = _HEAD.match((text or "").strip())
    if not m:
        return None
    phrase = re.sub(r"\s+", " ", m.group("phrase").strip(" ."))
    slots = _slots_of(phrase)
    clauses = split_clauses(m.group("body"))
    if not clauses:
        return None
    steps: List[Dict[str, Any]] = []
    for c in clauses:
        st = parse_step(c, slots)
        if st is None:
            raise LessonError(f"I can't do “{c}” yet -- I can open pages and apps, press keys, type, click what's written on the screen, click the "
                              "first result, read the title, wait, keep a note, remember something, or run another thing you've taught me. Tell me that step "
                              "another way and I'll learn the whole thing.")
        steps.append(st)
    # "open a new tab" before "go to <site>": opening an address already opens its own tab.
    steps = [s for i, s in enumerate(steps) if not (s["do"] == "new_tab" and i + 1 < len(steps) and steps[i + 1]["do"] in ("open_url", "search"))]
    if len(steps) > MAX_STEPS:
        raise LessonError(f"That's {len(steps)} steps -- I'll take up to {MAX_STEPS}. Teach me the first part as its own thing, then a second that uses it.")
    check_allowed(steps)
    low = phrase.lower()
    # the phrase must be something to say: at least a verb and something
    if len(low.split()) < 2 and not slots:
        return None
    return {"phrase": low, "slots": slots, "steps": steps}


# ------------------------------------------------------------------------------------------------------------------------------ saying a step
def say_step(st: Dict[str, Any], slots: Optional[List[str]] = None) -> str:
    d = st["do"]

    def v(x):
        return ("the " + x[1:].lower() if x.startswith("$") and x != "$it" else "it" if x == "$it" else f"“{x}”") if isinstance(x, str) else str(x)
    return {"new_tab": lambda: "open a new tab", "open_url": lambda: f"go to {st.get('url', '').replace('https://', '')}",
            "open_app": lambda: f"open {st.get('app')}", "search": lambda: f"search {st.get('engine', 'google')} for {v(st.get('q'))}",
            "type": lambda: f"type {v(st.get('text'))}", "press": lambda: "press " + " + ".join(st.get("keys", [])),
            "click_first_result": lambda: "click the first result", "click_text": lambda: f"click {v(st.get('text'))}",
            "click_at": lambda: f"click at {st.get('x')},{st.get('y')}", "branch": lambda: f"if the page shows “{st.get('text')}”, do {st.get('then')}",
            "wait": lambda: f"wait {st.get('s', 3):g} seconds", "read_title": lambda: "read the title", "read_screen": lambda: "read the screen",
            "stop_if_text": lambda: f"stop if the page shows “{st.get('text')}”", "note": lambda: "keep it as a note", "say": lambda: "tell you it",
            "copy": lambda: "copy it", "remember": lambda: f"remember it as {v(st.get('name'))}",
            "run_skill": lambda: f"do {st.get('text')}"}.get(d, lambda: d)()


# ------------------------------------------------------------------------------------------------------------------------------ hands
class Hands:
    """What a skill's steps reach. A fake in every test; RealHands (below) on the machine."""

    def open_url(self, url: str) -> bool: ...
    def open_app(self, app: str) -> bool: ...
    def keys(self, combo: Tuple[str, ...]) -> bool: ...
    def type_text(self, text: str) -> bool: ...
    def click_text(self, text: str) -> bool: ...
    def click_first_result(self) -> bool: ...
    def click_at(self, x: int, y: int) -> bool: ...
    def title(self) -> str: ...
    def read_screen(self) -> str: ...
    def copy(self, text: str) -> bool: ...
    def note(self, text: str) -> bool: ...
    def remember(self, name: str, value: str) -> bool: ...


class RealHands(Hands):
    def __init__(self, rt: Any):
        self.rt = rt

    def open_url(self, url: str) -> bool:
        d = getattr(self.rt, "desktop", None)
        return bool(d is not None and d.ex.open(url))

    def open_app(self, app: str) -> bool:
        d = getattr(self.rt, "desktop", None)
        if d is None:
            return False
        try:
            return bool(d.ex.open(d.apps.get(app.lower(), app) if isinstance(getattr(d, "apps", None), dict) else app))
        except Exception:
            return False

    def _ex(self):
        c = getattr(self.rt, "control", None)
        return getattr(c, "ex", None)

    def keys(self, combo):
        ex = self._ex()
        return bool(ex is not None and ex.keys(tuple(combo)))

    def type_text(self, text):
        ex = self._ex()
        return bool(ex is not None and ex.type_text(text))

    def click_text(self, text):
        ex = self._ex()
        if ex is None:
            return False
        at = ex.find_text(text)
        return bool(at is not None and ex.click("click", at))

    def click_at(self, x, y):
        ex = self._ex()
        return bool(ex is not None and ex.click("click", (int(x), int(y))))

    def click_first_result(self):
        ex = self._ex()
        if ex is None or not hasattr(ex, "lines"):
            return False
        lines = ex.lines()
        # The first big title under the page's own header: the tallest-ish lines of three words or more, below the top 220 pixels.
        cands = [(y, x, t, h) for (t, (x, y), h) in lines if y > 220 and len(t.split()) >= 3 and h >= 14]
        if not cands:
            return False
        y, x, _t, _h = min(cands)
        return bool(ex.click("click", (x, y)))

    def title(self):
        d = getattr(self.rt, "dwell", None)
        return str(getattr(d, "subject", "") or "") if d is not None else ""

    def read_screen(self):
        ex = self._ex()
        if ex is None or not hasattr(ex, "lines"):
            return ""
        return " ".join(t for t, _p, _h in ex.lines())[:4000]

    def copy(self, text):
        a = getattr(getattr(self.rt, "periphery", None), "actuators", {}).get("clipboard")
        try:
            return bool(a is not None and (a.act(text) or {}).get("ok"))
        except Exception:
            return False

    def note(self, text):
        n = getattr(self.rt, "notes", None)
        try:
            return bool(n is not None and n.keep_note(text))
        except Exception:
            return False

    def remember(self, name, value):
        kb = getattr(getattr(self.rt, "wonder", None), "kb", None)
        try:
            kb.store_term(name, value[:200])
            return True
        except Exception:
            return False


# ------------------------------------------------------------------------------------------------------------------------------ the skills
class Skills:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, pause: Callable[[float], None] = time.sleep):
        self.path = path
        self.clock = clock
        self.pause = pause
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.pending: Optional[Dict[str, Any]] = None        # {"kind": "confirm"|"watch"|"name", ...}
        self.log: List[Dict[str, Any]] = []                  # {"t", "skill", "ok", "step", "why"} for the digest
        self.triggers: List[Dict[str, Any]] = []             # "when I open Word, do X" / "every day at 9, do Y"
        self._lock = threading.RLock()
        self._load()

    # -- keeping ---------------------------------------------------------------------------------------------------
    def _load(self):
        if not self.path:
            return
        try:
            from src.memory.atomic import load_json
            d = load_json(self.path, {}) or {}
            self.skills = dict(d.get("skills") or {})
            self.log = list(d.get("log") or [])[-200:]
            self.triggers = list(d.get("triggers") or [])
        except Exception:
            logger.debug("could not read %s", self.path, exc_info=True)

    def save(self):
        if not self.path:
            return
        try:
            from src.memory.atomic import write_json
            write_json(self.path, {"skills": self.skills, "log": self.log[-200:], "triggers": self.triggers})
        except Exception:
            logger.debug("could not write %s", self.path, exc_info=True)

    # -- teaching ----------------------------------------------------------------------------------------------------
    def teach(self, lesson: Dict[str, Any], source: str = "told", now: Optional[float] = None) -> Tuple[str, Callable[[], None]]:
        now = self.clock() if now is None else now
        name = lesson["phrase"]
        with self._lock:
            prev = self.skills.get(name)
            self.skills[name] = {"name": name, "slots": lesson["slots"], "steps": lesson["steps"], "runs": 0, "fails": 0, "good": 0,
                                 "created": now, "source": source, "aliases": list((prev or {}).get("aliases") or [])}
            self.save()

        def undo(name=name, prev=prev):
            with self._lock:
                if prev is None:
                    self.skills.pop(name, None)
                else:
                    self.skills[name] = prev
                self.save()
        return self.read_back(name, learned=True), undo

    def read_back(self, name: str, learned: bool = False) -> str:
        sk = self.skills[name]
        steps = [f"{i}) {say_step(s, sk['slots'])}" for i, s in enumerate(sk["steps"], 1)]
        head = f"Got it -- {name}: " if learned else f"{name}: "
        s = head + ", ".join(steps) + "."
        if learned:
            s += " I'll walk through it the first couple of times and ask before I start. Tell me if a step is wrong."
        return s

    def listing(self) -> str:
        if not self.skills:
            return "You haven't taught me anything yet -- say “to search google for X means: go to google.com, type X, press enter”."
        rows = []
        for sk in sorted(self.skills.values(), key=lambda s: -s.get("runs", 0)):
            rows.append(f"{sk['name']} ({sk.get('runs', 0)} run{'s' if sk.get('runs', 0) != 1 else ''}"
                        + (f", {sk['fails']} failed" if sk.get("fails") else "") + ")")
        return "I know how to: " + "; ".join(rows[:8]) + "."

    def forget(self, name: str) -> Optional[Tuple[str, Callable[[], None]]]:
        key = self.match_name(name)
        if key is None:
            return None
        old = self.skills.pop(key)
        self.save()

        def undo(key=key, old=old):
            self.skills[key] = old
            self.save()
        return f"Okay -- I've forgotten how to {key}.", undo

    def amend(self, name: str, n: int, clause: str) -> str:
        key = self.match_name(name)
        if key is None:
            return f"I don't know how to {name}."
        sk = self.skills[key]
        if not 1 <= n <= len(sk["steps"]):
            return f"{key} has {len(sk['steps'])} steps."
        try:
            st = parse_step(clause, sk["slots"])
            if st is None:
                return f"I can't do “{clause}” yet."
            check_allowed([st])
        except LessonError as e:
            return str(e)
        sk["steps"][n - 1] = st
        sk["good"] = 0
        self.save()
        return f"Changed step {n} of {key} to: {say_step(st, sk['slots'])}."

    # -- finding what was asked ----------------------------------------------------------------------------------------
    def match_name(self, text: str) -> Optional[str]:
        t = _n(text)
        for k, sk in self.skills.items():
            if _n(k) == t or any(_n(a) == t for a in sk.get("aliases", [])):
                return k
        for k in self.skills:
            base = _n(re.sub(r"\b(?:x|y|z|something|anything)\b", "", k, flags=re.I))
            if base and t.startswith(base):
                return k
        return None

    def trigger(self, text: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """(skill name, {slot: value}) when `text` is one of the skills said aloud."""
        t = re.sub(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|go\s+ahead\s+and)\s+)*", "", (text or "").strip(), flags=re.I).strip(" .!?")
        low = t.lower()
        best: Optional[Tuple[int, str, Dict[str, str]]] = None
        for k, sk in self.skills.items():
            for phrase in [k] + list(sk.get("aliases", [])):
                parts = re.split(r"\b(" + "|".join(re.escape(s) for s in (sk["slots"] or ["\x00"])) + r")\b", phrase, flags=re.I)
                rx = ""
                names = []
                for p in parts:
                    if p.lower() in [s.lower() for s in sk["slots"]]:
                        names.append(next(s for s in sk["slots"] if s.lower() == p.lower()))
                        rx += f"(?P<s{len(names)}>.+?)"
                    else:
                        rx += re.escape(p).replace(r"\ ", r"\s+")
                m = re.fullmatch(rx, low, re.I)
                if m:
                    vals = {names[i]: m.group(f"s{i + 1}").strip() for i in range(len(names))}
                    score = len(phrase)
                    if best is None or score > best[0]:
                        best = (score, k, vals)
        return None if best is None else (best[1], best[2])

    # -- running ------------------------------------------------------------------------------------------------------
    def plan_line(self, name: str, args: Dict[str, str]) -> str:
        sk = self.skills[name]
        steps = [say_step(self._fill(s, args, {}), sk["slots"]) for s in sk["steps"]]
        return f"{name}" + (f" for {', '.join(args.values())}" if args else "") + ": " + ", then ".join(steps) + ". Go ahead?"

    def trusted(self, name: str) -> bool:
        sk = self.skills[name]
        return int(sk.get("good", 0)) >= TRUST_AFTER

    @staticmethod
    def _fill(st: Dict[str, Any], args: Dict[str, str], vars_: Dict[str, str]) -> Dict[str, Any]:
        out = dict(st)
        for k, v in list(out.items()):
            if isinstance(v, str) and v.startswith("$"):
                key = v[1:]
                out[k] = vars_.get("it", "") if key == "it" else next((val for a, val in args.items() if a.lower() == key.lower()), v)
        return out

    def run(self, hands: Hands, name: str, args: Dict[str, str], say: Callable[[str], None], depth: int = 0,
            vars_: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Do it. -> {"ok": bool, "step": index of the step that failed, "why": ..., "vars": ...}. `say` is for what a step wants said."""
        sk = self.skills[name]
        vars_ = {} if vars_ is None else vars_
        if depth > MAX_DEPTH:
            return {"ok": False, "step": 0, "why": "that goes more than three skills deep"}
        # A list for the slot: "search google for each of cats, dogs and birds".
        for a, val in list(args.items()):
            m = re.match(r"^each\s+of\s+(.+)$", val, re.I)
            if m:
                items = [x.strip() for x in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", m.group(1)) if x.strip()][:MAX_EACH]
                last: Dict[str, Any] = {"ok": True, "step": 0, "why": "", "vars": vars_}
                for it in items:
                    last = self.run(hands, name, {**args, a: it}, say, depth, {})
                    if not last["ok"]:
                        break
                    self.pause(1.0)
                return last
        for a, val in args.items():
            if _SECRET.search(val):
                return {"ok": False, "step": 0, "why": "that looks like a secret, and I don't type those"}
        result: Dict[str, Any] = {"ok": True, "step": 0, "why": "", "vars": vars_}
        for i, raw in enumerate(sk["steps"], 1):
            st = self._fill(raw, args, vars_)
            d = st["do"]
            before = hands.title()
            ok, why = True, ""
            if d == "new_tab":
                ok = hands.keys(("ctrl", "t"))
            elif d == "open_url":
                ok = hands.open_url(st["url"])
                if ok:
                    self.pause(2.5)
                    ok, why = self._changed(hands, before, st["url"])
            elif d == "open_app":
                ok = hands.open_app(st["app"])
                self.pause(1.5)
            elif d == "search":
                q = str(st.get("q") or "")
                from urllib.parse import quote_plus
                ok = bool(q) and hands.open_url(ENGINES.get(st.get("engine", "google"), ENGINES["google"]) + quote_plus(q))
                if ok:
                    self.pause(2.5)
                    ok, why = self._changed(hands, before, "")
            elif d == "type":
                t = str(st.get("text") or "")
                if _SECRET.search(t):
                    ok, why = False, "that looks like a secret, and I don't type those"
                else:
                    ok = bool(t) and hands.type_text(t)
            elif d == "press":
                ok = hands.keys(tuple(st["keys"]))
                if ok and st["keys"] == ["enter"]:
                    self.pause(2.0)
            elif d == "click_first_result":
                ok = hands.click_first_result()
                if ok:
                    self.pause(2.5)
                    ok, why = self._changed(hands, before, "")
                else:
                    why = "I couldn't find a first result on the screen"
            elif d == "click_text":
                t = str(st.get("text") or "")
                if _FORBIDDEN_TARGET.search(t):
                    ok, why = False, f"“{t}” is something I never click by myself"
                else:
                    ok = False
                    tried = ([st["alt"]] if st.get("alt") else []) + [t]
                    for cand in tried:
                        ok = hands.click_text(cand)
                        if ok:
                            break
                    if not ok:
                        # Self-repair: other wordings of the same text, then one page down and again -- and what worked is kept for next time.
                        for cand in text_variants(t):
                            if hands.click_text(cand):
                                ok = True
                                raw["alt"] = cand
                                break
                        if not ok and hands.keys(("pagedown",)):
                            self.pause(0.8)
                            ok = hands.click_text(t)
                    why = "" if ok else f"I can't see “{t}” on the screen -- tell me another name for it, or say the step again"
                    if ok:
                        self.pause(1.0)
            elif d == "click_at":
                ok = hands.click_at(int(st.get("x", 0)), int(st.get("y", 0)))
                why = "" if ok else "the click didn't land"
                if ok:
                    self.pause(1.0)
            elif d == "branch":
                if str(st.get("text", "")).lower() in hands.read_screen().lower():
                    found = self.trigger(str(st.get("then") or "")) or ((self.match_name(str(st.get("then") or "")), {}) if self.match_name(str(st.get("then") or "")) else None)
                    if found is None:
                        ok, why = False, f"I don't know how to {st.get('then')}"
                    else:
                        r = self.run(hands, found[0], found[1], say, depth + 1, vars_)
                        ok, why = r["ok"], r.get("why", "")
            elif d == "wait":
                self.pause(float(st.get("s", 3)))
            elif d == "read_title":
                v = hands.title()
                ok, why = bool(v), "" if v else "I couldn't read the title"
                vars_["it"] = v
            elif d == "read_screen":
                v = hands.read_screen()
                ok, why = bool(v), "" if v else "I couldn't read the screen"
                vars_["it"] = v
            elif d == "stop_if_text":
                if str(st.get("text", "")).lower() in hands.read_screen().lower():
                    return {"ok": False, "step": i, "why": f"the page shows “{st['text']}”", "vars": vars_, "stopped": True}
            elif d == "note":
                ok = bool(vars_.get("it")) and hands.note(vars_["it"])
                why = "" if ok else "there was nothing to keep"
            elif d == "say":
                if vars_.get("it"):
                    say(vars_["it"][:200])
                else:
                    ok, why = False, "there was nothing to say"
            elif d == "copy":
                ok = bool(vars_.get("it")) and hands.copy(vars_["it"])
                why = "" if ok else "there was nothing to copy"
            elif d == "remember":
                nm = str(st.get("name") or "")
                ok = bool(nm and vars_.get("it")) and hands.remember(nm, vars_["it"])
                why = "" if ok else "there was nothing to remember"
            elif d == "run_skill":
                t2 = str(st.get("text") or "")
                for sl, val in args.items():
                    t2 = re.sub(r"\b" + re.escape(sl) + r"\b", lambda _m, v=val: v, t2, flags=re.I)
                found = self.trigger(t2)
                if found is None:
                    key = self.match_name(t2)
                    found = (key, {}) if key else None
                if found is None:
                    ok, why = False, f"I don't know how to {t2}"
                else:
                    r = self.run(hands, found[0], found[1], say, depth + 1, vars_)
                    ok, why = r["ok"], r.get("why", "")
            if not ok:
                result = {"ok": False, "step": i, "why": why or "it didn't work", "vars": vars_}
                break
            self.pause(0.3)
        self.record(name, result)
        return result

    @staticmethod
    def _changed(hands: Hands, before: str, url: str) -> Tuple[bool, str]:
        """After a step that should change the screen: the window's title is not the one from before (or names the site). An unreadable title is not a failure."""
        after = hands.title()
        if not after and not before:
            return True, ""
        word = re.sub(r"^https?://(?:www\.)?", "", url).split(".")[0].lower() if url else ""
        if after and (after != before or (word and word in after.lower())):
            return True, ""
        return False, "the window didn't change"

    def record(self, name: str, result: Dict[str, Any]):
        with self._lock:
            sk = self.skills.get(name)
            if sk is None:
                return
            sk["runs"] = int(sk.get("runs", 0)) + 1
            if result["ok"] or result.get("stopped"):
                sk["good"] = int(sk.get("good", 0)) + (1 if result["ok"] else 0)
            else:
                sk["fails"] = int(sk.get("fails", 0)) + 1
                sk["good"] = 0
                sk["last_fail"] = {"step": result["step"], "why": result["why"]}
            self.log.append({"t": self.clock(), "skill": name, "ok": bool(result["ok"]), "step": result.get("step", 0), "why": result.get("why", "")})
            self.save()

    def outcome_line(self, name: str, result: Dict[str, Any], args: Dict[str, str]) -> str:
        sk = self.skills[name]
        if result["ok"]:
            return "Done."
        if result.get("stopped"):
            return f"Stopped: {result['why']}."
        st = sk["steps"][result["step"] - 1] if 0 < result["step"] <= len(sk["steps"]) else {"do": "?"}
        return f"I stopped at step {result['step']} of {name} -- {say_step(st, sk['slots'])}: {result['why']}."

    # -- showing -------------------------------------------------------------------------------------------------------
    def digest_line(self, since: float) -> str:
        rows = [r for r in self.log if r["t"] >= since]
        if not rows:
            return ""
        bad = [r for r in rows if not r["ok"]]
        s = f"{len(rows)} taught skill run{'s' if len(rows) != 1 else ''}"
        if bad:
            worst = max({(r['skill'], r['step']) for r in bad}, key=lambda k: sum(1 for r in bad if (r['skill'], r['step']) == k))
            s += f", {len(bad)} stopped (most often {worst[0]}, step {worst[1]})"
        return s

    def status(self) -> Dict[str, Any]:
        return {"skills": len(self.skills), "runs": sum(int(s.get("runs", 0)) for s in self.skills.values()),
                "failing": [k for k, s in self.skills.items() if s.get("fails", 0) > s.get("good", 0) and s.get("fails", 0) >= 2]}


class Showing:
    """"Watch me" ... "that's how you open my notes": what opened in between. Programs and web addresses only -- never keystrokes, never a window's title."""

    def __init__(self, tabs_before: List[str]):
        self.tabs_before = set(tabs_before)
        self.procs: List[str] = []

    def observe(self, process: str):
        p = (process or "").lower().removesuffix(".exe")
        if p and p not in self.procs and p not in ("python", "pythonw", "ashd", "explorer", "applicationframehost", "searchhost", "shellexperiencehost"):
            self.procs.append(p)

    def steps(self, tabs_now: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        browsers = {"opera", "chrome", "msedge", "firefox", "brave", "vivaldi"}
        seen = set()
        for url in tabs_now:
            if url in self.tabs_before or not url.startswith("http"):
                continue
            dom = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
            if dom and dom not in seen and not dom.startswith(("localhost", "127.")):
                seen.add(dom)
                out.append({"do": "open_url", "url": "https://" + dom})
        apps = [{"do": "open_app", "app": p} for p in self.procs if p not in browsers]
        return (apps + out)[:MAX_STEPS]


# ------------------------------------------------------------------------------------------------------------------------------ repair
def text_variants(t: str) -> List[str]:
    """Other ways the same screen text may be written: without a trailing 's' or punctuation, without its first or last word, lower case."""
    base = re.sub(r"[^\w ]+", "", t).strip()
    out: List[str] = []
    words = base.split()
    for v in (base, base.rstrip("s") if len(base) > 4 else "", " ".join(words[1:]) if len(words) > 2 else "", " ".join(words[:-1]) if len(words) > 2 else "",
              base.lower(), base.title()):
        if v and v != t and v not in out:
            out.append(v)
    return out[:5]


# ------------------------------------------------------------------------------------------------------------------------------ showing, with keys
_CHARS = {0x20: " ", 0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/", 0xDB: "[", 0xDD: "]", 0xDE: "'"}
_SHIFT_DIGITS = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")"}
_SHIFT_SYM = {";": ":", "=": "+", ",": "<", "-": "_", ".": ">", "/": "?", "'": '"'}


class KeyRecorder:
    """What was typed and pressed while 'watch me' was on. `poll()` returns (set of virtual-key codes held now, (left button down, x, y)); real on Windows
    (GetAsyncKeyState, nothing installed), a fake in tests. A new press is an edge: a held key is not repeated."""

    def __init__(self, poll: Optional[Callable[[], Tuple[set, Tuple[bool, int, int]]]] = None):
        self.poll = poll or _win_poll
        self.events: List[Tuple[str, Any]] = []
        self._down: set = set()
        self._mouse = False

    def tick(self) -> None:
        try:
            keys, (lb, x, y) = self.poll()
        except Exception:
            return
        shift, ctrl = 0x10 in keys, 0x11 in keys
        for vk in sorted(keys - self._down):
            if vk in (0x10, 0x11, 0x12, 0x5B) or 0xA0 <= vk <= 0xA5 or vk in (1, 2):
                continue
            if ctrl:
                if 0x41 <= vk <= 0x5A:
                    self.events.append(("combo", ("ctrl", chr(vk).lower())))
                continue
            if vk == 0x0D:
                self.events.append(("key", "enter"))
            elif vk == 0x08:
                self.events.append(("key", "backspace"))
            elif vk == 0x09:
                self.events.append(("key", "tab"))
            elif vk == 0x1B:
                self.events.append(("key", "esc"))
            elif 0x41 <= vk <= 0x5A:
                c = chr(vk).lower()
                self.events.append(("char", c.upper() if shift else c))
            elif 0x30 <= vk <= 0x39:
                c = chr(vk)
                self.events.append(("char", _SHIFT_DIGITS[c] if shift else c))
            elif vk in _CHARS:
                c = _CHARS[vk]
                self.events.append(("char", _SHIFT_SYM.get(c, c) if shift else c))
        self._down = set(keys)
        if lb and not self._mouse:
            self.events.append(("click", (x, y)))
        self._mouse = lb

    def steps(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        buf: List[str] = []

        def flush():
            if buf:
                out.append({"do": "type", "text": "".join(buf)})
                buf.clear()
        for kind, val in self.events:
            if kind == "char":
                buf.append(val)
            elif kind == "key" and val == "backspace":
                if buf:
                    buf.pop()
            elif kind == "key":
                flush()
                out.append({"do": "press", "keys": [val]})
            elif kind == "combo":
                flush()
                out.append({"do": "press", "keys": list(val)})
            elif kind == "click":
                flush()
                out.append({"do": "click_at", "x": int(val[0]), "y": int(val[1])})
        flush()
        return [s for s in out if not (s["do"] == "type" and not s["text"].strip())][:MAX_STEPS]


def _win_poll() -> Tuple[set, Tuple[bool, int, int]]:
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    held = {vk for vk in range(8, 256) if u.GetAsyncKeyState(vk) & 0x8000}
    pt = wintypes.POINT()
    u.GetCursorPos(ctypes.byref(pt))
    return held, (bool(u.GetAsyncKeyState(1) & 0x8000), int(pt.x), int(pt.y))


def merge_demos(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], List[str]]]:
    """Two demonstrations of the same thing: when they match step for step and only typed text (or a searched phrase) differs, that text is a slot. -> (steps, slots)."""
    if len(a) != len(b) or any(x["do"] != y["do"] for x, y in zip(a, b)):
        return None
    out, slots = [], []
    for x, y in zip(a, b):
        s = dict(x)
        for key in ("text", "q"):
            if x["do"] in ("type", "search") and x.get(key) != y.get(key):
                if "X" not in slots:
                    slots.append("X")
                s[key] = "$X"
        out.append(s)
    return (out, slots) if slots else None


# ------------------------------------------------------------------------------------------------------------------------------ triggers
_WHEN_OPEN = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?when\s+i\s+(?:open|start|launch)\s+(?P<app>[\w .+-]{2,25}?)\s*,?\s*(?:then\s+)?(?:do|run)\s+(?P<n>[a-z][a-z0-9 ]{2,60}?)\W*$", re.I)
_EVERY_DAY = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:every\s+(?:day|morning|weekday)|each\s+morning)\s+at\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?\s*,?\s*(?:then\s+)?(?:do|run)\s+(?P<n>[a-z][a-z0-9 ]{2,60}?)\W*$", re.I)


def parse_trigger(text: str) -> Optional[Dict[str, Any]]:
    m = _WHEN_OPEN.match((text or "").strip())
    if m:
        return {"kind": "open", "app": m.group("app").strip().lower().removesuffix(".exe"), "name": m.group("n").strip().lower()}
    m = _EVERY_DAY.match((text or "").strip())
    if m:
        h, mi = int(m.group("h")), int(m.group("m") or 0)
        ap = (m.group("ap") or "").lower().replace(".", "")
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        if h > 23 or mi > 59:
            return None
        return {"kind": "daily", "at": f"{h:02d}:{mi:02d}", "name": m.group("n").strip().lower()}
    return None


def due_triggers(triggers: List[Dict[str, Any]], proc: str, prev_proc: str, now: float) -> List[Dict[str, Any]]:
    """The triggers that fire now: an app that just came to the front, or a daily time that has passed today and not yet been run."""
    out = []
    p = (proc or "").lower().removesuffix(".exe")
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    hm = time.strftime("%H:%M", time.localtime(now))
    for t in triggers:
        if t["kind"] == "open" and p and p != (prev_proc or "").lower().removesuffix(".exe") and (t["app"] in p or p in t["app"]):
            if t.get("last_at", 0) < now - 600:
                out.append(t)
        elif t["kind"] == "daily" and hm >= t["at"] and t.get("last") != today:
            out.append(t)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ packs
def export_pack(skills: Dict[str, Dict[str, Any]], names: Optional[List[str]] = None) -> Dict[str, Any]:
    keep = {k: {kk: vv for kk, vv in v.items() if kk in ("name", "slots", "steps", "aliases")} for k, v in skills.items() if not names or k in names}
    return {"ash_skill_pack": 1, "skills": keep}


def import_pack(skills: Dict[str, Dict[str, Any]], data: Dict[str, Any], now: float) -> Tuple[List[str], List[str]]:
    """(added, refused). Every step is checked again, as if it were being taught now: a pack cannot smuggle in what a lesson may not contain."""
    added, refused = [], []
    for name, sk in (data.get("skills") or {}).items():
        try:
            steps = list(sk.get("steps") or [])
            if not steps or len(steps) > MAX_STEPS or any("do" not in s for s in steps):
                raise LessonError("bad steps")
            check_allowed(steps)
            skills[name] = {"name": name, "slots": list(sk.get("slots") or []), "steps": steps, "runs": 0, "fails": 0, "good": 0, "created": now,
                            "source": "imported", "aliases": list(sk.get("aliases") or [])}
            added.append(name)
        except Exception:
            refused.append(name)
    return added, refused
