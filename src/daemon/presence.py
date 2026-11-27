"""
src/daemon/presence.py

The small things that make ASH feel like it lives here, all offline:

    OpenApps      which programs you had open, kept every couple of minutes; after a shutdown that was not clean: "You had Opera, Word and Notepad open -- reopen them?"
    updated_line  after ASH itself changed (a new commit since it last ran): "I was updated: ..." from the real commit subjects, said once
    smoke         "run a self test" -- a read-only check that each thing ASH relies on works on THIS machine (the screen reader, the volume, Bluetooth, the notification
                  list, git, memory, the language model), each as ok / not ok with why; "works in the tests" is not "works here"
    banter        "tell me something interesting", "would you rather", "word of the day", "a tongue twister" -- from lists kept in the file, no model, no internet
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|for\s+me))*\s*[?.!]*\s*$"


# ----------------------------------------------------------------------------------------------------------------------------- open programs
_SKIP = {"python", "pythonw", "ashd", "explorer", "applicationframehost", "searchhost", "shellexperiencehost", "textinputhost", "lockapp", "systemsettings",
         "windowsterminal", "cmd", "powershell", "conhost", "taskmgr", "startmenuexperiencehost", "msedgewebview2"}


class OpenApps:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time, keep: int = 8):
        self.path, self.clock, self.keep = path, clock, keep
        self.seen: List[str] = []
        self._saved = 0.0

    def see(self, process: str) -> None:
        p = (process or "").lower().removesuffix(".exe")
        if not p or p in _SKIP:
            return
        if p in self.seen:
            self.seen.remove(p)
        self.seen.append(p)                      # most recent last
        self.seen = self.seen[-self.keep:]

    def save(self, now: Optional[float] = None, force: bool = False) -> None:
        now = self.clock() if now is None else now
        if not self.path or (not force and now - self._saved < 120.0) or not self.seen:
            return
        self._saved = now
        try:
            from src.memory.atomic import write_json
            write_json(self.path, {"apps": self.seen, "t": now})
        except Exception:
            pass

    def before_crash(self, max_age_s: float = 86400.0, now: Optional[float] = None) -> List[str]:
        now = self.clock() if now is None else now
        try:
            from src.memory.atomic import load_json
            d = load_json(self.path, {}) or {}
        except Exception:
            return []
        if now - float(d.get("t", 0)) > max_age_s:
            return []
        return list(d.get("apps") or [])


def recovery_line(apps: List[str], nice: Callable[[str], str] = lambda a: a.title()) -> str:
    names = [nice(a) for a in apps][-5:]
    if not names:
        return ""
    lst = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"It looks like the computer shut down unexpectedly. You had {lst} open -- want me to reopen {'it' if len(names) == 1 else 'them'}?"


# ----------------------------------------------------------------------------------------------------------------------------- updated
def updated_line(old_head: str, new_head: str, run: Optional[Callable[[List[str]], str]] = None, cwd: Optional[str] = None) -> str:
    """'I was updated: X; Y; Z.' from the commit subjects between the two heads (read-only git), or '' when nothing changed or git can't say."""
    if not new_head or old_head == new_head:
        return ""
    run = run or (lambda cmd: subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout)
    try:
        rng = f"{old_head}..{new_head}" if old_head else f"-3"
        out = run(["git", "log", "--no-merges", "--format=%s"] + ([rng] if old_head else ["-3"]))
    except Exception:
        return ""
    subs = [x.strip() for x in (out or "").splitlines() if x.strip()]
    if not subs:
        return ""
    short = []
    for s in subs[:3]:
        head = re.split(r"[:;]| -- ", s, maxsplit=1)[0].strip()
        short.append((head if 12 <= len(head) <= 90 else s[:90]).rstrip(".,"))
    more = len(subs) - 3
    return "I was updated: " + "; ".join(short) + (f" (and {more} more)" if more > 0 else "") + "."


def git_head(cwd: Optional[str] = None, run: Optional[Callable[[List[str]], str]] = None) -> str:
    run = run or (lambda cmd: subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout)
    try:
        return (run(["git", "rev-parse", "HEAD"]) or "").strip()[:40]
    except Exception:
        return ""


# ----------------------------------------------------------------------------------------------------------------------------- smoke test
_SMOKE = re.compile(_A + r"(?:run\s+(?:a\s+|the\s+)?(?:smoke\s+test|self[\s-]?test)|(?:smoke|self)[\s-]?test|check\s+that\s+everything\s+works(?:\s+(?:here|on\s+this\s+(?:pc|computer)))?|test\s+yourself)" + _Z, re.I)


def asks_smoke(text: str) -> bool:
    return bool(_SMOKE.match((text or "").strip()))


def smoke(checks: Dict[str, Callable[[], Tuple[bool, str]]]) -> List[Tuple[str, bool, str]]:
    out = []
    for name, fn in checks.items():
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {str(e)[:60]}"
        out.append((name, bool(ok), detail))
    return out


def smoke_say(rows: List[Tuple[str, bool, str]]) -> str:
    bad = [r for r in rows if not r[1]]
    good = [r[0] for r in rows if r[1]]
    s = f"{len(good)} of {len(rows)} work here" + (": " + ", ".join(good) + "." if good else ".")
    if bad:
        s += " Not working: " + "; ".join(f"{n} ({d})" for n, _ok, d in bad) + "."
    return s


def default_checks(rt: Any) -> Dict[str, Callable[[], Tuple[bool, str]]]:
    """Read-only: nothing is typed, clicked, opened, sent or changed."""
    def memory():
        import psutil
        return True, f"{psutil.virtual_memory().percent:.0f}% used"

    def git():
        h = git_head(os.getcwd())
        return bool(h), h[:8] or "no git here"

    def volume():
        from . import devicehelp
        v = devicehelp._core_get()
        return v is not None, "" if v is None else f"{v}%"

    def bluetooth():
        from . import devicehelp
        rows = devicehelp.bluetooth_devices()
        return True, f"{len(rows)} paired"

    def notifications():
        from . import toasts
        p = toasts.default_path()
        if not os.path.exists(p):
            return False, "no notification database"
        toasts.read_db(p, 2 ** 62)
        return True, "readable"

    def screen():
        from src.senses.screenread import ScreenReader
        r = ScreenReader()
        return r.backend == "rapidocr", f"backend {r.backend}"

    def model():
        llm = getattr(getattr(rt, "ash", None), "llm", None)
        name = str(getattr(llm, "_ollama_model", "") or "")
        if not name:
            return False, "no model set"
        import urllib.request
        url = re.sub(r"/api/(?:chat|generate)\s*$", "/api/tags", str(getattr(llm, "_ollama_url", "") or "http://127.0.0.1:11434/api/tags"))
        try:
            urllib.request.urlopen(url, timeout=2).read(10)
            return True, name
        except Exception:
            return False, f"{name} (Ollama not answering)"

    def mic():
        s = (getattr(getattr(rt, "periphery", None), "sensors", {}) or {}).get("microphone")
        return s is not None and getattr(getattr(s, "state", None), "value", "") == "active", "on" if s is not None else "no microphone sensor"
    return {"memory": memory, "git": git, "volume": volume, "Bluetooth": bluetooth, "notifications": notifications, "screen reader": screen, "language model": model, "microphone": mic}


# ----------------------------------------------------------------------------------------------------------------------------- banter
FACTS = [
    "Honey never spoils: jars found in Egyptian tombs were still good after three thousand years.",
    "Octopuses have three hearts, and two of them stop when they swim.",
    "A day on Venus is longer than its year.",
    "Bananas are slightly radioactive, thanks to the potassium in them.",
    "There are more possible chess games than atoms in the observable universe.",
    "The first computer bug was a real moth, found in a relay of the Harvard Mark II in 1947.",
    "Sharks are older than trees: they've been around for over 400 million years.",
    "Your brain uses about twenty percent of your body's energy, though it is two percent of its weight.",
    "The Eiffel Tower grows about fifteen centimetres taller in summer, because the iron expands.",
    "A group of crows is called a murder, and they can remember human faces for years.",
    "Wombat droppings are cube-shaped.",
    "The word “robot” comes from the Czech word for forced labour, from a 1920 play.",
    "Light from the Sun takes about eight minutes to reach Earth.",
    "The shortest war in history, between Britain and Zanzibar in 1896, lasted about forty minutes.",
    "Cleopatra lived closer in time to the Moon landing than to the building of the Great Pyramid.",
    "A single bolt of lightning is about five times hotter than the surface of the Sun.",
    "Ada Lovelace wrote what is often called the first computer program, for a machine that was never built.",
    "Sloths can hold their breath for up to forty minutes by slowing their heart rate.",
    "There are more trees on Earth than stars in the Milky Way, by most estimates.",
    "The first webcam was pointed at a coffee pot at Cambridge, so researchers could see if it was full.",
    "Dolphins call each other by name, using signature whistles.",
    "The human nose can tell apart over a trillion different smells, according to one study.",
    "A teaspoon of a neutron star would weigh about a billion tonnes.",
    "Neurons in your gut outnumber those in a cat's brain.",
    "The first email was sent in 1971, and the sender doesn't remember what it said.",
    "Water can boil and freeze at the same time, at the triple point.",
    "Tardigrades can survive the vacuum of space.",
    "The Great Wall is not visible from the Moon with the naked eye, despite the myth.",
    "Your fingerprints formed before you were born and almost never change.",
    "Python was named after Monty Python, not the snake.",
]
WYR = [
    "Would you rather always know the time, or always know the weather?",
    "Would you rather have a pause button or a rewind button for your life?",
    "Would you rather lose all your files, or all your photos?",
    "Would you rather code with no autocomplete, or write emails with no spellcheck?",
    "Would you rather have a perfect memory, or never feel tired?",
    "Would you rather talk to animals, or understand every language?",
    "Would you rather have unlimited battery, or unlimited storage?",
    "Would you rather live a year in the past, or a year in the future?",
    "Would you rather give up music, or give up films?",
    "Would you rather read minds once a day, or fly for ten minutes?",
    "Would you rather always be ten minutes early, or always be ten minutes late?",
    "Would you rather your computer never crash, or never be slow?",
]
WORDS = [
    ("ephemeral", "lasting for a very short time"), ("serendipity", "finding something good by chance"), ("ubiquitous", "present everywhere at once"),
    ("sonder", "realising everyone else's life is as full as yours"), ("petrichor", "the smell of rain on dry earth"), ("liminal", "on the threshold between two states"),
    ("pragmatic", "dealing with things sensibly and practically"), ("laconic", "using very few words"), ("mellifluous", "sweet and smooth to hear"),
    ("sagacious", "wise, with good judgement"), ("halcyon", "calm and peaceful, often looking back"), ("ineffable", "too great to be put into words"),
    ("quixotic", "idealistic to the point of being impractical"), ("verisimilitude", "the appearance of being true"), ("syzygy", "three bodies lined up, like sun, moon and earth"),
    ("obfuscate", "to make something unclear on purpose"), ("perspicacious", "quick to notice and understand"), ("equanimity", "calm in a difficult moment"),
    ("zeitgeist", "the spirit of a particular time"), ("lucid", "clear and easy to understand"), ("diaphanous", "light and see-through"),
    ("stoic", "enduring hardship without complaint"), ("tenacious", "holding on firmly, not giving up"), ("palimpsest", "something reused while still showing traces of what was before"),
    ("eloquent", "fluent and persuasive in speech"), ("idiosyncratic", "peculiar to one person"), ("nascent", "just beginning to exist"), ("cogent", "clear, logical and convincing"),
    ("ennui", "a feeling of weary boredom"), ("panacea", "a cure for everything"),
]
TWISTERS = [
    "She sells seashells by the seashore.", "Peter Piper picked a peck of pickled peppers.", "How can a clam cram in a clean cream can?",
    "Red lorry, yellow lorry.", "Unique New York, you know you need unique New York.", "Betty Botter bought some butter, but she said the butter's bitter.",
    "Six slippery snails slid slowly seaward.", "The sixth sick sheikh's sixth sheep's sick.",
]
_FACT = re.compile(_A + r"(?:tell\s+me\s+(?:something|a\s+(?:fun\s+)?fact)(?:\s+(?:interesting|cool|fun|new))?|(?:a\s+)?fun\s+fact|(?:give\s+me\s+)?(?:an?\s+)?(?:interesting|random)\s+fact|teach\s+me\s+something)" + _Z, re.I)
_WYRQ = re.compile(_A + r"(?:would\s+you\s+rather|give\s+me\s+a\s+would\s+you\s+rather|play\s+would\s+you\s+rather)" + _Z, re.I)
_WORD = re.compile(_A + r"(?:(?:what(?:'s|\s+is)\s+)?(?:the\s+)?word\s+of\s+the\s+day|teach\s+me\s+a\s+(?:new\s+)?word|give\s+me\s+a\s+(?:new\s+)?word)" + _Z, re.I)
_TWIST = re.compile(_A + r"(?:(?:give\s+me\s+|say\s+|tell\s+me\s+)?(?:a\s+)?tongue\s+twister|test\s+my\s+pronunciation)" + _Z, re.I)


class Banter:
    def __init__(self, rng: Optional[random.Random] = None, clock: Callable[[], float] = time.time):
        self.rng, self.clock = rng or random.Random(), clock
        self._told: Dict[str, List[int]] = {"fact": [], "wyr": [], "twist": []}

    def _pick(self, kind: str, pool: List[Any]) -> Any:
        used = self._told[kind]
        free = [i for i in range(len(pool)) if i not in used] or list(range(len(pool)))
        if len(free) == len(pool):
            used.clear()
        i = self.rng.choice(free)
        used.append(i)
        return pool[i]

    def hear(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if _FACT.match(t):
            return self._pick("fact", FACTS)
        if _WYRQ.match(t):
            return self._pick("wyr", WYR)
        if _WORD.match(t):
            day = int(self.clock() // 86400)
            w, d = WORDS[day % len(WORDS)]
            return f"Today's word is “{w}”: {d}."
        if _TWIST.match(t):
            return self._pick("twist", TWISTERS) + " Try it three times fast."
        return None
