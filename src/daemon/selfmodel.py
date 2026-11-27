"""
src/daemon/selfmodel.py

What ASH knows about ITSELF, read from the live runtime -- never from prose and never from the language model.

25 Nov: "What model are you using?" -> "You're not using any specific model here ... a general-purpose model that doesn't track which one I'm
using" (it knows: it chose the model at start). "What hardware are you using?" -> "That's what I just said." "What's new in the update?" -> a stale
answer about the user's frustration. "What do you know about yourself?" -> "You haven't told me anything about yourself Ash." Each of these went
to a 3B model that has no facts about the machine it is running on.

`parse(text)` -> kind or None;  `say(kind, rt)` -> the answer;  `block(rt)` -> a few lines for the model's prompt, so it cannot deny what is true.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|but|now|please)[,\s]+)*"
_Z = r"(?:[,\s]+(?:ash|please|now|right\s+now))*\s*[?.!]*\s*$"

_RULES = [
    ("model", r"(?:what|which)\s+(?:(?:language\s+)?model|llm|brain|ai)\s+(?:are\s+you|do\s+you)\s+(?:using|running|use|on|have)"
              r"|what\s+(?:are\s+you|is\s+your)\s+(?:running\s+on|powered\s+by|built\s+on)(?:\s+model)?"
              r"|what(?:'s|\s+is)\s+(?:your|the)\s+(?:language\s+)?model|which\s+model\s+(?:is\s+this|answers)"
              # 26 Nov 20:44: "What model are you?" went to the model (6 s); the next line, "...using?", was answered by rule.
              r"|(?:what|which)\s+(?:(?:language\s+)?model|llm)\s+(?:are\s+you|is\s+(?:this|that|it))(?:\s+(?:now|today|currently))?"
              r"|what\s+(?:model|llm)\s+(?:do\s+you\s+run|runs\s+you|are\s+you\s+on)"),
    ("hardware", r"what\s+(?:hardware|specs?|machine|computer|pc|laptop|processor|cpu|gpu|graphics\s+card|ram|memory)\s+(?:are\s+you|do\s+you|is\s+(?:this|it))"
                 r"\s*(?:using|running|on|have|got)?(?:\s+on)?"
                 r"|what\s+can\s+you\s+tell\s+me\s+about\s+(?:this|the|your)\s+(?:machine|computer|pc|laptop|hardware|system)"
                 r"|(?:tell\s+me\s+(?:about|the)|what\s+are)\s+(?:this|the|your)\s+(?:machine|computer|pc|laptop|hardware|system)(?:'s)?\s*(?:specs?|specifications)?"
                 r"|what(?:'s|\s+is)\s+(?:your|the)\s+(?:hardware|specs?|processor|cpu|gpu|graphics\s+card)"
                 r"|what\s+(?:are\s+)?(?:your|the)\s+specs"),
    ("whatsnew", r"what(?:'s|\s+is|\s+was)\s+(?:new|changed)(?:\s+with\s+you)?(?:\s+(?:in\s+(?:the\s+)?(?:new\s+)?(?:update|version|release)|since\s+(?:last|yesterday)|lately|recently))?"
                 r"|what\s+(?:has\s+)?changed(?:\s+(?:in|with)\s+(?:you|the\s+(?:update|version)))?"
                 r"|what(?:'s|\s+is|\s+was)\s+(?:recently\s+)?(?:added|updated)\s+(?:to|in)\s+(?:your\s+)?(?:memory|code|brain|system)?"
                 r"|what\s+did\s+(?:you\s+)?(?:they\s+)?(?:add|update|change)(?:\s+in\s+(?:the\s+)?update)?|what(?:'s|\s+is)\s+in\s+the\s+(?:new\s+)?update"
                 r"|(?:the\s+)?(?:release\s+notes|changelog|change\s+log)"
                 # 25 Nov 14:xx: "What was the last update?" / "what was added last" matched nothing and went to the model.
                 r"|what(?:'s|\s+is|\s+was)\s+(?:the\s+)?(?:last|latest|most\s+recent|newest)\s+(?:update|change|feature|thing\s+(?:added|you\s+got))"
                 r"|what\s+(?:was|did\s+(?:you|they)\s+get)\s+(?:added|updated|changed)\s+(?:last|most\s+recently|latest)"
                 r"|what\s+(?:is|was)\s+(?:added|new)\s+(?:last|most\s+recently)"
                 r"|when\s+were\s+you\s+(?:last\s+)?updated"
                 # 27 Nov: "what the latest update", "what's your latest update", "what's the latest update for" went to the model three times (50 s each).
                 r"|what(?:'s|\s+is|\s+was)?\s+(?:the\s+|your\s+)?(?:latest|newest|last|most\s+recent)\s+update(?:\s+(?:for|on|to)(?:\s+(?:you|ash|yourself))?)?"),
    ("self", r"what\s+do\s+you\s+know\s+about\s+yourself|describe\s+yourself|what\s+do\s+you\s+know\s+about\s+(?:you|ash)"
             r"|what\s+(?:all\s+)?do\s+you\s+know\s+(?:of|about)\s+(?:yourself|your\s+own\s+(?:self|system))"),   # ("who are you" / "tell me about yourself": honest.py)
    ("size", r"how\s+(?:much|many\s+things)\s+do\s+you\s+(?:remember|know)|how\s+big\s+is\s+your\s+memory|how\s+much\s+(?:is\s+)?(?:in\s+)?your\s+memory"
             r"|what\s+(?:have\s+you\s+)?learn(?:ed|t)\s+(?:today|lately|recently|so\s+far)|what\s+did\s+you\s+learn\s+(?:today|lately|recently)"),
]
_COMPILED = [(k, re.compile(_A + r"(?:" + p + r")" + _Z, re.I)) for k, p in _RULES]


# "How do you use your emotions?", "how do you remember", "what happens when you sleep", "what is repair mode": 25 Nov these went to the model, which
# had nothing to go on. Each is a short, true description of the mechanism, said as it stands.
_HOW = {
    "emotions": (r"(?:emotions?|feelings?|moods?)", "I keep one emotional state: about two dozen feelings on a scale of 0 to 10 that move with what happens -- thanks and "
                 "success lift them, cancelled answers, slowness and being told off lower them. That same state sets my tone and how long I think, "
                 "and it shows on my face. Separately, I note how you sounded on each thing you say, from your words only, and keep it with that memory. "
                 "Ask for my emotional telemetry and I'll read the numbers."),
    "memory": (r"(?:memory|memories|remember(?:ing)?)", "Everything you say becomes a remembered moment, stored with the time, what it was about and how you sounded. "
               "People, places and things you mention are kept as entries, and the ideas in them are linked into a web of concepts. When you ask, I look "
               "for moments that share your question's concepts, then answer from those. Anything you ask me to keep goes in your notes."),
    "sleep": (r"(?:sleep|sleeping|nap|napping|dream\w*)", "When it's quiet I take a short nap, and at night a longer one. I replay recent moments, strengthen links between "
              "ideas that keep coming up together, let weak ones fade, and check cause-and-effect links I'd guessed. Nothing is sent anywhere."),
    "concepts": (r"(?:concepts?|ideas?)", "A concept is an idea that keeps coming up -- a word or topic, like NoSQL or a person's name. Concepts are linked "
                 "when they show up together, and the links grow with use and fade without it. They are how I find related memories when the words differ."),
    "repair": (r"repair\s+mode", "When you sound frustrated -- \"bruh\", \"never mind\", \"that's wrong\" a few times in a few minutes -- I apologise once, stop "
               "the thinking fillers, answer in a sentence, and say \"I don't know\" instead of guessing. A thanks, or ten quiet minutes, ends it. "
               "The lines that led to it are kept in a failures file."),
    "privacy": (r"(?:privacy|private|data|information)", "Hearing, memory and the language model all run on this computer. The only things that leave it are what you "
                "ask for: a web search opens in your own browser, and the weather sends your town to the forecast service. Say \"go private\" and I stop "
                "listening and looking."),
}
_HOWQ = re.compile(_A + r"(?:how\s+(?:do|does|can)\s+(?:you|ash|your)\s+(?:use|work\s+with|handle|have|make\s+use\s+of)?\s*(?:your\s+|the\s+|my\s+)?(?P<a>[a-z ]{3,30}?)(?:\s+work)?"
                   r"|why\s+do\s+you\s+(?:use|have|need)\s+(?:your\s+)?(?P<b>[a-z ]{3,30}?)"
                   r"|what\s+(?:happens|do\s+you\s+do)\s+when\s+you\s+(?P<c>sleep|nap|dream)"
                   r"|what(?:'s|\s+is)\s+(?:a\s+|an\s+|the\s+)?(?P<d>concepts?|repair\s+mode|sleep|napping)(?:\s+in\s+your\s+(?:memory|brain))?"
                   r"|do\s+you\s+(?P<e>sleep|dream|nap)|how\s+(?:is|are)\s+my\s+(?P<f>privacy|data|information)\s+(?:handled|kept|protected))" + _Z, re.I)


def _how_topic(t: str) -> Optional[str]:
    m = _HOWQ.match(t)
    if not m:
        return None
    what = " ".join(x for x in m.groupdict().values() if x).strip().lower()
    for key, (pat, _) in _HOW.items():
        if re.fullmatch(pat, what, re.I) or (key == "repair" and "repair" in what):
            return "how:" + key
    if what.startswith(("sleep", "nap", "dream")):
        return "how:sleep"
    return None


def parse(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 90:
        return None
    h = _how_topic(t)
    if h:
        return h
    for kind, rx in _COMPILED:
        if rx.match(t):
            return kind
    return None


# ---- the machine ---------------------------------------------------------------------------------------------------
_CACHE: Dict[str, Any] = {}


def _cpu_name() -> str:
    if "cpu" in _CACHE:
        return _CACHE["cpu"]
    name = ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0])
    except Exception:
        import platform
        name = platform.processor() or platform.machine()
    name = re.sub(r"\s+", " ", re.sub(r"\((?:R|TM|tm|r)\)|CPU|Processor|@.*$", "", name)).strip()
    _CACHE["cpu"] = name
    return name


def _gpu() -> Dict[str, Any]:
    if "gpu" in _CACHE:
        return _CACHE["gpu"]
    out: Dict[str, Any] = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        if r.returncode == 0 and r.stdout.strip():
            name, _, mem = r.stdout.strip().splitlines()[0].partition(",")
            out = {"name": name.strip(), "gb": round(float(mem) / 1024.0, 1) if mem.strip().isdigit() else None}
    except Exception:
        out = {}
    _CACHE["gpu"] = out
    return out


def _git_subjects(n: int = 4) -> List[str]:
    key = ("git", n)
    if key in _CACHE and time.time() - _CACHE[key][0] < 600:
        return _CACHE[key][1]
    subs: List[str] = []
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        r = subprocess.run(["git", "-C", root, "log", f"-{n * 3}", "--format=%s"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            head = re.split(r"\s+--\s+", line.strip(), maxsplit=1)[0]
            if len(head) > 100:
                head = head[:100].rsplit(" ", 1)[0].rstrip(" ,;:'\"/") + "..."
            head = head.rstrip(" ,;")
            if head and head not in subs and not line.lower().startswith(("merge", "tests:", "test:", "tst_", "documentation", "fix: a microphone sensor built")):
                subs.append(head)
            if len(subs) >= n:
                break
    except Exception:
        subs = []
    _CACHE[key] = (time.time(), subs)
    return subs


def facts(rt) -> Dict[str, Any]:
    """Everything the answers use, read now."""
    f: Dict[str, Any] = {}
    llm = getattr(getattr(rt, "ash", None), "llm", None)
    f["model"] = str(getattr(llm, "_ollama_model", "") or "")
    f["light"] = str(getattr(llm, "_light_model", "") or "")
    f["mode"] = str(getattr(llm, "_mode", "") or "")
    f["notices"] = list(getattr(llm, "_said_notices", []) or [])
    f["cpu"] = _cpu_name()
    f["threads"] = os.cpu_count() or 0
    f["gpu"] = _gpu()
    try:
        import psutil
        vm = psutil.virtual_memory()
        f["ram_gb"] = round(vm.total / 1e9)
        f["ram_pct"] = int(vm.percent)
        f["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 1)
    except Exception:
        pass
    f["uptime_s"] = max(0.0, time.time() - float(getattr(rt, "_started_at", 0.0) or getattr(rt, "start_time", 0.0) or time.time()))
    ash = getattr(rt, "ash", None)
    try:
        f["episodes"] = len(getattr(getattr(getattr(ash, "episodic_memory", None), "store", None), "episodes", {}) or {})
        f["entities"] = len(getattr(getattr(getattr(ash, "entity_memory", None), "store", None), "entities", {}) or {})
        f["concepts"] = len(rt.brain.concepts.net.concepts) if getattr(rt.brain, "concepts", None) is not None else 0
        kb = getattr(getattr(rt, "wonder", None), "kb", None)
        f["people"] = len(getattr(kb, "people", {}) or {}) if kb is not None else 0
    except Exception:
        pass
    return f


def _span(sec: float) -> str:
    m = int(sec // 60)
    return f"{m} minute{'s' if m != 1 else ''}" if m < 90 else f"{m // 60} hours {m % 60} minutes"


def say(kind: str, rt) -> str:
    f = facts(rt)
    if kind.startswith("how:"):
        return _HOW[kind[4:]][1]
    if kind == "model":
        if not f["model"]:
            return "I have no language model loaded right now -- the rules answer what they can."
        s = f"I answer with {f['model']}, through Ollama, on this machine."
        if f["light"]:
            s += f" Small talk goes to {f['light']}."
        s += " Facts, timers, reminders and commands are done by rules and tools, not by the model."
        return s
    if kind == "hardware":
        g = f.get("gpu") or {}
        bits = [f"{f['cpu']} with {f['threads']} threads" if f.get("cpu") else f"{f['threads']} threads"]
        if f.get("ram_gb"):
            bits.append(f"{f['ram_gb']} GB of memory ({f.get('ram_pct', '?')}% in use)")
        if g.get("name"):
            bits.append(f"a {g['name']}" + (f" with {g['gb']} GB" if g.get("gb") else ""))
        s = "This machine has " + ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1] + "."
        if f.get("rss_gb"):
            s += f" I'm using about {f['rss_gb']} GB of that."
        return s
    if kind == "whatsnew":
        subs = _git_subjects(4)
        if not subs:
            return "I can't see my own update history from here."
        return "Lately: " + "; ".join(s[:1].lower() + s[1:] if i else s for i, s in enumerate(subs)) + "."
    if kind == "size":
        return (f"I hold {f.get('episodes', 0)} remembered moments, {f.get('entities', 0)} things and places, {f.get('people', 0)} people you told me "
                f"about and {f.get('concepts', 0)} concepts. I've been up {_span(f['uptime_s'])}.")
    # self
    from . import honest
    s = ("I'm A.S.H, an assistant that runs on this computer: rules and tools do the work, and a small language model only puts it into words. "
         f"I answer with {f['model'] or 'no model right now'}, on {f.get('cpu') or 'this machine'}. ")
    s += f"I hold {f.get('episodes', 0)} remembered moments and know {f.get('people', 0)} of your people. "
    try:
        s += honest.capabilities(rt)
    except Exception:
        pass
    return s.strip()


def block(rt) -> str:
    """A few lines for the model's prompt -- true things about itself, so it does not say it cannot do what it can."""
    try:
        f = facts(rt)
        from . import honest
        caps = honest.capabilities(rt)
    except Exception:
        return ""
    parts = [f"You run on the person's own computer with the model {f['model'] or 'none'}; you are part of a larger system (rules and tools) "
             "that does timers, reminders, media keys, notes, the screen and files."]
    if caps:
        parts.append(caps)
    return "[ABOUT YOU]\n" + "\n".join(parts)
