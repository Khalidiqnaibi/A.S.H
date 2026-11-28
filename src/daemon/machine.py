"""
src/daemon/machine.py

"What's using my memory?", "why is my computer slow?", "what are my specs?", "can my PC run GTA 6?" -- the machine itself, by rule.

26 Nov memory sat at 90-93% all day, every model answer took a minute or timed out, and nothing could say why. These went to the model (which
cannot see the machine): "can my pc run gta 6" waited 300 s twice. They are read off the machine instead:

    memory    the five processes holding the most memory (summed by name: twenty Opera processes are one Opera), and ASH's own share
              (this process and the language model's server) -- with what can be let go: "unload the model"
    specs     the processor, its threads, the memory, the graphics card (read once, kept)
    can run   the specs, and plainly that ASH cannot check a game's requirements itself: it offers to search them

Nothing is sent anywhere; only process NAMES are said, never titles or command lines.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.machine")

_LEAD = r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:so|ok(?:ay)?|hey|and)[,\s]+)?(?:(?:can|could)\s+you\s+(?:tell\s+me\s+)?)?"
_MEM = re.compile(_LEAD + r"(?:what(?:'s|\s+is|\s+are)?\s+(?:using|eating|taking(?:\s+up)?|hogging|filling)\s+(?:(?:all|so\s+much|most)\s+(?:of\s+)?)?(?:my\s+|the\s+)?(?:memory|ram)|"
                  r"what(?:'s|\s+is)\s+(?:my\s+|the\s+)?(?:memory|ram)\s+(?:being\s+)?used\s+(?:by|for)|why\s+is\s+(?:my\s+|the\s+)?(?:memory|ram)\s+(?:so\s+)?(?:high|full)|"
                  r"why\s+is\s+(?:my\s+|the\s+|this\s+)?(?:computer|pc|laptop|machine)\s+(?:so\s+|being\s+)?(?:slow|laggy|sluggish)(?:\s+(?:today|right\s+now|now))?|"
                  r"(?:how\s+much|which)\s+(?:memory|ram)\s+(?:are\s+you|is\s+(?:ash|ollama|the\s+model))\s+using|what\s+is\s+slowing\s+(?:down\s+)?(?:my\s+|the\s+)?(?:computer|pc|laptop))"
                  r"(?:\s+(?:right\s+now|now|today))?\s*[?.!]*\s*$", re.I)
_SPECS = re.compile(_LEAD + r"(?:what(?:'s|\s+is|\s+are)\s+(?:my|the|this)\s+(?:computer'?s\s+|pc'?s\s+|laptop'?s\s+)?(?:specs|specifications|hardware|config(?:uration)?)|"
                    r"what\s+(?:gpu|graphics\s+card|cpu|processor)\s+(?:do\s+i\s+have|is\s+(?:this|in\s+(?:my|this)\s+(?:computer|pc|laptop)))|"
                    r"how\s+much\s+(?:ram|memory)\s+(?:do\s+i\s+have|does\s+(?:my|this)\s+(?:computer|pc|laptop)\s+have)|"
                    r"what(?:'s|\s+is)\s+my\s+(?:gpu|graphics\s+card|cpu|processor))\s*[?.!]*\s*$", re.I)
_DEV = r"(?:pc|computer|laptop|machine|rig|device|system|setup|potato)"
# "Can this little stupid old device run GTA 6?" (28 Nov) -- adjectives between "this" and the device; "can I run X", "will X run on this pc"
_RUN = re.compile(_LEAD + r"(?:(?:can|could|will|would)\s+(?:my|this|the)\s+(?:[\w,' -]{0,40}?\s+)?" + _DEV + r"\s+(?:even\s+)?(?:run|handle|play)\s+(?P<what>.{2,60}?)"
                  r"|(?:can|could)\s+i\s+(?:even\s+)?(?:run|play)\s+(?P<what2>.{2,60}?)\s+on\s+(?:my|this|the)\s+(?:[\w,' -]{0,40}?\s+)?" + _DEV +
                  r"|will\s+(?P<what3>.{2,60}?)\s+(?:even\s+)?run\s+on\s+(?:my|this|the)\s+(?:[\w,' -]{0,40}?\s+)?" + _DEV + r")\s*[?.!]*\s*$", re.I)


def parse(text: str) -> Optional[Tuple[str, str]]:
    """("memory", "") / ("specs", "") / ("run", "GTA 6"), or None."""
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    if _MEM.match(t):
        return ("memory", "")
    if _SPECS.match(t):
        return ("specs", "")
    m = _RUN.match(t)
    if m:
        return ("run", (m.group("what") or m.group("what2") or m.group("what3")).strip(" .?!"))
    return None


def _gb(b: float) -> str:
    g = b / (1024 ** 3)
    return f"{g:.1f} GB" if g < 10 else f"{g:.0f} GB"


_FRIENDLY = {"opera": "Opera", "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "code": "VS Code", "ollama": "the language model (Ollama)",
             "ollama_llama_server": "the language model (Ollama)", "llama-server": "the language model", "discord": "Discord", "spotify": "Spotify",
             "steam": "Steam", "steamwebhelper": "Steam", "teams": "Teams", "ms-teams": "Teams", "explorer": "Explorer", "memcompression": "Windows memory compression",
             "msmpeng": "Windows Defender", "dwm": "the desktop (dwm)", "python": "Python", "pythonw": "Python", "java": "Java", "javaw": "Java",
             "olk": "Outlook", "outlook": "Outlook", "winword": "Word", "excel": "Excel", "obs64": "OBS"}


def _name(proc: str) -> str:
    stem = re.sub(r"\.(?:exe|bin|app)$", "", (proc or "").strip().lower())
    return _FRIENDLY.get(stem, stem or "?")


def processes() -> List[Tuple[str, int, int]]:
    """[(process name, resident bytes, pid)] for everything that can be read."""
    import psutil
    out = []
    for p in psutil.process_iter(["name", "memory_info", "pid"]):
        try:
            mi = p.info.get("memory_info")
            if mi is not None:
                out.append((p.info.get("name") or "", int(mi.rss), int(p.info.get("pid") or 0)))
        except Exception:
            continue
    return out


def memory_now() -> Tuple[float, int, int]:
    """(percent used, used bytes, total bytes)."""
    import psutil
    vm = psutil.virtual_memory()
    return float(vm.percent), int(vm.total - vm.available), int(vm.total)


def _gpu_names() -> List[str]:
    if not sys.platform.startswith("win"):
        return []
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"], capture_output=True, text=True,
                           timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    except Exception:
        return []


def _cpu_name() -> str:
    if sys.platform.startswith("win"):
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split())
        except Exception:
            pass
    return platform.processor() or ""


def specs() -> Dict[str, Any]:
    import psutil
    return {"cpu": _cpu_name(), "threads": psutil.cpu_count(logical=True) or 0, "cores": psutil.cpu_count(logical=False) or 0,
            "ram": int(psutil.virtual_memory().total), "gpus": _gpu_names()}


def spoken_cpu(name: str) -> str:
    """'Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz' -> 'Intel Core i5-10500H' (26 Nov it was read out as written)."""
    s = re.sub(r"\((?:R|TM|C)\)|[\u00ae\u2122]", "", name or "")
    s = re.sub(r"\s*@\s*[\d.]+\s*[GM]Hz", "", s, flags=re.I)
    s = re.sub(r"\b(?:CPU|Processor|with\s+Radeon\s+Graphics|\d+-Core)\b", "", s, flags=re.I)
    return " ".join(s.split())


def _article(name: str) -> str:
    """'an RTX 3060', 'a GTX 1650', 'an Intel Core i5', 'a Radeon RX 6600': by how the first letter or word is said."""
    first = (name or "").split()[0] if (name or "").split() else ""
    if first.isupper() and len(first) > 1:
        return "an " if first[0] in "AEFHILMNORSX" else "a "
    return "an " if first[:1].lower() in "aeiou" else "a "


def spoken_gpus(names: List[str]) -> str:
    """['Intel(R) UHD Graphics', 'NVIDIA GeForce GTX 1650 with Max-Q Design'] -> 'a GTX 1650 (and Intel graphics built in)'."""
    clean = []
    for g in names:
        s = re.sub(r"\((?:R|TM)\)|[\u00ae\u2122]", "", g)
        s = re.sub(r"\s+with\s+Max-Q\s+Design|\s+Laptop\s+GPU|\bGeForce\s+|\bNVIDIA\s+|\bAMD\s+(?=Radeon)", " ", s, flags=re.I)
        clean.append(" ".join(s.split()))
    built_in = [c for c in clean if re.match(r"^(?:Intel|AMD\s+Radeon\(?TM\)?\s+Graphics|Radeon\s+Graphics)", c, re.I) or re.search(r"\bUHD|Iris|Vega\b", c)]
    real = [c for c in clean if c not in built_in]
    if real and built_in:
        return " and ".join(_article(r) + r for r in real) + " (and " + ("Intel" if "intel" in built_in[0].lower() else "built-in") + " graphics)"
    return " and ".join(_article(c) + c for c in clean)


_ONSETS = frozenset("bl br cl cr dr fl fr gl gr pl pr sc sk sl sm sn sp st sw tr tw wr th sh ch ph wh kn gn qu sq".split())


def spoken_name(what: str) -> str:
    """'gta 6' -> 'GTA 6': a short lower-case word that cannot be read as an English word -- no vowel, or two consonants no English word starts
    with ('gt', 'nb', 'cp') -- is an acronym. 'elden ring' stays as it is."""
    def acr(w: str) -> bool:
        if not (2 <= len(w) <= 4 and w.isalpha() and w.islower()):
            return False
        if not re.search(r"[aeiouy]", w):
            return True
        return len(w) >= 3 and w[0] not in "aeiouy" and w[1] not in "aeiouy" and w[:2] not in _ONSETS
    return " ".join(w.upper() if acr(w) else w for w in (what or "").split())


class Machine:
    def __init__(self, procs: Callable[[], List[Tuple[str, int, int]]] = processes, mem: Callable[[], Tuple[float, int, int]] = memory_now,
                 spec: Callable[[], Dict[str, Any]] = specs, own_pid: Optional[int] = None,
                 online: Optional[Callable[[], bool]] = None, req: Optional[Callable[[str], Any]] = None):
        self.procs, self.mem, self.spec = procs, mem, spec
        self.online = online                    # is there a network (learnweb.is_online when not given)
        if req is None:
            from . import canrun
            req = canrun.lookup
        self.req = req                          # game name -> {"name", "minimum", "recommended"} or None
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self._spec: Optional[Dict[str, Any]] = None

    def memory(self) -> str:
        pct, used, total = self.mem()
        by: Dict[str, int] = {}
        mine = 0
        model = 0
        for name, rss, pid in self.procs():
            n = _name(name)
            if pid == self.own_pid:
                mine += rss
                continue
            if "language model" in n:
                model += rss
            by[n] = by.get(n, 0) + rss
        top = sorted(by.items(), key=lambda kv: -kv[1])[:5]
        line = f"Memory is at {pct:.0f}% ({_gb(used)} of {_gb(total)})."
        if top:
            line += " The most: " + ", ".join(f"{n} {_gb(b)}" for n, b in top) + "."
        line += f" I'm using {_gb(mine)} myself" + (f", and the language model holds {_gb(model)}" if model >= 200 * 1024 ** 2 else "") + "."
        if model >= 500 * 1024 ** 2:
            line += " Say “unload the model” and I'll let that go until the next question."
        return line

    def specs(self) -> str:
        s = self._spec = self._spec or self.spec()
        bits = []
        if s.get("cpu"):
            cpu = spoken_cpu(s["cpu"])
            bits.append(_article(cpu) + cpu + (f" ({s['threads']} threads)" if s.get("threads") else ""))
        if s.get("ram"):
            bits.append(f"{_gb(s['ram'])} of memory")
        gpus = [g for g in (s.get("gpus") or []) if not re.search(r"\b(?:basic\s+display|remote|virtual|parsec|meta)\b", g, re.I)]
        if gpus:
            bits.append(spoken_gpus(gpus))
        if not bits:
            return "I couldn't read this computer's hardware."
        return "This computer has " + (", ".join(bits[:-1]) + " and " + bits[-1] if len(bits) > 1 else bits[0]) + "."

    def can_run(self, what: str) -> str:
        """Online: what the game asks for (the Steam page) against this computer. Offline: the specs, and that it can't look it up -- never a guess."""
        what = spoken_name(what)
        online = self.online
        if online is None:
            from .net import online
        if not online():
            return (self.specs() + f" I'm offline right now, so I can't look up what {what} needs -- ask me again when I'm online.")
        try:
            game = self.req(what)
        except Exception as e:
            return self.specs() + f" I couldn't check what {what} needs -- {e}."
        if game is None:
            return (self.specs() + f" Steam doesn't list {what} with requirements (it may not be out on PC, or it's sold elsewhere), so I can't check it.")
        from . import canrun
        return canrun.judge(game, self._spec or self.spec())

    def answer(self, kind: str, what: str = "") -> str:
        if kind == "memory":
            return self.memory()
        if kind == "specs":
            return self.specs()
        return self.can_run(what)
