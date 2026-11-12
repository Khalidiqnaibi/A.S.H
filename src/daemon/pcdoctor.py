"""
src/daemon/pcdoctor.py

"Why is my computer slow?" -- answered from the real numbers, by rule, never the model.

    why is my computer slow? / why is my pc so laggy?      what is wrong right now, biggest first
    what's using my memory? / what's eating my ram?         the programs using the most memory (grouped: 40 Chrome
                                                            processes are one "Chrome")
    what's using my cpu? / what's hogging the processor?    the programs using the most processor
    am I running out of space? / how much disk space        free space on each drive
    is my computer okay? / check my pc                      the same as "why is it slow", but says so when all is fine

It may offer to close the biggest program it can ("Want me to close Chrome? It's using 3.1 GB.") and only does so on a
plain yes, through the desktop executor -- and never offers ASH itself, the local model server, or anything of
Windows' own. Reading is psutil only: no network, nothing kept.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you|so|and|um)[,\s]+)*"
_PC = r"(?:computer|pc|laptop|machine|system|desktop|windows)"
_Z = r"\s*(?:please|now|today|right\s+now|lately|all\s+of\s+a\s+sudden)?\s*[.!?]*\s*$"
ASKS = [
    ("slow", re.compile(_A + r"(?:why\s+is\s+(?:my|the|this)\s+" + _PC + r"\s+(?:so\s+|being\s+|running\s+|really\s+)*"
                        r"(?:slow|laggy|lagging|sluggish|stuttering|hot|loud|struggling)"
                        r"|why\s+(?:does|is)\s+(?:everything|it)\s+(?:so\s+)?(?:slow|lag(?:gy|ging)?)"
                        r"|what'?s\s+(?:making|slowing)\s+(?:my|the)\s+" + _PC + r"(?:\s+(?:so\s+)?slow|\s+down)?)" + _Z, re.I)),
    ("check", re.compile(_A + r"(?:is\s+(?:my|the)\s+" + _PC + r"\s+(?:ok(?:ay)?|alright|fine|healthy)"
                         r"|check\s+(?:on\s+)?(?:my|the)\s+" + _PC + r"(?:'?s\s+health)?|" + _PC + r"\s+(?:health\s+)?check"
                         r"|how'?s\s+my\s+" + _PC + r"\s+(?:holding\s+up|doing))" + _Z, re.I)),
    ("memory", re.compile(_A + r"(?:what'?s|what\s+is|which\s+(?:app|program)s?\s+(?:is|are))\s+(?:using|eating|hogging|taking(?:\s+up)?|filling)\s+"
                          r"(?:(?:all|most|so\s+much)\s+(?:of\s+)?)?(?:my|the)\s+(?:memory|ram)" + _Z, re.I)),
    ("cpu", re.compile(_A + r"(?:what'?s|what\s+is|which\s+(?:app|program)s?\s+(?:is|are))\s+(?:using|eating|hogging|maxing(?:\s+out)?)\s+"
                       r"(?:(?:all|most|so\s+much)\s+(?:of\s+)?)?(?:my|the)\s+(?:cpu|processor)" + _Z, re.I)),
    ("disk", re.compile(_A + r"(?:am\s+i\s+running\s+(?:out|low)\s+(?:of|on)\s+(?:disk\s+)?(?:space|storage)"
                        r"|how\s+much\s+(?:disk\s+|free\s+)?(?:space|storage)\s+(?:do\s+i\s+have|is\s+(?:there\s+)?left)(?:\s+left)?"
                        r"|what'?s\s+taking\s+up\s+(?:all\s+)?(?:my\s+)?(?:disk\s+)?(?:space|storage))" + _Z, re.I)),
]

# Never offered for closing: ASH, its model server, Windows itself, and the security software.
PROTECTED = frozenset("""python pythonw python3 py ashd llama-server llama ollama ollama_llama_server ollama app system
    registry idle smss csrss wininit winlogon services lsass lsaiso svchost dwm explorer fontdrvhost sihost ctfmon
    taskhostw runtimebroker searchhost searchindexer startmenuexperiencehost shellexperiencehost textinputhost
    memory compression memcompression securityhealthservice msmpeng nissrv audiodg spoolsv conhost dllhost wudfhost
    smartscreen applicationframehost systemsettings lockapp wmiprvse""".split())
NICE = {"chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "code": "VS Code", "discord": "Discord",
        "spotify": "Spotify", "steam": "Steam", "steamwebhelper": "Steam", "teams": "Teams", "ms-teams": "Teams",
        "outlook": "Outlook", "winword": "Word", "excel": "Excel", "powerpnt": "PowerPoint", "slack": "Slack",
        "obs64": "OBS", "javaw": "Java", "node": "Node", "brave": "Brave", "opera": "Opera", "zoom": "Zoom",
        "photoshop": "Photoshop", "blender": "Blender", "vmmem": "WSL / virtual machines", "msedgewebview2": "Edge WebView"}


def nice(name: str) -> str:
    base = re.sub(r"\.exe$", "", (name or "").lower())
    return NICE.get(base, base[:1].upper() + base[1:] if base else "?")


def _gb(b: float) -> str:
    g = b / 1024 ** 3
    return f"{g:.1f} GB" if g >= 1 else f"{b / 1024 ** 2:.0f} MB"


@dataclass
class Group:
    name: str                   # "chrome"
    exe: str                    # "chrome.exe", what the desktop executor closes
    count: int = 0
    rss: float = 0.0            # bytes
    cpu: float = 0.0            # percent of one core summed; divided by cores when said


@dataclass
class Snapshot:
    at: float
    cpu: float                  # percent, whole machine
    cores: int
    mem_total: float
    mem_used: float
    mem_percent: float
    swap_percent: float
    disks: List[Tuple[str, float, float]] = field(default_factory=list)   # (mount, free bytes, percent used)
    groups: List[Group] = field(default_factory=list)
    uptime_s: float = 0.0
    battery: Optional[Tuple[float, bool]] = None                          # (percent, plugged in)


def read(psutil_mod=None, sample_s: float = 0.4, clock: Callable[[], float] = time.time) -> Snapshot:
    """The machine right now. Process CPU needs two readings `sample_s` apart."""
    ps = psutil_mod
    if ps is None:
        import psutil as ps                                            # type: ignore
    procs = []
    for p in ps.process_iter(["name", "memory_info"]):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except Exception:
            continue
    cpu = ps.cpu_percent(interval=sample_s)
    groups: Dict[str, Group] = {}
    for p in procs:
        try:
            name = (p.info.get("name") or "").strip()
            mi = p.info.get("memory_info")
            if not name or mi is None:
                continue
            key = re.sub(r"\.exe$", "", name.lower())
            g = groups.setdefault(key, Group(key, name))
            g.count += 1
            g.rss += float(getattr(mi, "rss", 0) or 0)
            g.cpu += float(p.cpu_percent(None) or 0.0)
        except Exception:
            continue
    vm = ps.virtual_memory()
    try:
        sw = ps.swap_memory().percent
    except Exception:
        sw = 0.0
    disks = []
    for part in ps.disk_partitions(all=False):
        try:
            if "cdrom" in (part.opts or "") or not part.fstype:
                continue
            u = ps.disk_usage(part.mountpoint)
            disks.append((part.mountpoint, float(u.free), float(u.percent)))
        except Exception:
            continue
    try:
        b = ps.sensors_battery()
        battery = (float(b.percent), bool(b.power_plugged)) if b is not None else None
    except Exception:
        battery = None
    now = clock()
    try:
        up = now - ps.boot_time()
    except Exception:
        up = 0.0
    return Snapshot(at=now, cpu=float(cpu), cores=ps.cpu_count() or 1, mem_total=float(vm.total), mem_used=float(vm.used),
                    mem_percent=float(vm.percent), swap_percent=float(sw), disks=disks,
                    groups=sorted(groups.values(), key=lambda g: -g.rss), uptime_s=up, battery=battery)


@dataclass
class Finding:
    what: str                   # memory | cpu | disk | swap | uptime | battery
    say: str
    weight: float               # how much it explains slowness, 0..1
    close: Optional[Group] = None   # a program it may offer to close


class PCDoctor:
    def __init__(self, reader: Optional[Callable[[], Snapshot]] = None, clock: Callable[[], float] = time.time,
                 mem_high: float = 85.0, cpu_high: float = 85.0, disk_low_percent: float = 90.0,
                 disk_low_bytes: float = 10 * 1024 ** 3):
        self.reader = reader or (lambda: read(clock=clock))
        self.clock = clock
        self.mem_high, self.cpu_high = mem_high, cpu_high
        self.disk_low_percent, self.disk_low_bytes = disk_low_percent, disk_low_bytes
        self.pending: Optional[Tuple[Group, float]] = None      # an offer to close, waiting for yes
        self.stats = {"asked": 0, "offered": 0, "closed": 0}
        self.last: Optional[Snapshot] = None

    @staticmethod
    def parse(text: str) -> Optional[str]:
        t = (text or "").strip()
        if len(t) > 120:
            return None
        for kind, rx in ASKS:
            if rx.match(t):
                return kind
        return None

    def closable(self, g: Group) -> bool:
        return g.name not in PROTECTED and not g.name.startswith(("python", "llama", "ollama"))

    def _top(self, s: Snapshot, n: int = 3, by: str = "rss") -> List[Group]:
        gs = sorted(s.groups, key=lambda g: -(g.rss if by == "rss" else g.cpu))
        return [g for g in gs if g.name not in ("system idle process", "idle")][:n]

    def _cpu_of(self, g: Group, s: Snapshot) -> float:
        return g.cpu / max(1, s.cores)

    def findings(self, s: Snapshot) -> List[Finding]:
        out: List[Finding] = []
        if s.mem_percent >= self.mem_high:
            top = self._top(s, 3, "rss")
            names = ", ".join(f"{nice(g.name)} {_gb(g.rss)}" for g in top)
            cand = next((g for g in top if self.closable(g)), None)
            out.append(Finding("memory", f"memory is {s.mem_percent:.0f}% full ({_gb(s.mem_used)} of {_gb(s.mem_total)}) -- "
                                         f"most of it is {names}", min(1.0, s.mem_percent / 100.0), cand))
        if s.cpu >= self.cpu_high:
            top = self._top(s, 3, "cpu")
            names = ", ".join(f"{nice(g.name)} {self._cpu_of(g, s):.0f}%" for g in top)
            cand = next((g for g in top if self.closable(g)), None)
            out.append(Finding("cpu", f"the processor is at {s.cpu:.0f}% -- {names}", min(1.0, s.cpu / 100.0), cand))
        if s.swap_percent >= 60 and s.mem_percent >= 75:
            out.append(Finding("swap", f"it is swapping to disk ({s.swap_percent:.0f}% of the page file in use), which slows "
                                       "everything down", 0.6))
        for mount, free, used in s.disks:
            if used >= self.disk_low_percent or free <= self.disk_low_bytes:
                out.append(Finding("disk", f"drive {mount.rstrip(chr(92))} has only {_gb(free)} free", 0.5))
        if s.uptime_s >= 7 * 86400:
            out.append(Finding("uptime", f"it hasn't been restarted in {int(s.uptime_s // 86400)} days -- a restart often helps",
                               0.3))
        if s.battery is not None and not s.battery[1] and s.battery[0] <= 20:
            out.append(Finding("battery", f"the battery is at {s.battery[0]:.0f}% and on battery power the system may be "
                                          "slowed down to save it", 0.4))
        return sorted(out, key=lambda f: -f.weight)

    def answer(self, kind: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        self.stats["asked"] += 1
        s = self.reader()
        self.last = s
        self.pending = None
        if kind == "memory":
            top = self._top(s, 4, "rss")
            line = (f"Memory is {s.mem_percent:.0f}% used ({_gb(s.mem_used)} of {_gb(s.mem_total)}). The biggest: "
                    + ", ".join(f"{nice(g.name)} {_gb(g.rss)}" + (f" across {g.count} processes" if g.count > 3 else "")
                                for g in top) + ".")
            return line + (self._offer(next((g for g in top if self.closable(g)), None), s, now, "memory")
                           if s.mem_percent >= self.mem_high else "")     # nothing is wrong: nothing offered
        if kind == "cpu":
            top = self._top(s, 4, "cpu")
            line = (f"The processor is at {s.cpu:.0f}%. Using the most: "
                    + ", ".join(f"{nice(g.name)} {self._cpu_of(g, s):.0f}%" for g in top) + ".")
            return line + (self._offer(next((g for g in top if self.closable(g)), None), s, now, "cpu")
                           if s.cpu >= self.cpu_high else "")
        if kind == "disk":
            if not s.disks:
                return "I couldn't read the drives."
            parts = [f"{m.rstrip(chr(92))} has {_gb(f)} free ({100 - u:.0f}%)" for m, f, u in s.disks]
            low = any(u >= self.disk_low_percent or f <= self.disk_low_bytes for _m, f, u in s.disks)
            return ("; ".join(parts) + "." + (" That's getting low -- emptying Downloads and the Recycle Bin is a good start."
                                             if low else " That's plenty."))
        fs = self.findings(s)
        if not fs:
            return (f"It looks fine right now: processor {s.cpu:.0f}%, memory {s.mem_percent:.0f}%"
                    + (f", {s.disks[0][0].rstrip(chr(92))} has {_gb(s.disks[0][1])} free" if s.disks else "")
                    + ". If it felt slow a moment ago, it has passed.")
        head = fs[0]
        rest = [f.say for f in fs[1:3]]
        line = ("It's slow mainly because " if kind == "slow" else "One thing: ") + head.say + "."
        if rest:
            line += " Also, " + "; ".join(rest) + "."
        return line + self._offer(head.close, s, now, head.what)

    def _offer(self, g: Optional[Group], s: Snapshot, now: float, why: str) -> str:
        if g is None or not self.closable(g):
            return ""
        if why == "memory" and g.rss < 0.5 * 1024 ** 3:
            return ""                                        # not worth closing anything for
        self.pending = (g, now)
        self.stats["offered"] += 1
        amount = _gb(g.rss) if why == "memory" else f"{self._cpu_of(g, s):.0f}% of the processor"
        return f" Want me to close {nice(g.name)}? It's using {amount} -- anything unsaved there may be lost."

    YES = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:yes|yeah|yep|sure|ok(?:ay)?|do\s+it|go\s+ahead|close\s+it|please(?:\s+do)?)\b[\s.!]*$", re.I)
    NO = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:no|nope|nah|don'?t|leave\s+it|never\s*mind|not\s+now)\b[\s.!]*$", re.I)

    def reply(self, text: str, close: Callable[[str], bool], now: Optional[float] = None, window_s: float = 60.0) -> Optional[str]:
        """The answer to "Want me to close X?", or None if this is not one (or the offer is too old)."""
        if self.pending is None:
            return None
        now = self.clock() if now is None else now
        g, at = self.pending
        if now - at > window_s:
            self.pending = None
            return None
        if self.YES.match(text or ""):
            self.pending = None
            ok = False
            try:
                ok = bool(close(g.exe))
            except Exception:
                ok = False
            if ok:
                self.stats["closed"] += 1
                return f"Closed {nice(g.name)}."
            return f"I couldn't close {nice(g.name)} -- it may need you to close it yourself."
        if self.NO.match(text or ""):
            self.pending = None
            return "Okay, I'll leave it."
        return None

    def status(self) -> Dict[str, Any]:
        s = self.last
        return {**self.stats, "last": ({"cpu": round(s.cpu), "memory": round(s.mem_percent)} if s else None)}
