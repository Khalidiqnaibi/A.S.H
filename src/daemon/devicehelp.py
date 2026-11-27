"""
src/daemon/devicehelp.py

The machine's small practical questions, by rule, from what the computer itself reports -- never the language model:

    Bluetooth   "I'm having trouble with my Live Buds" (26 Nov: 75 s on the model for "could you tell me more?") -> "Your Galaxy Buds Live are paired
                but not connected right now. Want Bluetooth settings?"; "are my buds connected?", "what's connected over Bluetooth?", "open Bluetooth
                settings". Windows offers no plain way to connect a paired device, so ASH says what it sees and opens the page where it is done.
    the network "is the internet down?" / "why is the internet slow?" -> the Wi-Fi, the router, name lookups and the outside, each checked once on this
                machine: "Your Wi-Fi is fine as far as the router; nothing answers past it." Nothing about the person is sent anywhere.
    awake       "keep my PC awake for 2 hours" / "don't let it sleep until 6" / "let it sleep" -- the way PowerToys Awake does it, and it ends by itself.
    volume      "volume to 30", "set the volume to 50 percent", "what's the volume?" -- the media keys only step it.
    diagnostic  "run a diagnostic" / "self check" / "are your systems okay?" -> the microphone, the model, memory and CPU, what is on, the last
                trouble -- and what would help, in a few lines.

Each part takes its reader or setter as an argument, so tests never touch the real machine.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.devicehelp")

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|um+|uh+)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
_Z = r"(?:[,\s]+(?:ash|please|now|for\s+me))*\s*[?.!]*\s*$"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ps(command: str, timeout: float = 15.0) -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=timeout,
                              creationflags=_NO_WINDOW).stdout or ""
    except Exception:
        logger.debug("powershell failed", exc_info=True)
        return ""


# ------------------------------------------------------------------------------------------------------------------------------ Bluetooth
_NOT_A_DEVICE = re.compile(r"\b(?:enumerator|adapter|radio|protocol|rfcomm|generic\s+(?:attribute|access)|le\s+generic|wireless\s+bluetooth|"
                           r"bluetooth\s+device\s+\(|microsoft|intel\(r\)|realtek|qualcomm|mediatek|broadcom|personal\s+area\s+network|"
                           r"device\s+identification|service\s+discovery)\b", re.I)
_PART = re.compile(r"\s+(?:avrcp\s+transport|hands-?free(?:\s+ag)?(?:\s+audio)?|stereo|a2dp\s+sink|audio\s+gateway|hid|le)\s*$", re.I)


def bluetooth_devices() -> List[Tuple[str, bool]]:
    """(name, connected) for what Windows has paired, read with Get-PnpDevice. 'OK' is there now; 'Unknown' is paired and not present."""
    out = _ps("Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | ForEach-Object { $_.Status + '|' + $_.FriendlyName }")
    seen: Dict[str, bool] = {}
    for ln in out.splitlines():
        if "|" not in ln:
            continue
        status, name = ln.split("|", 1)
        name = _PART.sub("", name.strip())
        if not name or _NOT_A_DEVICE.search(name):
            continue
        seen[name] = seen.get(name, False) or status.strip().upper() == "OK"
    return list(seen.items())


_EARS = r"(?:ear\s?buds?|buds|ear\s?phones?|head\s?phones?|head\s?set|air\s?pods|pods)"
_KINDS = {"ears": re.compile(r"\b(?:buds|airpods|pods|headphones?|headset|earphones?|freebuds|wh-|wf-|jabra|bose|beats|soundcore|jbl|sony\s+w)", re.I),
          "speaker": re.compile(r"\b(?:speaker|soundbar|jbl|flip|charge|boom)", re.I),
          "mouse": re.compile(r"\bmouse|mx\s+master|mx\s+anywhere", re.I), "keyboard": re.compile(r"\bkeyboard|keys\b", re.I),
          "controller": re.compile(r"\b(?:controller|gamepad|xbox|dualsense|dualshock)", re.I), "watch": re.compile(r"\bwatch\b", re.I),
          "phone": re.compile(r"\b(?:phone|galaxy\s+s\d|iphone|pixel)", re.I)}
_KIND_OF = [(re.compile(r"\b" + _EARS + r"\b", re.I), "ears"), (re.compile(r"\bspeakers?\b|\bsoundbar\b", re.I), "speaker"),
            (re.compile(r"\bmouse\b", re.I), "mouse"), (re.compile(r"\bkeyboard\b", re.I), "keyboard"),
            (re.compile(r"\b(?:controller|gamepad)\b", re.I), "controller"), (re.compile(r"\bwatch\b", re.I), "watch"),
            (re.compile(r"\bphone\b", re.I), "phone")]

_DEV = r"(?:my\s+|the\s+)?(?P<dev>[\w' -]{2,40}?)"
_BT_RX = [
    ("trouble", re.compile(_A + r"(?:i'?m\s+|i\s+am\s+)?(?:having\s+)?(?:a\s+)?(?:some\s+)?(?:trouble|problems?|issues?|a\s+hard\s+time)\s+(?:with\s+)?"
                           r"(?:connecting\s+|pairing\s+|using\s+)?" + _DEV + _Z, re.I)),
    ("trouble", re.compile(_A + r"my\s+(?P<dev>[\w' -]{2,40}?)\s+(?:won'?t|wont|don'?t|doesn'?t|aren'?t|isn'?t|can'?t|will\s+not)\s+(?:connect|pair|work)"
                           r"(?:\s+(?:anymore|again|properly|right))?" + _Z, re.I)),
    ("status", re.compile(_A + r"(?:are|is)\s+" + _DEV + r"\s+(?:connected|paired|on|working)" + _Z, re.I)),
    ("connect", re.compile(_A + r"(?:connect|pair|reconnect)\s+" + _DEV + r"(?:\s+(?:to\s+(?:the\s+|my\s+)?(?:pc|computer|laptop)|again))?" + _Z, re.I)),
    ("list", re.compile(_A + r"(?:what(?:'s|\s+is|\s+are)\s+(?:connected|paired)\s+(?:over|via|by|with)\s+bluetooth|(?:what|which)\s+bluetooth\s+devices"
                        r"(?:\s+(?:are|do\s+i\s+have))?(?:\s+(?:connected|paired))?|list\s+(?:my\s+)?bluetooth\s+devices)" + _Z, re.I)),
    ("settings", re.compile(_A + r"(?:open|show|bring\s+up)\s+(?:the\s+)?bluetooth(?:\s+settings|\s+page)?" + _Z, re.I)),
    ("sound", re.compile(_A + r"(?:open|show|bring\s+up)\s+(?:the\s+)?(?:sound|audio)\s+settings" + _Z, re.I)),
]
BT_SETTINGS, SOUND_SETTINGS = "ms-settings:bluetooth", "ms-settings:sound"


def _bare(name: str) -> str:
    """'Galaxy Buds Live (0922)' -> 'Galaxy Buds Live': the code in brackets is Windows', not the name."""
    return re.sub(r"\s*\([0-9A-Fa-f]{4}\)$", "", name or "")


class Bluetooth:
    def __init__(self, reader: Callable[[], List[Tuple[str, bool]]] = bluetooth_devices, clock: Callable[[], float] = time.time):
        self.reader, self.clock = reader, clock
        self.pending: Optional[Tuple[str, float]] = None
        self._names: Tuple[float, List[str]] = (0.0, [])

    def names(self) -> List[str]:
        """Paired device names, for the recogniser's hints ("K-Live" / "live buds" were the Galaxy Buds Live). Read at most once an hour."""
        at, cached = self._names
        if cached and self.clock() - at < 3600.0:
            return cached
        try:
            names = [n for n, _c in self.reader()]
        except Exception:
            names = []
        self._names = (self.clock(), names)
        return names

    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, str]]:
        t = (text or "").strip()
        if len(t) > 120:
            return None
        for kind, rx in _BT_RX:
            m = rx.match(t)
            if not m:
                continue
            dev = (m.groupdict().get("dev") or "").strip()
            if kind in ("trouble", "status", "connect"):
                low = dev.lower()
                if not (any(rx2.search(low) for rx2, _k in _KIND_OF) or "bluetooth" in low or re.search(r"\b(?:live|galaxy|sony|bose|jbl)\b", low)):
                    return None                     # "trouble with my code", "connect the dots": not a device
            return kind, dev
        return None

    def _match(self, dev: str, rows: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
        low = dev.lower()
        kind = next((k for rx, k in _KIND_OF if rx.search(low)), "")
        words = {w for w in re.findall(r"[a-z0-9]{3,}", low) if w not in ("my", "the", "bluetooth", "wireless", "buds", "earbuds", "headphones",
                                                                          "headset", "speaker", "mouse", "keyboard", "controller")}
        hits = []
        for name, conn in rows:
            n = name.lower()
            score = (2 if kind and _KINDS[kind].search(n) else 0) + sum(1 for w in words if w in n)
            if score:
                hits.append((score, name, conn))
        if not hits:
            return []
        best = max(h[0] for h in hits)
        return [(n, c) for s, n, c in hits if s == best]

    def answer(self, kind: str, dev: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        if kind == "settings":
            return "open:" + BT_SETTINGS
        if kind == "sound":
            return "open:" + SOUND_SETTINGS
        try:
            rows = self.reader()
        except Exception:
            rows = []
        if kind == "list":
            if not rows:
                return "I don't see anything paired over Bluetooth."
            on = [_bare(n) for n, c in rows if c]
            off = [_bare(n) for n, c in rows if not c]
            parts = []
            if on:
                parts.append("Connected: " + ", ".join(on[:5]) + ".")
            if off:
                parts.append("Paired but not connected: " + ", ".join(off[:5]) + ".")
            return " ".join(parts)
        hits = self._match(dev, rows)
        what = re.sub(r"^(?:my|the)\s+", "", dev, flags=re.I).strip() or "that"
        if not hits:
            self.pending = (BT_SETTINGS, now)
            return f"I don't see your {what} paired with this computer. Want me to open Bluetooth settings so you can pair them?"
        name, conn = hits[0]
        name = _bare(name)
        plural = bool(re.search(r"(?:buds|phones|pods|s)\b", name.lower()))
        are, they, show = ("are", "them", "show") if plural else ("is", "it", "shows")
        if conn:
            self.pending = (SOUND_SETTINGS, now)
            if kind == "status":
                return f"Your {name} {show} as connected. Want the sound settings, to check the sound goes to {they}?"
            return (f"Your {name} {show} as connected, so if there's no sound it's probably going somewhere else. "
                    "Want me to open the sound settings?")
        self.pending = (BT_SETTINGS, now)
        return f"Your {name} {are} paired but not connected right now. Want me to open Bluetooth settings to connect {they}?"

    def reply(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """'open:<uri>' on a yes to the last offer (within two minutes), '' on a no, None when it is not an answer to it."""
        now = self.clock() if now is None else now
        if self.pending is None:
            return None
        uri, at = self.pending
        if now - at > 120.0:
            self.pending = None
            return None
        from . import yesno
        if yesno.is_yes(text):
            self.pending = None
            return "open:" + uri
        if yesno.is_no(text):
            self.pending = None
            return ""
        return None


# -------------------------------------------------------------------------------------------------------------------------------- network
_NET_RX = re.compile(_A + r"(?:is\s+(?:the|my)\s+(?:internet|wi-?fi|connection|network)\s+(?:down|working|ok(?:ay)?|up|out|slow|broken|dead)|"
                     r"(?:is|does)\s+(?:the|my)\s+(?:internet|wi-?fi|connection)\s+work(?:ing)?|am\s+i\s+(?:online|offline|connected(?:\s+to\s+the\s+internet)?)|"
                     r"why\s+is\s+(?:the|my)\s+(?:internet|wi-?fi|connection)\s+(?:so\s+)?(?:slow|down|not\s+working|bad)|"
                     r"(?:check|test)\s+(?:the|my)\s+(?:internet|wi-?fi|connection|network)(?:\s+connection)?|"
                     r"(?:is\s+)?(?:the\s+)?internet\s+(?:down|out)|do\s+i\s+have\s+(?:internet|wi-?fi)|"
                     r"(?:i\s+have\s+)?no\s+internet|the\s+internet\s+(?:isn'?t|is\s+not)\s+working)" + _Z, re.I)


def wifi_now() -> Tuple[str, Optional[int]]:
    """(network name, signal %) from netsh; ('', None) on a cable or with Wi-Fi off."""
    if not sys.platform.startswith("win"):
        return "", None
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=6, creationflags=_NO_WINDOW).stdout or ""
    except Exception:
        return "", None
    ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    sig = re.search(r"^\s*Signal\s*:\s*(\d+)%", out, re.M)
    state = re.search(r"^\s*State\s*:\s*(\w+)", out, re.M)
    if state and state.group(1).lower() != "connected":
        return "", None
    return (ssid.group(1).strip() if ssid else ""), (int(sig.group(1)) if sig else None)


def gateway() -> str:
    try:
        import psutil  # noqa: F401
    except Exception:
        pass
    out = _ps("(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue | Sort-Object RouteMetric | Select-Object -First 1).NextHop",
              timeout=8)
    gw = out.strip().splitlines()[0].strip() if out.strip() else ""
    return gw if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", gw) and gw != "0.0.0.0" else ""


def ping(host: str, timeout_ms: int = 1000) -> Optional[float]:
    """Round trip in ms, or None."""
    if not host:
        return None
    try:
        out = subprocess.run(["ping", "-n", "1", "-w", str(timeout_ms), host], capture_output=True, text=True, timeout=timeout_ms / 1000 + 2,
                             creationflags=_NO_WINDOW).stdout or ""
    except Exception:
        return None
    m = re.search(r"time[=<](\d+)\s*ms", out)
    return float(m.group(1)) if m else None


def resolves(name: str = "wikipedia.org") -> bool:
    try:
        socket.setdefaulttimeout(3)
        return bool(socket.getaddrinfo(name, 443))
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def outside(host: str = "1.1.1.1", port: int = 443) -> Optional[float]:
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=3):
            return (time.time() - t0) * 1000.0
    except Exception:
        return None


class Network:
    def __init__(self, wifi=wifi_now, gw=gateway, ping_fn=ping, dns=resolves, out=outside):
        self.wifi, self.gw, self.ping, self.dns, self.out = wifi, gw, ping_fn, dns, out

    @staticmethod
    def parse(text: str) -> bool:
        return bool(_NET_RX.match((text or "").strip()))

    def check(self) -> str:
        name, sig = self.wifi()
        wifi = (f"Wi-Fi “{name}”" + (f" at {sig}%" if sig is not None else "")) if name else ""
        o = self.out()
        d = self.dns()
        if o is not None and d:
            slow = o > 150
            return (f"The internet is working" + (f": {wifi}" if wifi else "") + f", and the outside answers in {o:.0f} ms"
                    + (" -- slow, so the line or the Wi-Fi is struggling." if slow else ".")
                    + (" The signal is weak -- closer to the router helps." if sig is not None and sig < 40 else ""))
        if o is not None and not d:
            return ("The connection is up, but names aren't resolving -- a DNS problem. Restarting the router, or setting the DNS to 1.1.1.1, usually "
                    "fixes it.")
        gw = self.gw()
        if not gw and not name:
            return "You're not connected to any network -- Wi-Fi is off or not joined, and there's no cable."
        rt = self.ping(gw) if gw else None
        if rt is not None:
            return ((f"Your {wifi} is fine" if wifi else "Your network is fine") + " as far as the router, but nothing answers past it -- the "
                    "internet line itself is down. Restarting the router is the first thing to try.")
        tail = ("can't reach the router -- the local connection is the problem. Turning Wi-Fi off and on, or restarting the router, usually "
                "fixes it.")
        return f"You're on {wifi}, but I {tail}" if wifi else f"I {tail}"


# ---------------------------------------------------------------------------------------------------------------------------------- awake
_AWAKE_ON = re.compile(_A + r"(?:keep\s+(?:my\s+|the\s+|this\s+)?(?:pc|computer|laptop|machine|screen|it)\s+(?:awake|on|from\s+sleeping)|"
                       r"don'?t\s+let\s+(?:my\s+|the\s+|this\s+)?(?:pc|computer|laptop|machine|it|the\s+screen)\s+(?:go\s+to\s+)?sleep|"
                       r"stay\s+awake|no\s+sleep(?:ing)?)(?:\s+(?P<how>(?:for|until|till|til)\s+.+?))?" + _Z, re.I)
_AWAKE_OFF = re.compile(_A + r"(?:let\s+(?:my\s+|the\s+|this\s+)?(?:pc|computer|laptop|machine|it)\s+sleep(?:\s+again)?|stop\s+keeping\s+(?:it|the\s+(?:pc|computer|laptop))\s+awake|"
                        r"you\s+can\s+let\s+it\s+sleep(?:\s+now)?)" + _Z, re.I)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10, "twenty": 20, "thirty": 30,
        "forty": 40, "forty-five": 45, "ninety": 90, "half an": 0.5, "half a": 0.5}
MAX_AWAKE_S = 8 * 3600.0


def _for(how: str, now: float) -> Optional[float]:
    """'for 2 hours' / 'for 30 minutes' / 'until 6' / 'until 6:30 pm' -> seconds from now (None: not understood)."""
    h = (how or "").strip().lower()
    m = re.match(r"(?:for\s+)?(?:an?\s+)?(?P<n>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|eight|ten|twenty|thirty|forty|forty-five|ninety|half\s+an|half\s+a)"
                 r"\s*(?P<u>hours?|hrs?|h|minutes?|mins?|m)\b", h)
    if m:
        n = m.group("n")
        v = float(n) if re.match(r"\d", n) else float(_NUM.get(re.sub(r"\s+", " ", n), 1))
        return v * (3600.0 if m.group("u").startswith("h") else 60.0)
    if re.match(r"for\s+(?:an?\s+)?hour\b", h):
        return 3600.0
    m = re.match(r"(?:until|till|til)\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?", h)
    if m:
        hh, mm = int(m.group("h")), int(m.group("m") or 0)
        ap = (m.group("ap") or "").replace(".", "")
        lt = time.localtime(now)
        if ap == "pm" and hh < 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        elif not ap and hh < 12 and hh <= lt.tm_hour:
            hh += 12                                         # "until 6" said in the afternoon is 6 in the evening
        target = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh % 24, mm, 0, 0, 0, -1))
        if target <= now:
            target += 86400.0
        return target - now
    return None


def _es_setter(on: bool) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)))
    except Exception:
        return False


class Awake:
    """Keeps the computer from sleeping until a time, on a thread of its own (the request belongs to the thread that made it)."""

    def __init__(self, setter: Callable[[bool], bool] = _es_setter, clock: Callable[[], float] = time.time, threaded: bool = True):
        self.setter, self.clock, self.threaded = setter, clock, threaded
        self.until: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def active(self, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        return self.until is not None and now < self.until

    def start(self, seconds: float, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        self.stop()
        self.until = now + min(MAX_AWAKE_S, max(60.0, seconds))
        if not self.threaded:
            return bool(self.setter(True))
        self._stop = threading.Event()
        ok = {"v": False}
        started = threading.Event()

        def run(stop=self._stop):
            ok["v"] = bool(self.setter(True))
            started.set()
            while not stop.wait(5.0):
                if self.until is None or self.clock() >= self.until:
                    break
            self.setter(False)
            self.until = None
        self._thread = threading.Thread(target=run, name="ash-awake", daemon=True)
        self._thread.start()
        started.wait(2.0)
        return ok["v"]

    def stop(self) -> bool:
        was = self.until is not None
        self.until = None
        self._stop.set()
        if not self.threaded and was:
            self.setter(False)
        return was

    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, str]]:
        t = (text or "").strip()
        if _AWAKE_OFF.match(t):
            return "off", ""
        m = _AWAKE_ON.match(t)
        if m:
            return "on", (m.group("how") or "")
        return None

    def answer(self, kind: str, how: str, now: Optional[float] = None) -> str:
        now = self.clock() if now is None else now
        if kind == "off":
            return "Okay -- it can sleep again." if self.stop() else "It wasn't being kept awake."
        if re.search(r"download|finish|done|i\s+say|i'?m\s+back|stop", how or "", re.I) or not how:
            secs, said = 3 * 3600.0, "for up to 3 hours -- say “let it sleep” when you're done"
        else:
            secs = _for(how, now)
            if secs is None:
                return "For how long? Say something like “keep it awake for 2 hours” or “until 6”."
            secs = min(secs, MAX_AWAKE_S)
            end = time.strftime("%H:%M", time.localtime(now + secs))
            said = f"until {end}" + (" (8 hours is the most I'll hold it)" if secs >= MAX_AWAKE_S else "")
        if not self.start(secs, now):
            return "I couldn't stop it from sleeping from here."
        return f"Okay -- I'll keep the computer awake {said}. The screen can still turn off."


# --------------------------------------------------------------------------------------------------------------------------------- volume
_VOL_SET = re.compile(_A + r"(?:(?:set|put|turn|change|make)\s+(?:the\s+)?(?:volume|sound)(?:\s+(?:up|down))?\s+(?:to|at)\s+|volume\s+(?:to\s+|at\s+)?)"
                      r"(?P<n>\d{1,3}|[a-z]+(?:[\s-][a-z]+)?)\s*(?:%|percent|per\s+cent)?" + _Z, re.I)
_VOL_GET = re.compile(_A + r"(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:volume|sound\s+level)(?:\s+at)?|how\s+loud\s+is\s+(?:it|the\s+(?:volume|sound))(?:\s+set)?)" + _Z, re.I)
_WORDNUM = {"zero": 0, "ten": 10, "fifteen": 15, "twenty": 20, "twenty five": 25, "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "seventy five": 75, "seventy-five": 75, "eighty": 80, "ninety": 90, "hundred": 100, "a hundred": 100,
            "max": 100, "maximum": 100, "full": 100, "half": 50, "mute": 0}


class Volume:
    """The default speaker's level, through Windows' Core Audio (comtypes). get() -> 0..100 or None; set(n) -> bool."""

    def __init__(self, get: Optional[Callable[[], Optional[int]]] = None, set_: Optional[Callable[[int], bool]] = None):
        self.get = get or _core_get
        self.set = set_ or _core_set

    @staticmethod
    def parse(text: str) -> Optional[Tuple[str, Optional[int]]]:
        t = (text or "").strip()
        if _VOL_GET.match(t):
            return "get", None
        m = _VOL_SET.match(t)
        if m:
            n = m.group("n").lower()
            v = int(n) if n.isdigit() else _WORDNUM.get(n)
            if v is None or not 0 <= v <= 100:
                return None
            return "set", v
        return None

    def answer(self, kind: str, n: Optional[int]) -> str:
        if kind == "get":
            v = self.get()
            return f"The volume is at {v}%." if v is not None else "I can't read the volume from here."
        if not self.set(int(n or 0)):
            return "I couldn't set the volume from here."
        return f"Volume at {n}%." if n else "Volume at zero."


def _endpoint():
    """IAudioEndpointVolume of the default render device, built with comtypes (the interfaces pycaw uses, declared here)."""
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
    from ctypes import POINTER, c_float, c_uint, c_int, c_void_p
    from ctypes.wintypes import BOOL, DWORD, LPCWSTR

    class IAudioEndpointVolume(IUnknown):
        _iid_ = GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")
        _methods_ = [
            COMMETHOD([], HRESULT, "RegisterControlChangeNotify", (["in"], c_void_p, "pNotify")),
            COMMETHOD([], HRESULT, "UnregisterControlChangeNotify", (["in"], c_void_p, "pNotify")),
            COMMETHOD([], HRESULT, "GetChannelCount", (["out"], POINTER(c_uint), "pnChannelCount")),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevel", (["in"], c_float, "fLevelDB"), (["in"], POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevelScalar", (["in"], c_float, "fLevel"), (["in"], POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevel", (["out"], POINTER(c_float), "pfLevelDB")),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevelScalar", (["out"], POINTER(c_float), "pfLevel")),
        ]

    class IMMDevice(IUnknown):
        _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = [COMMETHOD([], HRESULT, "Activate", (["in"], POINTER(GUID), "iid"), (["in"], DWORD, "dwClsCtx"),
                               (["in"], c_void_p, "pActivationParams"), (["out"], POINTER(POINTER(IUnknown)), "ppInterface"))]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], c_int, "dataFlow"), (["in"], DWORD, "dwStateMask"),
                               (["out"], POINTER(c_void_p), "ppDevices")),
                     COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], c_int, "dataFlow"), (["in"], c_int, "role"),
                               (["out"], POINTER(POINTER(IMMDevice)), "ppEndpoint"))]
    comtypes.CoInitialize()
    enum = comtypes.CoCreateInstance(GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"), IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER)
    dev = enum.GetDefaultAudioEndpoint(0, 1)                     # eRender, eMultimedia
    unk = dev.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_INPROC_SERVER, None)
    return unk.QueryInterface(IAudioEndpointVolume)


def _core_get() -> Optional[int]:
    if not sys.platform.startswith("win"):
        return None
    try:
        return int(round(_endpoint().GetMasterVolumeLevelScalar() * 100))
    except Exception:
        logger.debug("reading the volume failed", exc_info=True)
        return None


def _core_set(n: int) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        _endpoint().SetMasterVolumeLevelScalar(max(0.0, min(1.0, n / 100.0)), None)
        return True
    except Exception:
        logger.debug("setting the volume failed", exc_info=True)
        return False


# ----------------------------------------------------------------------------------------------------------------------------- diagnostic
_DIAG = re.compile(_A + r"(?:run\s+(?:a\s+|some\s+|the\s+)?(?:diagnostics?|self[\s-]?(?:check|test|diagnostics?)|system\s+check|health\s+check|check-?up)|"
                   r"(?:do|give\s+me)\s+a\s+(?:self[\s-]?check|system\s+check|health\s+check|diagnostic)|self[\s-]?(?:check|test)|diagnose\s+yourself|"
                   r"(?:are|is)\s+(?:all\s+)?(?:your\s+systems|everything)\s+(?:ok(?:ay)?|working|running|fine|all\s+right|good)(?:\s+with\s+you)?|"
                   r"how\s+are\s+your\s+systems(?:\s+doing)?|system\s+status|status\s+report)" + _Z, re.I)


def is_diagnostic(text: str) -> bool:
    return bool(_DIAG.match((text or "").strip()))


def diagnose(rt: Any, now: Optional[float] = None) -> str:
    """A few lines from what the runtime holds right now, and what would help. Read-only."""
    now = time.time() if now is None else now
    lines: List[str] = []
    fixes: List[str] = []
    per = getattr(rt, "periphery", None)
    sensors = getattr(per, "sensors", {}) or {}
    mic = sensors.get("microphone") if hasattr(sensors, "get") else None
    if mic is not None:
        state = str(getattr(getattr(mic, "state", None), "value", "") or "")
        conf = getattr(mic, "_last_asr_conf", None)
        heard = "" if conf is None else (" -- the last thing I heard, I was sure of" if conf >= -0.4 else " -- I wasn't sure of the last thing I heard")
        lines.append(f"Microphone: {state or 'present'}{heard}.")
    else:
        lines.append("Microphone: not running.")
    llm = getattr(getattr(rt, "ash", None), "llm", None)
    model = str(getattr(llm, "_ollama_model", "") or "")
    slow = None
    try:
        rows = list(getattr(getattr(rt, "unrouted", None), "rows", []) or [])[-10:]
        took = [float(r.get("took_s") or 0) for r in rows if r.get("took_s")]
        slow = (sum(took) / len(took)) if took else None
    except Exception:
        pass
    if model:
        s = f"Language model: {model}"
        if getattr(rt, "_model_unloaded", False):
            s += ", not loaded right now (the next question loads it)"
        if slow is not None:
            s += f", recent answers about {slow:.0f} s"
        lines.append(s + ".")
        if slow is not None and slow >= 30:
            fixes.append("a smaller model in the settings would answer faster")
    else:
        lines.append("Language model: none -- the rules answer what they can.")
    import os as _os
    if _os.environ.get("ASH_OLLAMA_STARTED"):
        lines.append("I started Ollama myself at launch; it wasn't running.")
    g = getattr(rt, "governor", None)
    mem, cpu = getattr(g, "mem", None), getattr(g, "cpu", None)
    if mem is not None:
        lines.append(f"Memory {float(mem):.0f}%" + (f", CPU {float(cpu):.0f}%" if cpu is not None else "") + ".")
        if float(mem) >= 88:
            top = ""
            try:
                from . import machine
                m = machine.Machine()
                by: Dict[str, int] = {}
                for name, rss, pid in m.procs():
                    if pid != m.own_pid:
                        n = machine._name(name)
                        by[n] = by.get(n, 0) + rss
                if by:
                    n, b = max(by.items(), key=lambda kv: kv[1])
                    top = f"closing {n} would free about {machine._gb(b)}"
            except Exception:
                pass
            fixes.append(top or "closing a big program would free memory")
    try:
        on = [s.name for s in per.active_sensors()] if per is not None and hasattr(per, "active_sensors") else []
        if on:
            lines.append("On: " + ", ".join(sorted(set(on))[:8]) + ".")
    except Exception:
        pass
    errs = int((getattr(rt, "stats", {}) or {}).get("errors", 0) or 0)
    lines.append("No errors since I started." if not errs else f"{errs} internal error{'s' if errs != 1 else ''} since I started -- the log has them.")
    out = " ".join(lines)
    if fixes:
        out += " What would help: " + "; ".join(fixes) + "."
    return out
