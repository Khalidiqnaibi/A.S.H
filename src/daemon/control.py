"""
src/daemon/control.py

The keyboard, the mouse and the screen, by voice -- for whatever is in front of the person.

25 Nov, 14:xx: "play the video and stream", "just click space", "go full screen", "click the channel" -- ASH answered with talk. It controlled
the music keys and apps, but not the computer itself. This is the rest of it:

    keys        "press space / enter / escape / tab / delete", "hit control S", "click space"
    shortcuts   "copy", "paste", "undo", "select all", "save", "find", "refresh", "new tab", "close tab", "reopen the tab", "next tab",
                "switch to tab 3", "go back a page", "zoom in", "minimize", "maximize", "show the desktop", "switch window", "take a screenshot"
    full screen "full screen" (F for a video page, F11 elsewhere), "exit full screen"
    scrolling   "scroll down", "scroll up a lot", "page down", "go to the top / bottom"
    mouse       "click", "double click", "right click", "move the mouse left (a lot)", "click on <what is written on the screen>"
    typing      "type hello there"
    devices     "what's connected" (monitors, audio, Bluetooth, printers, cameras, drives, as the computer lists them)

Safe by construction, not by hope:
  * only a line meant for ASH, never a guest's voice, never while privacy is paused, never into a window on the privacy blocklist
  * a rate limit (a mis-heard loop must not type forty times)
  * the sharp ones ASK first and only a "yes" within a minute does them: Delete, Alt+F4 (close the window), typing more than a line
  * never: lock the computer, Run (Win+R), Ctrl+Alt+Delete, Task Manager, power -- those are not for a speech recogniser that can mishear
The machine is reached through a `ControlExecutor`, so tests hand in their own.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.control")

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|now|just|then)[,\s]+)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:go\s+ahead\s+and\s+)?"
_END = r"(?:[,\s]+(?:please|ash|now|for\s+me|there|here))*\s*[.!?]*\s*$"

# named keys -> the names the executor knows
KEYNAMES = {
    "space": "space", "space bar": "space", "spacebar": "space", "enter": "enter", "return": "enter", "escape": "esc", "esc": "esc", "tab": "tab",
    "backspace": "backspace", "delete": "delete", "up": "up", "down": "down", "left": "left", "right": "right", "up arrow": "up", "down arrow": "down",
    "left arrow": "left", "right arrow": "right", "page up": "pageup", "page down": "pagedown", "home": "home", "end": "end",
    **{f"f{i}": f"f{i}" for i in range(1, 13)}, "windows": "win", "windows key": "win",
}
_MODS = {"control": "ctrl", "ctrl": "ctrl", "shift": "shift", "alt": "alt", "windows": "win", "win": "win"}
_KEYRX = "|".join(sorted((re.escape(k) for k in KEYNAMES), key=len, reverse=True))
_MODRX = "|".join(sorted(_MODS, key=len, reverse=True))

# never, whatever is asked
_NEVER = {("ctrl", "alt", "delete"), ("win", "l"), ("win", "r"), ("ctrl", "shift", "esc"), ("ctrl", "esc"), ("win", "x"), ("alt", "ctrl", "delete")}
_ASK_FIRST = {("delete",), ("alt", "f4"), ("shift", "delete")}

# phrase -> (key combo, what is said). Order matters: the more specific first.
_SHORTCUTS: List[Tuple[str, Tuple[str, ...], str]] = [
    (r"(?:exit|leave|get\s+out\s+of|close)\s+full\s*screen", ("esc",), "Out of full screen."),
    (r"(?:copy|copy\s+(?:that|it|this|the\s+(?:text|selection)))", ("ctrl", "c"), "Copied."),
    (r"(?:paste|paste\s+(?:that|it|this|here))", ("ctrl", "v"), "Pasted."),
    (r"(?:cut|cut\s+(?:that|it|this))", ("ctrl", "x"), "Cut."),
    (r"undo(?:\s+(?:that|it|this))?", ("ctrl", "z"), "Undone."),
    (r"redo(?:\s+(?:that|it|this))?", ("ctrl", "y"), "Redone."),
    (r"select\s+all", ("ctrl", "a"), "Selected everything."),
    (r"save(?:\s+(?:it|this|that|the\s+(?:file|document)))?", ("ctrl", "s"), "Saved."),
    (r"(?:find|search)\s+(?:on|in)\s+(?:this\s+|the\s+)?(?:page|document|file|window)", ("ctrl", "f"), "Find is open."),
    (r"(?:refresh|reload)(?:\s+(?:the\s+)?(?:page|tab|window))?", ("f5",), "Refreshing."),
    (r"(?:open\s+)?(?:a\s+)?new\s+tab", ("ctrl", "t"), "New tab."),
    (r"close\s+(?:this\s+|the\s+|that\s+)?tab", ("ctrl", "w"), "Tab closed."),
    (r"(?:re-?open|bring\s+back)\s+(?:the\s+|that\s+|my\s+)?(?:last\s+|closed\s+)?(?:closed\s+)?tab", ("ctrl", "shift", "t"), "Reopened it."),
    (r"(?:next|right)\s+tab|go\s+to\s+the\s+next\s+tab", ("ctrl", "tab"), "Next tab."),
    (r"(?:previous|last|left)\s+tab|go\s+to\s+the\s+previous\s+tab", ("ctrl", "shift", "tab"), "Previous tab."),
    (r"(?:open\s+)?(?:a\s+)?new\s+window", ("ctrl", "n"), "New window."),
    (r"(?:go|navigate)\s+back(?:\s+a\s+page|\s+in\s+the\s+browser|\s+one\s+page)?|back\s+a\s+page|browser\s+back", ("alt", "left"), "Back."),
    (r"(?:go|navigate)\s+forward(?:\s+a\s+page|\s+in\s+the\s+browser|\s+one\s+page)?|forward\s+a\s+page|browser\s+forward", ("alt", "right"), "Forward."),
    (r"zoom\s+in", ("ctrl", "="), "Zooming in."),
    (r"zoom\s+out", ("ctrl", "-"), "Zooming out."),
    (r"(?:reset\s+(?:the\s+)?zoom|zoom\s+(?:back\s+)?to\s+(?:normal|100))", ("ctrl", "0"), "Zoom reset."),
    (r"minimi[sz]e(?:\s+(?:this|the|that)(?:\s+window)?)?", ("win", "down"), "Minimised."),
    (r"maximi[sz]e(?:\s+(?:this|the|that)(?:\s+window)?)?", ("win", "up"), "Maximised."),
    (r"(?:show|go\s+to)\s+(?:the\s+)?desktop", ("win", "d"), "Desktop."),
    (r"(?:switch|change)\s+(?:the\s+)?window|alt\s*tab|next\s+window", ("alt", "tab"), "Switching window."),
    (r"(?:take\s+a\s+)?screenshot(?:\s+of\s+(?:this|the\s+screen))?|snip\s+(?:the\s+)?screen", ("win", "shift", "s"), "Pick the area to capture."),
    (r"(?:show\s+)?task\s+view|show\s+all\s+(?:my\s+)?windows", ("win", "tab"), "All windows."),
    (r"close\s+(?:this|the|that)\s+window", ("alt", "f4"), "Closed the window."),
]
_SHORTCUT_RX = [(re.compile(_LEAD + r"(?:" + p + r")" + _END, re.I), k, s) for p, k, s in _SHORTCUTS]

_FULLSCREEN = re.compile(_LEAD + r"(?:(?:go|make\s+(?:it|this|that|the\s+(?:video|page|screen)))\s+)?(?:to\s+|in\s+|into\s+)?full\s*[-\s]?screen(?:\s+(?:mode|please))?" + _END, re.I)
_VIDEO_TITLE = re.compile(r"youtube|netflix|stremio|twitch|vimeo|prime\s+video|disney|vlc|media\s+player|mpv|plex|crunchyroll|hulu|kick", re.I)

_PRESS = re.compile(_LEAD + r"(?:press|hit|tap|push|click|type)\s+(?:the\s+)?(?:(?P<mods>(?:(?:" + _MODRX + r")(?:\s*\+\s*|\s+))+))?(?P<key>" + _KEYRX + r"|[a-z0-9])(?:\s+key)?(?:\s+(?P<n>\d{1,2}|twice|two\s+times|three\s+times|once))?" + _END, re.I)
_TABN = re.compile(_LEAD + r"(?:switch|go|jump)\s+to\s+tab\s+(?P<n>[1-8]|one|two|three|four|five|six|seven|eight)" + _END, re.I)
_SCROLL = re.compile(_LEAD + r"scroll\s+(?P<d>up|down)(?:\s+(?P<amt>a\s+(?:bit|little|lot)|(?:a\s+)?(?:little|bit)|(?:a\s+)?lot|\d{1,2}(?:\s+times)?|(?:two|three|four|five)\s+times|all\s+the\s+way))?" + _END, re.I)
_TOPBOTTOM = re.compile(_LEAD + r"(?:go\s+to|jump\s+to|scroll\s+to|take\s+me\s+to)\s+the\s+(?P<w>top|bottom)(?:\s+of\s+(?:the\s+)?(?:page|document|file|list))?" + _END, re.I)
_CLICK = re.compile(_LEAD + r"(?P<k>double[-\s]?click|right[-\s]?click|left[-\s]?click|middle[-\s]?click|click)(?:\s+(?:here|there|it|that|this|again|once|twice))?" + _END, re.I)
_CLICK_ON = re.compile(_LEAD + r"(?:click|press|tap|select|choose|open)\s+(?:on\s+)?(?:the\s+)?(?P<t>[\w][\w '\-./:&]{1,50}?)(?:\s+(?:button|link|tab|icon|option|menu|text))?" + _END, re.I)
_MOVE = re.compile(_LEAD + r"move\s+(?:the\s+)?(?:mouse|cursor|pointer)\s+(?P<d>up|down|left|right)(?:\s+(?P<amt>a\s+(?:bit|little|lot)|(?:a\s+)?(?:little|bit)|(?:a\s+)?lot|\d{1,4}(?:\s+pixels)?))?" + _END, re.I)
_CENTER = re.compile(_LEAD + r"(?:move\s+(?:the\s+)?(?:mouse|cursor|pointer)\s+to\s+(?:the\s+)?(?:centre|center|middle)|(?:centre|center)\s+(?:the\s+)?(?:mouse|cursor|pointer))" + _END, re.I)
_TYPE = re.compile(_LEAD + r"type(?:\s+out)?\s+(?P<t>.{1,200}?)\s*$", re.I)
_DEVICES = re.compile(_LEAD + r"(?:what(?:'s|\s+is|\s+are)\s+(?:devices\s+)?(?:connected|plugged\s+in|attached|hooked\s+up)(?:\s+to\s+(?:this|the|my)\s+(?:computer|pc|laptop|machine))?|"
                      r"(?:what|which)\s+devices\s+(?:are\s+)?(?:connected|plugged\s+in|attached)(?:\s+to\s+(?:this|the|my)\s+(?:computer|pc|laptop|machine))?|"
                      r"list\s+(?:the\s+|my\s+)?(?:connected\s+)?devices|(?:show|tell)\s+me\s+(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:connected\s+)?devices(?:\s+connected)?)" + _END, re.I)
_YES = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:yes|yeah|yep|yup|sure|ok(?:ay)?|do\s+it|go\s+ahead|confirm(?:ed)?|please\s+do|affirmative)\b", re.I)
_NO = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:no|nope|nah|don'?t|cancel|stop|leave\s+it|never\s*mind)\b", re.I)

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "once": 1, "twice": 2}
_NOT_A_CLICK_TARGET = re.compile(r"^(?:it|that|this|here|there|space|enter|escape|esc|tab|delete|backspace|the\s+screen|my\s+(?:files?|notes?|memory)|everything|something|anything|"
                                 r"me|you|ash|yes|no|ok|okay|up|down|left|right|home|end|[a-z]|f\d{1,2}|play|pause|stop|next|previous|mute|settings?|memory)$", re.I)


@dataclass
class Action:
    kind: str                           # keys | mouse | scroll | move | type | devices | find_click
    keys: Tuple[str, ...] = ()
    repeat: int = 1
    arg: Any = None
    say: str = ""
    ask: str = ""                       # set: the line to ask first


def _count(s: Optional[str]) -> int:
    if not s:
        return 1
    s = s.strip().lower()
    if s.isdigit():
        return max(1, min(int(s), 20))
    if s.endswith("times"):
        w = s.split()[0]
        return _NUM.get(w, int(w) if w.isdigit() else 1)
    return _NUM.get(s, 1)


def parse(text: str, title: str = "") -> Optional[Action]:
    """What a spoken line asks the computer to do, or None. `title` is the window in front (a video page takes 'f' for full screen)."""
    t = (text or "").strip()
    if not t or len(t) > 220:
        return None
    if _DEVICES.match(t):
        return Action("devices")
    m = _FULLSCREEN.match(t)
    if m:
        return Action("keys", ("f",) if _VIDEO_TITLE.search(title or "") else ("f11",), say="Full screen.")
    m = _TABN.match(t)
    if m:
        n = m.group("n").lower()
        return Action("keys", ("ctrl", str(_NUM.get(n, n))), say=f"Tab {_NUM.get(n, n)}.")
    m = _PRESS.match(t)
    if m and len(m.group("key")) == 1 and not (m.group("mods") or "").strip() and re.match(_LEAD + r"type\s", t, re.I):
        m = None                                            # "type a" is typing a letter, not pressing a key (below)
    if m:
        key = KEYNAMES.get(m.group("key").lower(), m.group("key").lower())
        mods = tuple(_MODS[x.lower()] for x in re.findall(_MODRX, m.group("mods") or "", re.I))
        combo = mods + (key,)
        if tuple(sorted(combo)) in {tuple(sorted(x)) for x in _NEVER} or tuple(combo) in _NEVER:
            return Action("refuse", combo, say="I don't press that one by voice -- it's too easy to mishear. Do it yourself.")
        nice = " ".join(combo)
        a = Action("keys", combo, repeat=_count(m.group("n")), say=f"Pressed {nice}.")
        if tuple(combo) in _ASK_FIRST:
            a.ask = f"Press {nice}? That can delete things. Say yes."
        return a
    for rx, keys, say in _SHORTCUT_RX:
        if rx.match(t):
            a = Action("keys", keys, say=say)
            if keys in _ASK_FIRST:
                a.ask = "Close this window? Anything unsaved may be lost. Say yes."
            return a
    m = _TOPBOTTOM.match(t)
    if m:
        return Action("keys", ("ctrl", "home" if m.group("w").lower() == "top" else "end"), say="Top." if m.group("w").lower() == "top" else "Bottom.")
    m = _SCROLL.match(t)
    if m:
        amt = (m.group("amt") or "").lower()
        n = 10 if "all the way" in amt else 8 if "lot" in amt else 2 if ("bit" in amt or "little" in amt) else _count(amt) * 3 if amt else 4
        return Action("scroll", repeat=n if m.group("d").lower() == "up" else -n, say=f"Scrolling {m.group('d').lower()}.")
    m = _CENTER.match(t)
    if m:
        return Action("move", arg="center", say="Mouse in the middle.")
    m = _MOVE.match(t)
    if m:
        amt = (m.group("amt") or "").lower()
        px = 400 if "lot" in amt else 40 if ("bit" in amt or "little" in amt) else int(re.sub(r"\D", "", amt)) if re.search(r"\d", amt) else 150
        dx, dy = {"up": (0, -px), "down": (0, px), "left": (-px, 0), "right": (px, 0)}[m.group("d").lower()]
        return Action("move", arg=(dx, dy), say=f"Mouse {m.group('d').lower()}.")
    m = _CLICK.match(t)
    if m:
        k = re.sub(r"[-\s]", "", m.group("k").lower())
        return Action("mouse", arg=k, say={"doubleclick": "Double click.", "rightclick": "Right click.", "middleclick": "Middle click."}.get(k, "Click."))
    m = _TYPE.match(t)
    if m:
        what = re.sub(r"^(?:the\s+)?(?:words?|text|letters?|phrase|following)\s+", "", m.group("t").strip(), flags=re.I)
        a = Action("type", arg=what, say="Typed it.")
        if len(what) > 60:
            a.ask = f"Type {len(what)} characters into the window in front? Say yes."
        return a
    m = _CLICK_ON.match(t)
    if m:
        target = m.group("t").strip(" .,'\"")
        if not _NOT_A_CLICK_TARGET.match(target) and len(target.split()) <= 5 and re.match(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:click|press|tap|select|choose)\b", t, re.I):
            return Action("find_click", arg=target)
    return None


class ControlExecutor:
    """Does it on Windows (ctypes, no extra package). Every method returns True if it worked."""

    _VK = {"space": 0x20, "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "backspace": 0x08, "delete": 0x2E, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
           "pageup": 0x21, "pagedown": 0x22, "home": 0x24, "end": 0x23, "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "=": 0xBB, "-": 0xBD,
           **{f"f{i}": 0x6F + i for i in range(1, 13)}}
    _EXT = {"up", "down", "left", "right", "pageup", "pagedown", "home", "end", "delete", "win"}

    def __init__(self):
        self._ok = sys.platform.startswith("win")
        if self._ok:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _vk(self, k: str) -> int:
        if k in self._VK:
            return self._VK[k]
        if len(k) == 1:
            import ctypes
            return ctypes.windll.user32.VkKeyScanW(ord(k)) & 0xFF
        return 0

    def keys(self, combo: Tuple[str, ...]) -> bool:
        if not self._ok:
            return False
        try:
            import ctypes
            u = ctypes.windll.user32
            codes = [(self._vk(k), k in self._EXT) for k in combo]
            if not all(c for c, _ in codes):
                return False
            for c, ext in codes:
                u.keybd_event(c, 0, 1 if ext else 0, 0)
            for c, ext in reversed(codes):
                u.keybd_event(c, 0, (1 if ext else 0) | 2, 0)
            return True
        except Exception:
            logger.exception("Key press failed")
            return False

    def type_text(self, text: str) -> bool:
        if not self._ok:
            return False
        try:
            import ctypes
            from ctypes import wintypes

            class KI(ctypes.Structure):
                _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                            ("dwExtraInfo", ctypes.c_size_t)]

            class MI(ctypes.Structure):
                _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                            ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

            class U(ctypes.Union):
                _fields_ = [("ki", KI), ("mi", MI)]

            class IN(ctypes.Structure):
                _fields_ = [("type", wintypes.DWORD), ("u", U)]

            for ch in text[:300]:
                for up in (0, 2):
                    i = IN(1, U(ki=KI(0, ord(ch), 4 | up, 0, 0)))          # KEYEVENTF_UNICODE
                    ctypes.windll.user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(IN))
                time.sleep(0.004)
            return True
        except Exception:
            logger.exception("Typing failed")
            return False

    def click(self, kind: str = "click", at: Optional[Tuple[int, int]] = None) -> bool:
        if not self._ok:
            return False
        try:
            import ctypes
            u = ctypes.windll.user32
            if at is not None:
                u.SetCursorPos(int(at[0]), int(at[1]))
            down, up = {"rightclick": (0x08, 0x10), "middleclick": (0x20, 0x40)}.get(kind, (0x02, 0x04))
            for _ in range(2 if kind == "doubleclick" else 1):
                u.mouse_event(down, 0, 0, 0, 0)
                u.mouse_event(up, 0, 0, 0, 0)
                time.sleep(0.05)
            return True
        except Exception:
            logger.exception("Click failed")
            return False

    def move(self, arg: Any) -> bool:
        if not self._ok:
            return False
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            if arg == "center":
                u.SetCursorPos(int(u.GetSystemMetrics(0) // 2), int(u.GetSystemMetrics(1) // 2))
                return True
            p = wintypes.POINT()
            u.GetCursorPos(ctypes.byref(p))
            w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
            u.SetCursorPos(max(0, min(w - 1, p.x + int(arg[0]))), max(0, min(h - 1, p.y + int(arg[1]))))
            return True
        except Exception:
            logger.exception("Mouse move failed")
            return False

    def scroll(self, clicks: int) -> bool:
        if not self._ok:
            return False
        try:
            import ctypes
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(clicks) * 120, 0)
            return True
        except Exception:
            logger.exception("Scroll failed")
            return False

    def find_text(self, phrase: str) -> Optional[Tuple[int, int]]:
        """Where `phrase` is written on the screen (the centre of the best matching line), by the screen reader's OCR; None if it can't be found."""
        try:
            import numpy as np
            from PIL import ImageGrab
            from src.senses.screenread import ScreenReader
            rd = getattr(self, "_reader", None)
            if rd is None:
                rd = self._reader = ScreenReader()
            if rd.backend != "rapidocr":
                return None
            if rd._engine is None:
                rd._engine = rd._build_rapidocr()
            img = ImageGrab.grab()
            result, _ = rd._engine(np.asarray(img))
        except Exception:
            logger.exception("Looking for text on the screen failed")
            return None
        want = re.sub(r"[^a-z0-9 ]", "", phrase.lower()).split()
        best, where = 0.0, None
        for r in (result or []):
            try:
                box, text = r[0], str(r[1])
                words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
                if not want or not words:
                    continue
                hit = sum(1 for w in want if w in words) / len(want)
                exact = 1.0 if " ".join(want) in " ".join(words) else 0.0
                score = max(hit, exact) - 0.01 * max(0, len(words) - len(want))
                if score > best and (hit >= 1.0 or exact):
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    best, where = score, (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            except Exception:
                continue
        return where

    def devices(self) -> List[Tuple[str, str]]:
        """(kind, name) for what the computer lists as present: monitors, audio, Bluetooth, printers, cameras, drives."""
        if not self._ok:
            return []
        ps = ("Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.Class -in "
              "'Monitor','AudioEndpoint','Bluetooth','Camera','Image','Printer','DiskDrive','HIDClass','Mouse','Keyboard','USB' -and $_.Status -eq 'OK' } | "
              "ForEach-Object { $_.Class + '|' + $_.FriendlyName }")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20).stdout
        except Exception:
            logger.exception("Listing devices failed")
            return []
        seen, rows = set(), []
        for ln in out.splitlines():
            if "|" not in ln:
                continue
            c, n = ln.split("|", 1)
            n = n.strip()
            if n and (c, n) not in seen and not re.search(r"root hub|composite|generic usb hub|host controller|usb input device|hid-compliant", n, re.I):
                seen.add((c, n))
                rows.append((c, n))
        return rows


_KIND_WORDS = {"Monitor": "screens", "AudioEndpoint": "audio", "Bluetooth": "Bluetooth", "Camera": "cameras", "Image": "cameras", "Printer": "printers",
               "DiskDrive": "drives", "Mouse": "mice", "Keyboard": "keyboards", "HIDClass": "other input", "USB": "USB"}


def devices_say(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return "I can't see a list of connected devices from here."
    groups: Dict[str, List[str]] = {}
    for c, n in rows:
        groups.setdefault(_KIND_WORDS.get(c, c.lower()), []).append(n)
    parts = []
    for k in ("screens", "audio", "Bluetooth", "cameras", "printers", "drives", "mice", "keyboards", "other input", "USB"):
        if k in groups:
            names = groups[k][:4]
            parts.append(f"{k}: " + ", ".join(names) + (f" and {len(groups[k]) - 4} more" if len(groups[k]) > 4 else ""))
    return "Connected: " + "; ".join(parts) + "."


@dataclass
class ControlConfig:
    enabled: bool = True
    max_per_minute: int = 40
    confirm_window_s: float = 60.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ControlConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k == "enabled":
                c.enabled = bool(v)
            elif k in ("max_per_minute", "confirm_window_s"):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


class Control:
    def __init__(self, cfg: Optional[ControlConfig] = None, executor: Optional[ControlExecutor] = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or ControlConfig()
        self.ex = executor or ControlExecutor()
        self.clock = clock
        self._pending: Optional[Tuple[Action, float]] = None
        self._recent: List[float] = []
        self.stats = {"done": 0, "refused": 0, "asked": 0, "failed": 0}

    @property
    def waiting(self) -> bool:
        return self._pending is not None

    def _rate_ok(self, now: float) -> bool:
        self._recent = [t for t in self._recent if now - t < 60.0]
        if len(self._recent) >= self.cfg.max_per_minute:
            return False
        self._recent.append(now)
        return True

    def hear(self, text: str, now: Optional[float] = None, title: str = "", blocked: bool = False, privacy_paused: bool = False) -> Optional[str]:
        """What to say back for a computer-control line, or None if it is not one."""
        if not self.cfg.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        p = self._pending
        if p is not None:
            self._pending = None
            if now - p[1] <= self.cfg.confirm_window_s:
                if _YES.match(t):
                    return self._do(p[0], now, title, blocked, privacy_paused, confirmed=True)
                if _NO.match(t):
                    return "Okay, not doing it."
        a = parse(t, title)
        if a is None:
            return None
        if a.kind == "devices":
            return devices_say(self.ex.devices())
        if a.kind == "refuse":
            self.stats["refused"] += 1
            return a.say
        if privacy_paused:
            return "I'm paused for privacy, so I'm not touching the keyboard or mouse."
        if a.ask:
            if blocked:
                return "That window is on the privacy blocklist, so I won't type or click in it."
            self._pending = (a, now)
            self.stats["asked"] += 1
            return a.ask
        return self._do(a, now, title, blocked, privacy_paused)

    def _do(self, a: Action, now: float, title: str, blocked: bool, privacy_paused: bool, confirmed: bool = False) -> str:
        if blocked:
            self.stats["refused"] += 1
            return "That window is on the privacy blocklist, so I won't type or click in it."
        if not self._rate_ok(now):
            self.stats["refused"] += 1
            return "That's a lot of key presses in a minute -- I've stopped for a moment in case I misheard something."
        ok = False
        if a.kind == "keys":
            ok = all(self.ex.keys(a.keys) for _ in range(max(1, a.repeat)))
        elif a.kind == "type":
            ok = self.ex.type_text(str(a.arg))
        elif a.kind == "mouse":
            ok = self.ex.click(str(a.arg))
        elif a.kind == "move":
            ok = self.ex.move(a.arg)
        elif a.kind == "scroll":
            ok = self.ex.scroll(a.repeat)
        elif a.kind == "find_click":
            where = self.ex.find_text(str(a.arg))
            if where is None:
                self.stats["failed"] += 1
                return f"I can't see “{a.arg}” on the screen."
            ok = self.ex.click("click", where)
            a.say = f"Clicked {a.arg}."
        self.stats["done" if ok else "failed"] += 1
        return a.say if ok else "I couldn't do that -- I couldn't reach the keyboard or mouse."

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "waiting": self.waiting, **self.stats}
