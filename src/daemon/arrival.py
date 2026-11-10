"""
src/daemon/arrival.py

"Remind me when I get home to water the plants."

ASH can't see a map, so "home" is a network: the Wi-Fi name (or a host on the
LAN) that is only reachable at home. Say, while connected there:

    this is home / home is this wifi / remember this network as home

and then:

    remind me when I get home to <thing>      -> kept, said the moment the laptop
                                                 joins the home network after
                                                 having been away
    what's waiting for me at home?            -> lists them
    forget my home wifi / cancel home reminders

"Away" means the home network was not seen for a few minutes in a row, so a
dropped connection does not count as leaving. Rules and the network name only;
no model, no location services, nothing leaves the machine. Kept under state/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"\s*[.!]*\s*$"
_WHEN = re.compile(_A + r"remind me\s+(?:(?:when|as soon as|once|after)\s+(?:i\s+)?(?:get|got|come|arrive|am|reach|walk in)\s+(?:back\s+)?(?:at\s+|to\s+|in\s+)?home"
                   r"|(?:when|once)\s+(?:i'?m|i\s+am)\s+(?:back\s+)?(?:at\s+)?home|on\s+(?:getting|arriving|coming)\s+home)"
                   r"[,\s]*(?:to\s+|that\s+|about\s+)?(?P<what>.*?)" + _Z, re.I)
_WHEN_END = re.compile(_A + r"remind me\s+(?:to\s+)?(?P<what>.+?)\s+(?:when|once|as soon as)\s+(?:i\s+)?(?:get|got|come|arrive|reach)\s+(?:back\s+)?home" + _Z, re.I)
_SET = re.compile(_A + r"(?:this\s+is\s+(?:my\s+)?home(?:\s+(?:wifi|network|wi-fi))?|home\s+is\s+(?:this|here)(?:\s+(?:wifi|network|wi-fi))?|"
                  r"(?:remember|save|set)\s+(?:this|the current)\s+(?:wifi|network|wi-fi)\s+as\s+(?:my\s+)?home|i'?m\s+(?:at\s+)?home\s+now)" + _Z, re.I)
_FORGET = re.compile(_A + r"(?:forget|clear|remove)\s+(?:my\s+)?home\s+(?:wifi|network|wi-fi)" + _Z, re.I)
_LIST = re.compile(_A + r"(?:what(?:'s|\s+is)?|list|any)\s+(?:is\s+)?(?:waiting|pending|left)?\s*(?:for me\s+)?(?:at|when i get)\s+home" + _Z
                   + r"|" + _A + r"(?:what|which)\s+home\s+reminders(?:\s+do\s+i\s+have)?" + _Z, re.I)
_CANCEL = re.compile(_A + r"(?:cancel|clear|delete)\s+(?:all\s+)?(?:my\s+)?home\s+reminders?" + _Z, re.I)


def current_ssid() -> str:
    """The Wi-Fi network this machine is on now ('' if none or unknown)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True,
                                 timeout=4, creationflags=0x08000000).stdout
            m = re.search(r"^\s*SSID\s*:\s*(.+?)\s*$", out, re.M)
            return m.group(1).strip() if m else ""
        out = subprocess.run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], capture_output=True, text=True, timeout=4).stdout
        for line in out.splitlines():
            if line.startswith("yes:"):
                return line[4:].strip()
    except Exception:
        pass
    return ""


class Arrivals:
    def __init__(self, path: Optional[str] = None, ssid: Callable[[], str] = current_ssid,
                 away_after_s: float = 180.0, enabled: bool = True, home: str = ""):
        self.path = path
        self.ssid = ssid
        self.away_after_s = away_after_s
        self.enabled = enabled
        self._lock = threading.RLock()
        self.home = home.strip()
        self.items: List[Dict[str, object]] = []        # {"what", "created", "done"}
        self._last_home_seen: Optional[float] = None
        self._away = False
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.home = str(raw.get("home") or self.home)
            self.items = [i for i in raw.get("items", []) if not i.get("done")]
        except Exception:
            self.items = []

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"home": self.home, "items": self.items}, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- requests --------------------------------------------------------------
    def pending(self) -> List[Dict[str, object]]:
        return [i for i in self.items if not i.get("done")]

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.enabled:
            return None
        t = (text or "").strip()
        if not t or len(t) > 160:
            return None
        now = time.time() if now is None else now
        if _SET.match(t):
            name = self.ssid()
            if not name:
                return "I can't tell which Wi-Fi this is -- I need to be on one to call it home."
            with self._lock:
                self.home = name
                self._away = False
                self._last_home_seen = now
                self.save()
            return f"Okay, “{name}” is home."
        if _FORGET.match(t):
            with self._lock:
                self.home = ""
                self.save()
            return "I've forgotten which network is home."
        if _CANCEL.match(t):
            with self._lock:
                n = len(self.pending())
                self.items = []
                self.save()
            return "Cancelled your home reminders." if n else "You have no home reminders."
        if _LIST.match(t):
            p = self.pending()
            if not p:
                return "Nothing is waiting for you at home."
            return "At home: " + "; ".join(str(i["what"]) for i in p[:6]) + "."
        m = _WHEN.match(t) or _WHEN_END.match(t)
        if m:
            what = re.sub(r"\s+", " ", (m.group("what") or "")).strip(" ,.")
            what = re.sub(r"^(?:to|that|about)\s+", "", what, flags=re.I)
            if not what:
                return "What should I remind you about when you get home?"
            with self._lock:
                self.items.append({"what": what[:120], "created": now, "done": False})
                self.save()
            if not self.home:
                return (f"I'll remind you to {what} -- but I don't know which network is home yet. "
                        f"Say “this is home” while you're on it.")
            return f"Okay, I'll remind you to {what} when you get home."
        return None

    # -- the arrival -----------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> List[str]:
        """Called now and then; returns reminders to say if you just got home."""
        if not self.enabled or not self.home:
            return []
        now = time.time() if now is None else now
        try:
            here = (self.ssid() or "").strip().lower() == self.home.lower()
        except Exception:
            return []
        out: List[str] = []
        with self._lock:
            if here:
                if self._away:
                    due = self.pending()
                    for i in due:
                        i["done"] = True
                        out.append(f"Welcome home. Reminder: {i['what']}.")
                    if due:
                        self.items = [i for i in self.items if not i.get("done")]
                        self.save()
                self._away = False
                self._last_home_seen = now
            elif self._last_home_seen is None or now - self._last_home_seen >= self.away_after_s:
                self._away = True
        return out

    def status(self) -> Dict[str, object]:
        return {"home": bool(self.home), "away": self._away, "waiting": len(self.pending())}
