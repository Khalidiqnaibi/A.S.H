"""
src/daemon/phone.py

Your phone, from ASH -- through KDE Connect on the local network, and (for alerts only) an ntfy address you choose.

    find my phone / ring my phone                   it rings, even on silent (KDE Connect's "find my phone")
    send "buy milk" to my phone / note to my phone: buy milk     a notification on the phone
    send it to my phone                             the file ASH just found (file search), or the link it just gave
    text Omar saying I'm late / text 0791234567 saying ...   read back first; sent only on "send it"
                                                    (Omar's number from phone.contacts, or one you say)
    is my phone connected?

Guard mode (guard.py) uses notify() to reach you when you are away.

What it cannot do on Windows: hear an incoming call or read the phone's notifications -- KDE Connect's command-line
tool cannot listen, only send. Nothing here works until KDE Connect is installed and paired (kdeconnect.kde.org; the
phone app is "KDE Connect" on Android). ntfy: set phone.ntfy_url to your own server's topic (or a ntfy.sh topic --
that one is a public service, so only short alert texts go there, never files).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you)[,\s]+)*"
_END = r"\s*(?:please|now|for\s+me)?\s*[.!?]*\s*$"
_RING = re.compile(_LEAD + r"(?:find|ring|locate|where'?s|where\s+is|call)\s+my\s+(?:phone|mobile|cell(?:\s*phone)?)" + _END, re.I)
_NOTE = re.compile(_LEAD + r"(?:send\s+(?P<q>[\"\u201c](?P<text>.+?)[\"\u201d])\s+to\s+my\s+phone|"
                   r"(?:send\s+a\s+)?(?:note|reminder|message)\s+to\s+my\s+phone(?:\s+saying)?[:,]?\s+(?P<text2>.+?)|"
                   r"(?:put|send)\s+(?P<text3>.+?)\s+(?:on|to)\s+my\s+phone\s+as\s+a\s+note)" + _END, re.I)
_SHARE = re.compile(_LEAD + r"(?:send|share|put)\s+(?:it|that|this|the\s+file|that\s+file|the\s+link|that\s+link)\s+(?:to|on)\s+my\s+phone" + _END, re.I)
_TEXT = re.compile(_LEAD + r"(?:text|sms|send\s+(?:a\s+)?(?:text|sms)\s+to|message)\s+(?P<to>\+?[\d\s()-]{6,20}?|[a-z][a-z'\- ]{1,30}?)"
                   r"(?:\s*,\s*|\s+)(?:saying|that|to\s+say)\s+(?P<text>.+?)\s*[.!]?\s*$", re.I)
_SEND = re.compile(_LEAD + r"(?:(?:yes|yeah|ok(?:ay)?)[,\s]+)?(?:go\s+ahead\s+and\s+)?send\s+(?:it|that|the\s+text)(?:\s+now)?\s*[.!]?\s*$", re.I)
_CANCEL = re.compile(_LEAD + r"(?:don'?t\s+send\s+(?:it|that)|cancel\s+(?:it|that|the\s+text))\s*[.!]?\s*$", re.I)
_STATUS = re.compile(_LEAD + r"(?:is\s+my\s+phone\s+(?:connected|there|paired|reachable)|can\s+you\s+(?:see|reach)\s+my\s+phone)" + _END, re.I)


def find_cli() -> Optional[str]:
    p = shutil.which("kdeconnect-cli")
    if p:
        return p
    for c in (r"C:\Program Files\KDE Connect\bin\kdeconnect-cli.exe", r"C:\Program Files (x86)\KDE Connect\bin\kdeconnect-cli.exe"):
        if os.path.exists(c):
            return c
    return None


def run(args: Sequence[str], timeout: float = 10.0) -> Tuple[int, str]:
    """Run the KDE Connect tool: (exit code, output). Never through a shell."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout, creationflags=flags)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ntfy_post(url: str, text: str, title: str = "ASH", timeout: float = 10.0) -> bool:
    req = urllib.request.Request(url, data=text.encode("utf-8"), method="POST",
                                 headers={"Title": title, "Tags": "robot", "User-Agent": "ASH-local-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return 200 <= r.status < 300


@dataclass
class PhoneConfig:
    enabled: bool = True
    device: str = ""                         # a KDE Connect device id or name; empty: the first reachable one
    contacts: Dict[str, str] = None          # {"Omar": "+962791234567"}
    ntfy_url: str = ""                       # alerts only
    allow_sms: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PhoneConfig":
        d = d or {}
        return cls(enabled=bool(d.get("enabled", True)), device=str(d.get("device", "") or ""),
                   contacts={str(k): str(v) for k, v in (d.get("contacts") or {}).items()},
                   ntfy_url=str(d.get("ntfy_url", "") or ""), allow_sms=bool(d.get("allow_sms", True)))


class Phone:
    NOT_SET_UP = ("I can't reach your phone: KDE Connect isn't installed or paired here. Install KDE Connect on this "
                  "computer and the KDE Connect app on your phone, pair them on the same Wi-Fi, and ask me again.")

    def __init__(self, cfg: Optional[PhoneConfig] = None, cli: Optional[str] = "auto",
                 runner: Callable[[Sequence[str], float], Tuple[int, str]] = run,
                 poster: Callable[[str, str], bool] = lambda url, text: ntfy_post(url, text),
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or PhoneConfig.from_dict({})
        self.cli = find_cli() if cli == "auto" else cli
        self.runner = runner
        self.poster = poster
        self.clock = clock
        self.pending_sms: Optional[Tuple[str, str, str, float]] = None      # (name, number, text, when)
        self.stats = {"rang": 0, "notified": 0, "shared": 0, "texted": 0, "alerts": 0, "failed": 0}

    # -- the device ----------------------------------------------------------------------------------------
    def _device(self) -> Optional[str]:
        if not self.cli:
            return None
        if self.cfg.device:
            return self.cfg.device
        try:
            code, out = self.runner([self.cli, "--list-available", "--id-only"], 10.0)
        except Exception:
            return None
        ids = [ln.strip() for ln in out.splitlines() if ln.strip() and code == 0]
        return ids[0] if ids else None

    def _do(self, *args: str) -> bool:
        dev = self._device()
        if dev is None:
            return False
        flag = "-n" if (self.cfg.device and not re.fullmatch(r"[0-9a-f_]{8,}", self.cfg.device)) else "-d"
        try:
            code, _out = self.runner([self.cli, flag, dev, *args], 15.0)
        except Exception:
            code = 1
        if code != 0:
            self.stats["failed"] += 1
        return code == 0

    @property
    def ready(self) -> bool:
        return bool(self.cfg.enabled and self.cli)

    def ring(self) -> bool:
        ok = self._do("--ring")
        self.stats["rang"] += int(ok)
        return ok

    def notify(self, text: str, alert: bool = False) -> bool:
        """A notification on the phone. Alerts (guard mode) may also go to the ntfy address; notes never do."""
        ok = self._do("--ping-msg", text[:300]) if self.cli else False
        if not ok and alert and self.cfg.ntfy_url:
            try:
                ok = bool(self.poster(self.cfg.ntfy_url, text[:300]))
            except Exception:
                ok = False
        if ok:
            self.stats["alerts" if alert else "notified"] += 1
        return ok

    def share(self, target: str) -> bool:
        ok = self._do("--share", target)
        self.stats["shared"] += int(ok)
        return ok

    def can_alert(self) -> bool:
        return bool(self.cfg.enabled and (self.cli or self.cfg.ntfy_url))

    # -- what is said --------------------------------------------------------------------------------------
    def _number(self, who: str) -> Tuple[str, Optional[str]]:
        w = who.strip()
        digits = re.sub(r"[^\d+]", "", w)
        if len(re.sub(r"\D", "", digits)) >= 6:
            return digits, digits
        for name, num in (self.cfg.contacts or {}).items():
            if name.lower() == w.lower() or name.lower().split()[0] == w.lower():
                return name, num
        return w, None

    def slow(self, text: str, now: Optional[float] = None) -> bool:
        """Would `hear` talk to the phone for this line? (Ringing, sharing, notifying, sending a text, asking if it is
        connected: each runs the KDE Connect tool, which can take seconds -- so the caller may run `hear` off its loop.)"""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if len(t) > 300 or not self.cfg.enabled or not self.cli:
            return False
        p = self.pending_sms
        if p is not None and now - p[3] <= 300 and (_SEND.match(t) or (_yesno.is_yes(t, strong=True) and now - p[3] <= 90)):
            return True
        return bool(_STATUS.match(t) or _RING.match(t) or _SHARE.match(t) or _NOTE.match(t))

    def hear(self, text: str, last_thing: Optional[str] = None, now: Optional[float] = None) -> Optional[str]:
        """What to say back, or None. `last_thing` is a file path or link ASH just gave (for "send it to my phone")."""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if len(t) > 300 or not self.cfg.enabled:
            return None
        p = self.pending_sms
        if p is not None and now - p[3] <= 300:
            if _SEND.match(t) or (_yesno.is_yes(t, strong=True) and now - p[3] <= 90):      # a plain yes only right after the read-back
                self.pending_sms = None
                if self._do("--send-sms", p[2], "--destination", p[1]):
                    self.stats["texted"] += 1
                    return f"Sent the text to {p[0]}."
                return "The text didn't go -- is the phone connected?"
            if (_CANCEL.match(t) or _yesno.is_no(t)):
                self.pending_sms = None
                return "Okay, I won't send it."
        if _STATUS.match(t):
            if not self.cli:
                return self.NOT_SET_UP
            return "Yes, your phone is connected." if self._device() else "No -- KDE Connect can't see your phone right now."
        kind = ("ring" if _RING.match(t) else "share" if _SHARE.match(t) else None)
        note = _NOTE.match(t)
        text_m = _TEXT.match(t)
        if kind is None and note is None and text_m is None:
            return None
        if not self.cli:
            return self.NOT_SET_UP
        if kind == "ring":
            return "Ringing your phone." if self.ring() else "I couldn't reach your phone -- is it on the same Wi-Fi?"
        if kind == "share":
            if not last_thing:
                return "There's nothing I've just found to send -- ask me to find the file first."
            return (f"Sent {os.path.basename(last_thing) or last_thing} to your phone." if self.share(last_thing)
                    else "I couldn't reach your phone.")
        if note is not None:
            body = (note.group("text") or note.group("text2") or note.group("text3") or "").strip()
            return "Sent to your phone." if self.notify(body) else "I couldn't reach your phone."
        if not self.cfg.allow_sms:
            return "Texting is switched off (phone.allow_sms)."
        name, number = self._number(text_m.group("to"))
        if number is None:
            return f"I don't have a number for {name}. Add it under phone.contacts, or say the number."
        body = text_m.group("text").strip()
        self.pending_sms = (name, number, body, now)
        return f"Text to {name} ({number}): \u201c{body}\u201d. Say \u201csend it\u201d to send it."

    def status(self) -> Dict[str, Any]:
        return {"ready": self.ready, "ntfy": bool(self.cfg.ntfy_url), **self.stats}


from . import yesno as _yesno  # noqa: E402  (a plain yes sends what was read back; a plain no drops it)
