"""
src/daemon/toasts.py

Windows notifications, said at the right moment -- JARVIS's "Ms. Potts is calling". OFF unless the person turns it on ("Read Windows notifications"
in the settings). When on, Windows' own notification list (wpndatabase.db) is read READ-ONLY every half minute, and only two things are taken from a
new notification: the app it came from and its first line, which is who it is from ("WhatsApp: Omar"). The message itself is never read, kept or
said. A name ASH knows is said with who they are ("Omar, your cousin").

The line is offered like any other remark, so focus, a game, a call, quiet hours, a mute and the first minutes after a start all hold it;
"what did I miss?" lists what came while the person was away; "any notifications?" says the last hour's.
"""

from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.toasts")

EPOCH_DIFF = 11644473600          # seconds between 1601-01-01 (FILETIME) and 1970-01-01

_APPS = (("whatsapp", "WhatsApp"), ("telegram", "Telegram"), ("discord", "Discord"), ("signal", "Signal"), ("messenger", "Messenger"),
         ("instagram", "Instagram"), ("teams", "Teams"), ("slack", "Slack"), ("outlook", "Outlook"), ("microsoft.windowscommunicationsapps", "Mail"),
         ("olk", "Outlook"), ("thunderbird", "Thunderbird"), ("spotify", "Spotify"), ("steam", "Steam"), ("zoom", "Zoom"), ("skype", "Skype"),
         ("phonelink", "Phone Link"), ("yourphone", "Phone Link"), ("chrome", "Chrome"), ("msedge", "Edge"), ("opera", "Opera"),
         ("firefox", "Firefox"), ("brave", "Brave"), ("kdeconnect", "KDE Connect"), ("calendar", "Calendar"))
_MESSAGING = {"WhatsApp", "Telegram", "Discord", "Signal", "Messenger", "Instagram", "Teams", "Slack", "Skype", "Phone Link", "KDE Connect"}
_MAIL = {"Outlook", "Mail", "Thunderbird"}
_SKIP = re.compile(r"windows\.systemtoast|microsoft\.windows\.|windows\.defender|securityhealth|microsoft\.windowsstore|windowsupdate|"
                   r"\bpython|\bash\b|a\.s\.h|focusassist|nvidia|realtek|onedrive|microsoft\.xbox|gamebar|cortana|explorer", re.I)


def app_name(primary_id: str) -> str:
    """'5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App' -> 'WhatsApp'; '' for Windows itself, ASH's own and the like."""
    low = (primary_id or "").lower()
    if not low or _SKIP.search(low):
        return ""
    for key, name in _APPS:
        if key in low:
            return name
    tail = re.split(r"[!_]", primary_id)[0].split(".")[-1]
    tail = re.sub(r"(?:desktop|app|exe)$", "", tail, flags=re.I)
    return tail[:20] if len(tail) >= 3 else ""


_GENERIC = re.compile(r"^(?:you\s+(?:may\s+)?have\s+(?:\d+\s+)?new\b|new\s+(?:message|notification|email|mail)s?\b|\d+\s+new\b|notification\b|you\s+got\b)", re.I)


def first_line(payload: Any) -> str:
    """The first <text> of a toast's XML: who or what it is from. Nothing after it."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", "replace")
    m = re.search(r"<text[^>]*>(.*?)</text>", str(payload or ""), re.S | re.I)
    if not m:
        return ""
    t = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    t = re.sub(r"\s+", " ", t)
    if _GENERIC.match(t):
        return ""                                   # "You may have new messages" is not a name: no sender is better than a wrong one
    return t[:40]


def default_path() -> str:
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Notifications", "wpndatabase.db")


def read_db(path: str, after_filetime: int) -> List[Tuple[int, str, Any]]:
    """(arrival FILETIME, handler PrimaryId, payload) for toasts that arrived after `after_filetime`. Read-only; nothing is written."""
    if not path or not os.path.exists(path):
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        rows = con.execute("SELECT n.ArrivalTime, h.PrimaryId, n.Payload FROM Notification n JOIN NotificationHandler h ON n.HandlerId = h.RecordId "
                           "WHERE n.Type = 'toast' AND n.ArrivalTime > ? ORDER BY n.ArrivalTime LIMIT 50", (int(after_filetime),)).fetchall()
    finally:
        con.close()
    return [(int(r[0] or 0), str(r[1] or ""), r[2]) for r in rows]


def to_filetime(t: float) -> int:
    return int((t + EPOCH_DIFF) * 10_000_000)


def from_filetime(ft: int) -> float:
    return ft / 10_000_000 - EPOCH_DIFF


class Toasts:
    def __init__(self, enabled: bool = False, path: Optional[str] = None, reader: Optional[Callable[[int], List[Tuple[int, str, Any]]]] = None,
                 clock: Callable[[], float] = time.time, who: Optional[Callable[[str], str]] = None):
        self.enabled = bool(enabled)
        self.path = path or default_path()
        self.reader = reader or (lambda after: read_db(self.path, after))
        self.clock = clock
        self.who = who or (lambda name: "")
        self.since = to_filetime(clock())             # what was there before ASH started is not news
        self.recent: Deque[Tuple[float, str, str]] = deque(maxlen=60)    # (when, app, from)
        self._lock = threading.Lock()
        self.stats = {"read": 0, "new": 0, "failed": 0}

    def poll(self) -> List[Tuple[float, str, str]]:
        """New notifications since the last look, as (when, app, from) -- only when switched on."""
        if not self.enabled:
            return []
        try:
            rows = self.reader(self.since)
            self.stats["read"] += 1
        except Exception:
            self.stats["failed"] += 1
            logger.debug("reading notifications failed", exc_info=True)
            return []
        out = []
        with self._lock:
            for ft, pid, payload in rows:
                self.since = max(self.since, ft)
                app = app_name(pid)
                if not app:
                    continue
                item = (from_filetime(ft), app, first_line(payload))
                self.recent.append(item)
                out.append(item)
        self.stats["new"] += len(out)
        return out

    def _one(self, app: str, frm: str) -> str:
        rel = self.who(frm) if frm else ""
        who = f"{frm}, {rel}" if rel else frm
        if not frm:
            return app
        if app in _MESSAGING:
            return f"{app} from {who}"
        if app in _MAIL:
            return f"{app}, mail from {who}"
        return f"{app}: {who}"

    def line(self, items: List[Tuple[float, str, str]]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            _t, app, frm = items[0]
            s = self._one(app, frm)
            if app in _MESSAGING and frm:
                return f"A message on {app} from " + s.split(" from ", 1)[1] + "."
            return s[0].upper() + s[1:] + "."
        parts, seen = [], set()
        for _t, app, frm in items:
            k = (app, frm)
            if k not in seen:
                seen.add(k)
                parts.append(self._one(app, frm))
        shown = parts[:3]
        more = len(parts) - len(shown)
        return f"{len(items)} notifications: " + ", ".join(shown[:-1]) + (" and " if len(shown) > 1 else "") + shown[-1] + \
            (f", and {more} more" if more > 0 else "") + "."

    def since_(self, since: float) -> List[Tuple[float, str, str]]:
        with self._lock:
            return [x for x in self.recent if x[0] >= since]

    _ASK = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:any\s+(?:new\s+)?notifications|(?:read|check)\s+(?:me\s+)?my\s+notifications|"
                      r"what\s+notifications\s+(?:did\s+i\s+get|do\s+i\s+have|came\s+in)|did\s+i\s+get\s+any\s+notifications|"
                      r"who\s+(?:messaged|texted)\s+me)(?:\s+(?:today|lately|recently|while\s+i\s+was\s+away))?\s*[?.!]*\s*$", re.I)

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self._ASK.match((text or "").strip()):
            return None
        if not self.enabled:
            return ("Reading Windows notifications is off. Turn on “Read Windows notifications” in the settings if you want it -- I'd "
                    "only ever say the app and who it's from.")
        now = self.clock() if now is None else now
        self.poll()
        got = self.since_(now - 3600.0)
        return ("In the last hour: " + self.line(got)[0].lower() + self.line(got)[1:]) if got else "No notifications in the last hour."
