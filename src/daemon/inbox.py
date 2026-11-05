"""
src/daemon/inbox.py

Email, carefully: read-only.

Once an IMAP account is set (email.host / user / password in daemon.json or
the console's settings -- use an app password, not your main one), ASH
checks the inbox at most every ten minutes, on its own thread, and keeps
only the unread count and, for the newest few, who sent them and their
subjects. It never reads a message body, never marks anything as read (the
folder is opened read-only and headers fetched with BODY.PEEK), and never
sends, moves or deletes.

    morning briefing      "Email: 4 unread, from Sara, Omar and GitHub."
    "any new email?"      "3 unread: from Sara, Omar and GitHub."
    "who emailed me?"     the same, with the subjects
    "what are they about" the subjects

Drafting and sending are not here. When they come, sending will always be
confirmed first.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.inbox")

_LEAD = r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?ash[,\s]+)?(?:(?:please|can you|could you)[,\s]+)?"
_MAIL = r"(?:e-?mails?|mails?|inbox)"
_ASK = re.compile(_LEAD + r"(?:(?:do\s+i\s+have|have\s+i\s+got|did\s+i\s+get|is\s+there|are\s+there|got)\s+(?:any\s+)?"
                  r"(?:new\s+|unread\s+)*" + _MAIL + r"|any\s+(?:new\s+|unread\s+)*" + _MAIL +
                  r"|check\s+(?:my\s+)?" + _MAIL + r"|how\s+many\s+(?:unread\s+|new\s+)*" + _MAIL +
                  r"(?:\s+do\s+i\s+have)?|what'?s\s+in\s+my\s+inbox|anything\s+in\s+my\s+inbox)"
                  r"(?:\s+(?:today|now|yet|at\s+all))?\s*[?.!]*\s*$", re.I)
_WHO = re.compile(_LEAD + r"(?:who\s+(?:has\s+)?(?:e-?mailed|emailed|mailed|wrote\s+to|written\s+to)\s+me|"
                  r"who\s+are\s+(?:they|the\s+e-?mails)\s+from|who'?s\s+(?:it|that|the\s+e-?mail)\s+from)"
                  r"(?:\s+(?:today|lately|recently))?\s*[?.!]*\s*$", re.I)
_ABOUT = re.compile(_LEAD + r"(?:what\s+are\s+(?:they|the\s+e-?mails)\s+about|(?:read\s+me\s+)?the\s+subjects?"
                    r"|what\s+(?:are\s+)?the\s+subjects?)\s*[?.!]*\s*$", re.I)


@dataclass
class InboxConfig:
    enabled: bool = True
    host: str = ""
    port: int = 993
    user: str = ""
    password: str = ""
    folder: str = "INBOX"
    refresh_s: float = 600.0
    timeout_s: float = 15.0
    keep: int = 6                    # newest unread kept (sender, subject)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "InboxConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k in ("host", "user", "password", "folder") and isinstance(v, str):
                setattr(c, k, v.strip())
            elif k == "enabled":
                c.enabled = bool(v)
            elif k in ("port", "keep"):
                try:
                    setattr(c, k, int(v))
                except (TypeError, ValueError):
                    pass
            elif k in ("refresh_s", "timeout_s"):
                try:
                    setattr(c, k, float(v))
                except (TypeError, ValueError):
                    pass
        return c


def _decode(raw: str) -> str:
    try:
        parts = email.header.decode_header(raw or "")
        out = "".join(p.decode(enc or "utf-8", errors="replace") if isinstance(p, bytes) else p for p, enc in parts)
    except Exception:
        out = raw or ""
    return re.sub(r"\s+", " ", out).strip()


def sender_name(raw_from: str) -> str:
    """'"Sara Ali" <sara@x.com>' -> 'Sara Ali'; 'noreply@github.com' -> 'GitHub'-ish ('github')."""
    name, addr = email.utils.parseaddr(_decode(raw_from))
    name = name.strip().strip('"')
    if name:
        return name
    if "@" in addr:
        dom = addr.split("@", 1)[1].split(".")
        return dom[-2] if len(dom) >= 2 else dom[0]
    return addr or "someone"


def _imap_fetch(cfg: InboxConfig) -> Tuple[int, List[Tuple[str, str]]]:
    """(unread count, [(sender, subject)] newest first). Read-only."""
    m = imaplib.IMAP4_SSL(cfg.host, cfg.port, timeout=cfg.timeout_s)
    try:
        m.login(cfg.user, cfg.password)
        typ, _ = m.select(cfg.folder, readonly=True)
        if typ != "OK":
            raise RuntimeError("could not open the folder")
        typ, data = m.search(None, "UNSEEN")
        ids = (data[0] or b"").split() if typ == "OK" and data else []
        out: List[Tuple[str, str]] = []
        for i in reversed(ids[-cfg.keep:]):
            typ, msg = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
            if typ != "OK" or not msg or not isinstance(msg[0], tuple):
                continue
            h = email.message_from_bytes(msg[0][1])
            out.append((sender_name(h.get("From", "")), _decode(h.get("Subject", ""))[:80]))
        return len(ids), out
    finally:
        try:
            m.logout()
        except Exception:
            pass


def _names(names: List[str]) -> str:
    seen: List[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    if len(seen) > 3:
        return ", ".join(seen[:3]) + " and others"
    return seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " and " + seen[-1]


class Inbox:
    def __init__(self, cfg: Optional[InboxConfig] = None,
                 fetch: Callable[[InboxConfig], Tuple[int, List[Tuple[str, str]]]] = _imap_fetch,
                 clock: Callable[[], float] = time.time, background: bool = True):
        self.cfg = cfg or InboxConfig()
        self.fetch = fetch
        self.clock = clock
        self.background = background
        self._lock = threading.Lock()
        self.unread: Optional[int] = None
        self.recent: List[Tuple[str, str]] = []
        self.checked_at = -1e18
        self._inflight = False
        self.last_error = ""
        self.stats = {"checked": 0, "errors": 0}

    @property
    def configured(self) -> bool:
        c = self.cfg
        return bool(c.enabled and c.host and c.user and c.password)

    def refresh(self, now: Optional[float] = None, force: bool = False):
        if not self.configured or self._inflight:
            return
        now = self.clock() if now is None else now
        if not force and now - self.checked_at < self.cfg.refresh_s:
            return
        self.checked_at = now
        self._inflight = True
        if self.background:
            threading.Thread(target=self._check, daemon=True, name="ash-inbox").start()
        else:
            self._check()

    def _check(self):
        try:
            n, rows = self.fetch(self.cfg)
            with self._lock:
                self.unread, self.recent = int(n), list(rows)
            self.stats["checked"] += 1
            self.last_error = ""
        except Exception as e:
            self.stats["errors"] += 1
            # Never the password: only the kind of failure.
            self.last_error = "login failed" if isinstance(e, imaplib.IMAP4.error) else type(e).__name__
            logger.info("Inbox check failed: %s", self.last_error)
        finally:
            self._inflight = False

    def reset(self):
        """After the account changes: forget what was read, check again soon."""
        with self._lock:
            self.unread, self.recent = None, []
        self.checked_at = -1e18
        self.last_error = ""

    # -- saying it ---------------------------------------------------------
    def summary(self, subjects: bool = False) -> Optional[str]:
        with self._lock:
            n, rows = self.unread, list(self.recent)
        if n is None:
            return None
        if n == 0:
            return "No unread email."
        head = f"{n} unread" if n != 1 else "1 unread email"
        if not rows:
            return head + "."
        if subjects:
            return head + ": " + "; ".join(f"{s} -- {subj or '(no subject)'}" for s, subj in rows[:4]) + "."
        return head + ": from " + _names([s for s, _ in rows]) + "."

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        if not self.configured or kind not in ("morning", "evening"):
            return None
        with self._lock:
            n, rows = self.unread, list(self.recent)
        if not n:
            return None
        out = f"Email: {n} unread"
        if rows:
            out += ", from " + _names([s for s, _ in rows])
        return out + "."

    def hear(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if not t or len(t) > 80:
            return None
        about = bool(_ABOUT.match(t))
        if about and self.unread is None and not self.configured:
            return None                       # "what are they about" is not ours unless email is set
        who = bool(_WHO.match(t))
        if not (about or who or _ASK.match(t)):
            return None
        if not self.cfg.enabled:
            return None
        if not self.configured:
            return ("I'm not connected to your email. Add the account in the console's settings "
                    "(an app password works best) -- I'll only ever read who sent what.")
        s = self.summary(subjects=about or who)
        if s is None:
            if self.last_error:
                return f"I couldn't check your email just now ({self.last_error})."
            self.refresh(force=True)
            return "I haven't checked your email yet -- give me a moment and ask again."
        return s

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "configured": self.configured, "unread": self.unread,
                "error": self.last_error, **self.stats}
