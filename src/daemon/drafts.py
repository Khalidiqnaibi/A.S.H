"""
src/daemon/drafts.py

Email drafts, read back, and sent only when you say so.

    reply to Omar saying I'll be late                  a reply to Omar's newest email, threaded under it
    reply to the last email saying thanks, got it
    email sara@example.com saying the files are ready   a new email -- only to an address you say or type
    change it to I'll be there at 9                     a new body
    make it more formal / make it friendlier            reworded by the language model -- the one step that uses it,
                                                        and only because you asked
    send it / yes, send it                              sent -- only if email.allow_send is on (it is off until you
                                                        switch it on), only within ten minutes of the read-back
    don't send it / cancel the email                    dropped

The draft itself is made by rule from your own words ("Hi Omar, I'll be late. Khalid") -- no model, nothing made up.
Every draft is read back in full before it can be sent, and "send it" sends that exact text. Nothing is sent for a
guest, for speech that was not said to ASH, or while someone ASH doesn't know is in view (the read-back is held).
"""

from __future__ import annotations

import re
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Callable, Dict, Optional

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you)[,\s]+)*"
_SAY = r"(?:\s*,\s*|\s+)(?:saying|to\s+say|and\s+say|that\s+says|telling\s+(?:him|her|them)|to\s+tell\s+(?:him|her|them)|that)\s+(?P<gist>.+?)"
_REPLY = re.compile(_LEAD + r"(?:reply|respond|write\s+back|answer)\s+to\s+(?P<who>.+?)(?:'s)?(?:\s+(?:e-?mail|message|mail))?" + _SAY + r"\s*[.!]?\s*$", re.I)
_NEW = re.compile(_LEAD + r"(?:e-?mail|write\s+(?:an?\s+)?e-?mail\s+to|send\s+(?:an?\s+)?e-?mail\s+to|draft\s+(?:an?\s+)?e-?mail\s+to)\s+"
                  r"(?P<to>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)" + _SAY + r"\s*[.!]?\s*$", re.I)
_SEND = re.compile(_LEAD + r"(?:(?:yes|yeah|ok(?:ay)?)[,\s]+)?(?:go\s+ahead\s+and\s+)?send\s+(?:it|that|the\s+(?:e-?mail|reply|draft))(?:\s+now)?\s*[.!]?\s*$", re.I)
_CANCEL = re.compile(_LEAD + r"(?:don'?t\s+send\s+(?:it|that)|cancel\s+(?:it|that|the\s+(?:e-?mail|reply|draft))|discard\s+(?:it|the\s+draft)|"
                     r"never\s*mind\s+the\s+(?:e-?mail|reply))\s*[.!]?\s*$", re.I)
_CHANGE = re.compile(_LEAD + r"(?:change\s+it\s+to|make\s+it\s+say|say\s+instead|instead\s+say)\s+(?P<gist>.+?)\s*[.!]?\s*$", re.I)
_POLISH = re.compile(_LEAD + r"make\s+it\s+(?P<how>(?:more\s+|a\s+bit\s+more\s+|less\s+)?(?:formal|friendly|friendlier|polite|politer|"
                     r"professional|casual|warmer|shorter|longer|nicer|clearer))\s*[.!]?\s*$", re.I)


@dataclass
class Draft:
    to_addr: str
    to_name: str
    subject: str
    body: str
    in_reply_to: str = ""
    made_at: float = 0.0


def _sentence(gist: str) -> str:
    g = (gist or "").strip().strip('"“”').strip()
    if not g:
        return ""
    g = g[0].upper() + g[1:]
    return g if g[-1] in ".!?" else g + "."


def compose(to_name: str, gist: str, signer: str = "") -> str:
    first = (to_name or "").split()[0] if to_name and "@" not in to_name else ""
    greet = f"Hi {first}," if first else "Hi,"
    return f"{greet}\n\n{_sentence(gist)}\n\n{signer}".rstrip() + ("" if signer else "")


def smtp_host_for(imap_host: str) -> str:
    h = (imap_host or "").lower()
    if h in ("outlook.office365.com", "imap-mail.outlook.com"):
        return "smtp.office365.com"
    return re.sub(r"^imap\.", "smtp.", h) if h.startswith("imap.") else h


def smtp_send(cfg: Dict[str, Any], d: Draft) -> None:
    """Send `d` with the account in cfg (host/user/password from the email settings; smtp_host/smtp_port optional)."""
    host = cfg.get("smtp_host") or smtp_host_for(cfg.get("host", ""))
    port = int(cfg.get("smtp_port") or 465)
    msg = EmailMessage()
    msg["From"] = cfg["user"]
    msg["To"] = d.to_addr
    msg["Subject"] = d.subject
    if d.in_reply_to:
        msg["In-Reply-To"] = d.in_reply_to
        msg["References"] = d.in_reply_to
    msg.set_content(d.body)
    timeout = float(cfg.get("timeout_s") or 20)
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=timeout) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as s:
            s.starttls()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)


class Drafts:
    def __init__(self, inbox, account: Callable[[], Dict[str, Any]], allow_send: bool = False,
                 send: Callable[[Dict[str, Any], Draft], None] = smtp_send,
                 polish: Optional[Callable[[str, str], Optional[str]]] = None,
                 signer: Callable[[], str] = lambda: "", clock: Callable[[], float] = time.time, window_s: float = 600.0):
        self.inbox = inbox
        self.account = account
        self.allow_send = allow_send
        self.send = send
        self.polish = polish
        self.signer = signer
        self.clock = clock
        self.window_s = window_s
        self.draft: Optional[Draft] = None
        self.stats = {"drafted": 0, "sent": 0, "cancelled": 0, "failed": 0}

    def _readback(self, d: Draft) -> str:
        to = f"{d.to_name} <{d.to_addr}>" if d.to_name and d.to_name != d.to_addr else d.to_addr
        body = d.body.replace("\n\n", " / ").replace("\n", " ")
        tail = ("Say “send it” to send it, or “change it to ...”." if self.allow_send else
                "Sending is switched off (email.allow_send), so it stays a draft -- I can only read it to you.")
        return f"Here's the draft to {to}, subject “{d.subject}”: “{body}” {tail}"

    def _current(self, now: float) -> Optional[Draft]:
        if self.draft is not None and now - self.draft.made_at > self.window_s:
            self.draft = None
        return self.draft

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """What to say back, or None if `text` is not about a draft."""
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if len(t) > 300:
            return None
        d = self._current(now)
        if d is not None:
            if _SEND.match(t) or (_yesno.is_yes(t, strong=True) and now - d.made_at <= 90):    # a plain yes only right after the read-back
                return self._send(d)
            if (_CANCEL.match(t) or _yesno.is_no(t)):
                self.draft = None
                self.stats["cancelled"] += 1
                return "Okay, I won't send it."
            m = _CHANGE.match(t)
            if m:
                d.body = compose(d.to_name, m.group("gist"), self.signer())
                d.made_at = now
                return self._readback(d)
            m = _POLISH.match(t)
            if m:
                if self.polish is None:
                    return "I can't reword it here -- tell me what to say instead (“change it to ...”)."
                new = None
                try:
                    new = self.polish(d.body, m.group("how"))
                except Exception:
                    new = None
                if not new or not new.strip():
                    return "I couldn't reword it just now -- it's unchanged."
                d.body = new.strip()
                d.made_at = now
                return self._readback(d)
        elif (_SEND.match(t) or _yesno.is_yes(t, strong=True)) or (_CANCEL.match(t) or _yesno.is_no(t)):
            return None                                      # nothing waiting: "send it" may be someone else's
        m = _REPLY.match(t)
        if m:
            return self._reply(m.group("who"), m.group("gist"), now)
        m = _NEW.match(t)
        if m:
            to = m.group("to")
            self.draft = Draft(to_addr=to, to_name=to, subject=self._subject(m.group("gist")),
                               body=compose("", m.group("gist"), self.signer()), made_at=now)
            self.stats["drafted"] += 1
            return self._readback(self.draft)
        return None

    @staticmethod
    def _subject(gist: str) -> str:
        words = re.sub(r"[^\w\s']", "", gist or "").split()
        s = " ".join(words[:6])
        return (s[:1].upper() + s[1:]) if s else "A note"

    def _reply(self, who: str, gist: str, now: float) -> str:
        if self.inbox is None or not getattr(self.inbox, "configured", False):
            return ("I'm not connected to your email. Add the account in the console's settings, and I can draft "
                    "replies -- and send them once you switch sending on.")
        row = self.inbox.last_from(who)
        if row is None:
            return f"I don't see a recent email from {who.strip()} -- I only know the newest few unread ones."
        name, subject, addr, mid = row
        if not addr:
            return f"I can see {name}'s email, but not their address, so I can't reply to it."
        subj = subject if subject.lower().startswith("re:") else f"Re: {subject}" if subject else "Re:"
        self.draft = Draft(to_addr=addr, to_name=name, subject=subj, body=compose(name, gist, self.signer()),
                           in_reply_to=mid, made_at=now)
        self.stats["drafted"] += 1
        return self._readback(self.draft)

    def _send(self, d: Draft) -> str:
        if not self.allow_send:
            return "Sending is switched off (email.allow_send in daemon.json), so I've kept it as a draft."
        acct = self.account() or {}
        if not (acct.get("user") and acct.get("password") and (acct.get("smtp_host") or acct.get("host"))):
            return "I don't have the email account details to send with."
        try:
            self.send(acct, d)
        except Exception as e:
            self.stats["failed"] += 1
            from .inbox import why_it_failed
            kind = "the login was refused" if isinstance(e, smtplib.SMTPAuthenticationError) else why_it_failed(e)
            if isinstance(e, smtplib.SMTPException) and kind == "something went wrong":
                kind = "the mail server wouldn't answer"
            return f"It didn't send ({kind}). The draft is still here -- say “send it” to try again."
        self.draft = None
        self.stats["sent"] += 1
        return f"Sent to {d.to_name or d.to_addr}."

    def status(self) -> Dict[str, Any]:
        return {**self.stats, "allow_send": self.allow_send, "waiting": self.draft is not None}


from . import yesno as _yesno  # noqa: E402  (a plain yes sends what was read back; a plain no drops it)
