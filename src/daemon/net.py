"""
src/daemon/net.py

The one place ASH touches the network, so what it says about the network is true and what left this computer is on record.

28 Nov: "No connection" was said while online -- DuckDuckGo had answered a search with a 202 bot check, nothing came back, and every error was read as "offline". A request here
ends in exactly one of four ways, and the words ASH uses follow it:

    ok        the server answered and there is a body
    refused   the server answered and said no (a 4xx/5xx, a bot check, a rate limit): the network works, this service won't
    empty     the server answered 200 and had nothing
    down      no network at all (no route, DNS, timeout before any answer) -- and only then "no connection" / "I'm offline"

Every request is written to state/network.jsonl (when, host, why, the query -- never a body) so "what did you send out today?" is answered from the record:
the promise that ASH only goes online when asked is checkable. `online()` is a cached TCP probe (nothing is sent).
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

_UA = "ASH-local-assistant/1.0"
_lock = threading.Lock()
_state: Dict[str, Any] = {"log": None, "probe": None, "at": -1e18, "up": True, "clock": time.time}


def configure(log_path: Optional[str] = None, probe: Optional[Callable[[], bool]] = None, clock: Optional[Callable[[], float]] = None):
    """Where the record is kept; tests hand in their own probe and clock."""
    if log_path is not None:
        _state["log"] = log_path
    if probe is not None:
        _state["probe"] = probe
        _state["at"] = -1e18
    if clock is not None:
        _state["clock"] = clock


def _tcp_probe(timeout: float = 2.0) -> bool:
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        try:
            socket.create_connection((host, port), timeout).close()
            return True
        except OSError:
            continue
    return False


def online(ttl: float = 20.0) -> bool:
    """Is there a network at all? Cached for `ttl` seconds (a probe sends nothing but a TCP connect)."""
    now = _state["clock"]()
    if now - _state["at"] >= ttl:
        _state["up"] = bool((_state["probe"] or _tcp_probe)())
        _state["at"] = now
    return bool(_state["up"])


@dataclass
class Result:
    status: str                    # ok | refused | empty | down
    code: int = 0
    text: str = ""
    host: str = ""
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except ValueError:
            return None

    def words(self, what: str = "that") -> str:
        """What to say about a request that did not give a body, in terms of the real cause."""
        return {"down": "I have no connection right now",
                "refused": f"{self.host or 'the site'} answered but wouldn't give me {what} (it said {self.code or 'no'})",
                "empty": f"{self.host or 'the site'} answered but had nothing for {what}"}.get(self.status, "")


def _log(res: Result, url: str):
    path = _state["log"]
    if not path:
        return
    try:
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        sent = " ".join(v for k, vs in q.items() if k.lower() in ("q", "term", "search", "srsearch", "name", "query") for v in vs)[:120]
        row = {"t": round(_state["clock"]()), "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(_state["clock"]())), "host": u.netloc, "why": res.why,
               "sent": sent or u.path[:80], "status": res.status, "code": res.code}
        with _lock:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def fetch(url: str, why: str, timeout: float = 6.0, headers: Optional[Dict[str, str]] = None, limit: int = 2_000_000,
          opener: Callable[..., Any] = urllib.request.urlopen) -> Result:
    """One GET. Never raises: the Result says what happened. `why` is the plain reason ("game requirements"), kept in the record."""
    host = urllib.parse.urlparse(url).netloc
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        with opener(req, timeout=timeout) as r:
            code = int(getattr(r, "status", 200) or 200)
            body = r.read(limit).decode("utf-8", "ignore")
        if code >= 400 or code == 202:                           # 202 from a search page is a bot check, not a result
            res = Result("refused", code, body if code != 202 else "", host, why)
        elif not body.strip():
            res = Result("empty", code, "", host, why)
        else:
            res = Result("ok", code, body, host, why)
    except urllib.error.HTTPError as e:
        res = Result("refused", int(e.code), "", host, why)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        # a timeout or refusal of the connection with the network up is the site's, not ours
        res = Result("down" if not online(ttl=5.0) else "refused", 0, "", host, why)
    _log(res, url)
    return res


def recent(path: Optional[str] = None, hours: float = 24.0, now: Optional[float] = None) -> List[Dict[str, Any]]:
    path = path or _state["log"]
    now = _state["clock"]() if now is None else now
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if now - float(d.get("t") or 0.0) <= hours * 3600.0:
                    out.append(d)
    except OSError:
        pass
    return out


_ASK_SENT = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:what\s+(?:did|have)\s+you\s+(?:send|sent|look(?:ed)?\s+up|search(?:ed)?(?:\s+for)?|ask(?:ed)?)(?:\s+(?:out|online|on\s+the\s+(?:web|internet)))?(?:\s+(?:today|so\s+far|lately|this\s+(?:morning|week)))?"
                       r"|what\s+(?:has\s+)?left\s+(?:this|my)\s+(?:computer|pc|machine|laptop)(?:\s+today)?|what\s+(?:data\s+)?did\s+you\s+(?:send|share)\s+(?:out|online)(?:\s+today)?"
                       r"|have\s+you\s+(?:been\s+)?(?:online|on\s+the\s+(?:web|internet))\s*(?:today)?|did\s+you\s+go\s+online(?:\s+today)?)\W*$", re.I)
_ASK_ONLINE = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:am\s+i\s+online|are\s+(?:we|you)\s+online|is\s+(?:the\s+)?(?:internet|network|wi-?fi)\s+(?:up|working|on|connected)|do\s+(?:we|you)\s+have\s+(?:internet|a\s+connection))\W*$", re.I)


def asks_sent(text: str) -> bool:
    return bool(_ASK_SENT.match((text or "").strip()))


def asks_online(text: str) -> bool:
    return bool(_ASK_ONLINE.match((text or "").strip()))


def sent_say(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Nothing -- I haven't sent anything out of this computer in the last day."
    by: Dict[str, List[str]] = {}
    for r in rows:
        by.setdefault(str(r.get("host", "?")), []).append(str(r.get("sent", "")))
    parts = []
    for host, sent in sorted(by.items(), key=lambda kv: -len(kv[1]))[:5]:
        uniq = list(dict.fromkeys(s for s in sent if s))[:3]
        parts.append(f"{host} {len(sent)} time{'s' if len(sent) != 1 else ''}" + (" (" + "; ".join(uniq) + ")" if uniq else ""))
    return f"In the last day I sent {len(rows)} request{'s' if len(rows) != 1 else ''}: " + ", ".join(parts) + ". Only the words of what you asked for -- nothing else."


def online_say() -> str:
    return "Yes, there's a connection." if online(ttl=0.0) else "No -- I can't reach the internet right now."
