"""
src/ui/settings.py

The console's settings panel, and the "what is ASH waiting on / up to" view.

Settings are a fixed whitelist -- the panel can never write an arbitrary key
into daemon.json. A change is applied to the running daemon where the part
already exists (turning wondering off takes effect at once), and saved to
daemon.json so it survives a restart. A part that was off when the daemon
started does not exist yet: the change is saved and the reply says a restart
is needed. Secrets (the smart-home token) are never sent back to the page,
only whether one is set.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.ui.settings")

# daemon.json section -> the runtime attribute that holds it
PART = {
    "wondering": "wonder", "follow_ups": "followups", "research": "research", "calendar": "calendar",
    "smarthome": "smarthome", "reminders": "reminders", "play": "play", "focus": "focus",
    "weather": "weather", "watch": "watcher", "briefings": "briefings", "routines": "routines",
    "free_moments": "moments", "desktop": "desktop", "remarks": "remarks", "tasks": "tasks", "email": "inbox",
}

# key -> (type, label, group). Types: bool, str, secret, list (comma separated).
SETTINGS: Dict[str, Tuple[str, str, str]] = {
    "wondering.enabled": ("bool", "Ask about things it doesn't know", "Talking"),
    "free_moments.enabled": ("bool", "Speak up at free moments", "Talking"),
    "follow_ups.enabled": ("bool", "Ask how things went", "Talking"),
    "play.enabled": ("bool", "Games and stories", "Talking"),
    "briefings.enabled": ("bool", "Morning and evening briefings", "Talking"),
    "routines.enabled": ("bool", "Notice routines", "Talking"),
    "focus.enabled": ("bool", "Focus mode", "Talking"),
    "remarks.enabled": ("bool", "Keep a log of what it says", "Talking"),
    "reminders.enabled": ("bool", "Reminders and timers", "Helping"),
    "watch.enabled": ("bool", "Watch downloads, battery, disk", "Helping"),
    "research.enabled": ("bool", "Look things up in the background", "Helping"),
    "desktop.enabled": ("bool", "Open and close apps", "Helping"),
    "tasks.enabled": ("bool", "Multi-step tasks", "Helping"),
    "weather.enabled": ("bool", "Weather", "Weather"),
    "weather.city": ("str", "City", "Weather"),
    "calendar.enabled": ("bool", "Calendar", "Calendar"),
    "calendar.urls": ("list", "Calendar feed links (.ics), comma separated", "Calendar"),
    "smarthome.enabled": ("bool", "Smart home", "Smart home"),
    "smarthome.url": ("str", "Home Assistant address", "Smart home"),
    "smarthome.token": ("secret", "Access token", "Smart home"),
    "email.enabled": ("bool", "Email (read-only)", "Email"),
    "email.host": ("str", "IMAP server (e.g. imap.gmail.com)", "Email"),
    "email.user": ("str", "Account", "Email"),
    "email.password": ("secret", "App password", "Email"),
    "wondering.lookup.roots": ("folders", "Folders to look terms up in, comma separated", "Looking things up"),
    "wondering.lookup.web": ("bool", "Look acronyms up on Wikipedia", "Looking things up"),
}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse(key: str, raw: Any) -> Any:
    """The value for `key`, from what the page sent. ValueError if bad."""
    kind = SETTINGS[key][0]
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"{key} takes on or off")
    if kind in ("list", "folders"):
        items = raw if isinstance(raw, (list, tuple)) else str(raw or "").split(",")
        out = [str(x).strip() for x in items if str(x).strip()]
        if any(len(x) > 400 or any(c in x for c in "\r\n\x00") for x in out):
            raise ValueError(f"{key}: an entry is too long or not one line")
        if kind == "folders":
            return out
        for u in out:
            if not u.lower().startswith(("http://", "https://", "webcal://")):
                raise ValueError(f"not a link: {u[:60]}")
        return out
    s = str(raw if raw is not None else "").strip()
    if len(s) > 400 or any(c in s for c in "\r\n\x00"):
        raise ValueError(f"{key}: too long or not one line")
    if key == "smarthome.url" and s and not s.lower().startswith(("http://", "https://")):
        raise ValueError("the address starts with http:// or https://")
    return s


def save(path: str, key: str, value: Any):
    """Set one key in daemon.json, atomically: read, change, write a temp
    file beside it, replace. Everything else in the file is kept as is."""
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("daemon.json is not an object")
    *parents, name = key.split(".")
    sec = data
    for p in parents:
        nxt = sec.get(p)
        if not isinstance(nxt, dict):
            nxt = sec[p] = {}
        sec = nxt
    sec[name] = value
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".daemon-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _part(runtime, key: str):
    if key.startswith("wondering.lookup."):
        return getattr(runtime, "lookup_engine", None)
    return getattr(runtime, PART.get(key.split(".", 1)[0], ""), None)


def _name(key: str) -> str:
    return key.rsplit(".", 1)[1]


def _saved(saved: Dict[str, Any], key: str) -> Any:
    cur: Any = saved
    for p in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def current(runtime, key: str, saved: Optional[Dict[str, Any]] = None) -> Any:
    """What the setting is now: the running part's value, else the saved one."""
    name = _name(key)
    part = _part(runtime, key)
    val: Any = None
    if part is not None:
        cfg = getattr(part, "cfg", None)
        if name == "enabled" and not hasattr(cfg, "enabled") and hasattr(part, "enabled"):
            val = part.enabled
        elif cfg is not None and hasattr(cfg, name):
            val = getattr(cfg, name)
        elif name == "enabled":
            val = True                       # it exists and has no switch of its own
        if name == "enabled" and key == "wondering.enabled" and val:
            val = not getattr(part, "off", False)
    elif saved is not None:
        val = _saved(saved, key)
        if name == "enabled" and val is None:
            val = False
    if SETTINGS[key][0] in ("list", "folders"):
        return list(val or [])
    if SETTINGS[key][0] == "secret":
        return bool(val)
    return val


def apply(runtime, key: str, value: Any) -> bool:
    """Change the running daemon. False if that part is not running (it will
    be after a restart)."""
    section, name = key.split(".", 1)[0], _name(key)
    part = _part(runtime, key)
    if part is None:
        return False
    cfg = getattr(part, "cfg", None)
    if name == "enabled":
        if cfg is not None and hasattr(cfg, "enabled"):
            cfg.enabled = bool(value)
        elif hasattr(part, "enabled"):
            part.enabled = bool(value)
        else:
            return False                     # no live switch (the remark log): a restart
        if key == "wondering.enabled" and value:
            part.off = False                 # "stop wondering" said earlier is lifted too
        return True
    if cfg is None or not hasattr(cfg, name):
        return False
    if key == "wondering.lookup.roots":
        value = tuple(value)
    if key == "calendar.urls":
        value = tuple(value)
        with part._lock:
            for src in [s for s in part._texts if s.startswith(("http", "webcal")) and s not in value]:
                part._texts.pop(src, None)
                part._fetched.pop(src, None)
    setattr(cfg, name, value)
    if key == "weather.city":
        part._coords = None                  # look the new place up
        part._fetched_at = -1e18
    if section == "smarthome":
        part._states_at = -1e18
    if section == "email":
        part.reset()                         # a new account: check it afresh
    return True


def listing(runtime, path: Optional[str]) -> List[Dict[str, Any]]:
    saved = None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception:
            saved = None
    out = []
    for key, (kind, label, group) in SETTINGS.items():
        try:
            val = current(runtime, key, saved)
        except Exception:
            val = None
        out.append({"key": key, "type": kind, "label": label, "group": group, "value": val,
                    "live": _part(runtime, key) is not None})
    return out


def change(runtime, path: Optional[str], key: str, raw: Any) -> Dict[str, Any]:
    if key not in SETTINGS:
        return {"ok": False, "error": f"not a setting: {key!r}"}
    try:
        value = parse(key, raw)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    live = False
    try:
        live = apply(runtime, key, value)
    except Exception:
        logger.exception("Applying %s failed", key)
    saved = False
    if path:
        try:
            save(path, key, value)
            saved = True
        except Exception as e:
            logger.exception("Saving %s failed", key)
            if not live:
                return {"ok": False, "error": f"could not save: {e}"}
    return {"ok": True, "key": key, "live": live, "saved": saved, "restart": not live}


# ----------------------------------------------------------------------
# waiting on you, and up to
# ----------------------------------------------------------------------
def _hm(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


def waiting(runtime, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The one thing ASH is waiting on the user for, most pressing first --
    the answer the next thing said would be taken as."""
    now = runtime.clock() if now is None else now
    desk = getattr(runtime, "desktop", None)
    p = getattr(desk, "_pending", None)
    if p and now - p[2] <= desk.cfg.confirm_window_s:
        return {"kind": "confirm", "text": f"Close {p[0]}?", "replies": ["Yes", "No"]}
    tk = getattr(runtime, "tasks", None)
    plan = tk.awaiting(now) if tk is not None else None
    if plan:
        from src.daemon.tasks import describe
        return {"kind": "plan", "text": "Go ahead with: " + describe(plan["steps"]) + "?", "replies": ["Yes", "No"]}
    aw = getattr(runtime, "_awaiting_meaning", None)
    if aw and now - aw.get("t", 0) <= runtime.cfg.correction_window_s:
        return {"kind": "correction", "text": f"What did you want, instead of “{aw.get('query', '')[:60]}”?",
                "replies": []}
    play = getattr(runtime, "play", None)
    if play is not None:
        if getattr(play, "game", None) is not None:
            return {"kind": "game", "text": f"Playing {getattr(play.game, 'name', 'a game')} -- your turn.",
                    "replies": ["Stop"]}
        off = getattr(play, "offered", None)
        if off and now - off[1] <= 120.0:
            return {"kind": "offer", "text": f"Offered a {off[0]}.", "replies": ["Yes", "No thanks"]}
    w = getattr(runtime, "wonder", None)
    if w is not None:
        q = w.awaiting(now)
        if q is not None:
            return {"kind": "question", "text": q.question, "replies": ["I don't know", "Not now"]}
    f = getattr(runtime, "focus", None)
    if f is not None and f.active(now):
        until = getattr(f, "until", None)
        return {"kind": "focus", "text": "Focusing" + (f" until {_hm(until)}" if until else "")
                + " -- holding remarks back.", "replies": ["Done focusing"]}
    return None


def up_to(runtime, now: Optional[float] = None) -> List[Dict[str, Any]]:
    """What ASH is keeping track of, a few of each."""
    now = runtime.clock() if now is None else now
    out: List[Dict[str, Any]] = []

    def add(title: str, items: List[str], more: int = 0):
        if items:
            out.append({"title": title, "items": items, "more": max(0, more)})

    def safe(fn: Callable[[], None]):
        try:
            fn()
        except Exception:
            logger.debug("up_to part failed", exc_info=True)

    def reminders():
        r = runtime.reminders
        rows = sorted((x for x in r.items.values() if x.status == "set"), key=lambda x: x.due)
        add("Reminders", [f"{_hm(x.due)} {x.text or x.label or x.kind}".strip() for x in rows[:3]], len(rows) - 3)

    def watches():
        rows = runtime.watcher.active()
        add("Watching", [w.what for w in rows[:3]], len(rows) - 3)

    def wonders():
        rows = runtime.wonder.pending()
        add("Wondering", [r["question"] for r in rows[:3]], len(rows) - 3)

    def followups():
        rows = runtime.followups.pending()
        add("Will ask how it went", [r["question"] for r in rows[:3]], len(rows) - 3)

    def research():
        r = runtime.research
        with r._lock:
            jobs = [j for j in r.jobs.values() if j.status in ("queued", "working", "ready")]
        add("Looking up", [j.topic + (" (ready)" if j.status == "ready" else "") for j in jobs[:3]], len(jobs) - 3)

    def holding():
        names = []
        for part in (getattr(runtime, "moments", None), getattr(runtime, "wonder", None)):
            b = getattr(part, "backoff", None)
            if b is not None:
                names += b.status(now)["benched"]
        add("Holding off on", [n.replace("_", " ") for n in names])

    def routines():
        r = runtime.routines
        with r._lock:
            rows = sorted((x for x in r.seen.values() if len(x["days"]) >= r.cfg.min_days),
                          key=lambda x: (-len(x["days"]), -len(x["steps"])))
        add("Routines noticed", [" > ".join(x["steps"]) for x in rows[:3]], len(rows) - 3)

    def corrections():
        from src.brain.corrections import default
        mem = default()
        with mem._lock:
            rows = sorted(mem.items, key=lambda c: -c.t)
        add("Recent corrections", [f"“{c.said[:40]}” is not {c.wrong.replace('_', ' ')}"
                                   + (f" (it's {c.right.replace('_', ' ')})" if c.right else "")
                                   for c in rows[:3]], len(rows) - 3)

    for name, fn in (("reminders", reminders), ("watcher", watches), ("wonder", wonders),
                     ("followups", followups), ("research", research), ("routines", routines)):
        if getattr(runtime, name, None) is not None:
            safe(fn)
    safe(holding)
    if getattr(runtime.cfg, "corrections", True):
        safe(corrections)
    return out
