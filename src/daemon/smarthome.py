"""
src/daemon/smarthome.py

Lights, plugs and the like, through Home Assistant.

Home Assistant runs on the local network and already knows every device in
the house by a friendly name, so ASH does not talk to devices itself: it
asks Home Assistant's REST API, with a long-lived token the user creates in
their Home Assistant profile and puts in daemon.json.

    "turn on the living room lights"     light.turn_on on the entity whose
    "switch off the fan"                 friendly name matches best
    "dim the bedroom light to 30%"       brightness
    "is the porch light on?"             read its state

Only lights, switches, fans and plain "on/off" helpers are touched. Locks,
alarms, covers (garage doors) and anything named like one are never operated
by voice here: that is a security decision for Home Assistant itself, not
for a speech recogniser that can mishear. Names are matched by rule; nothing
leaves the local network.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.smarthome")

SAFE_DOMAINS = ("light", "switch", "fan", "input_boolean")
NEVER = re.compile(r"\b(?:lock|locks|unlock|door|doors|garage|gate|alarm|security|camera|oven|stove|heater|boiler|"
                       r"pump|valve|furnace|siren|sprinklers?)\b", re.I)

_ON_OFF = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:turn|switch|put)\s+(?P<a>on|off)\s+(?:the\s+|my\s+)?"
                     r"(?P<t>[\w '\-]{2,40}?)\s*(?:please)?\s*[.!]?\s*$"
                     r"|^\s*(?:please\s+)?(?:turn|switch)\s+(?:the\s+|my\s+)?(?P<t2>[\w '\-]{2,40}?)\s+(?P<a2>on|off)"
                     r"\s*(?:please)?\s*[.!]?\s*$", re.I)
_DIM = re.compile(r"^\s*(?:please\s+)?(?:dim|set|brighten)\s+(?:the\s+|my\s+)?(?P<t>[\w '\-]{2,40}?)\s+(?:to\s+)?"
                  r"(?P<p>\d{1,3})\s*(?:%|percent)\s*[.!]?\s*$", re.I)
_STATE = re.compile(r"^\s*(?:is|are)\s+(?:the\s+|my\s+)?(?P<t>[\w '\-]{2,40}?)\s+(?:still\s+)?(?:on|off)\s*\??\s*$",
                    re.I)
# What a home device sounds like -- only these get "not connected" while no smart home is set up.
_DEVICE = re.compile(r"\b(?:lights?|lamps?|bulbs?|fans?|heat(?:er|ing)?|thermostat|a/?c|air\s*con(?:ditioning|ditioner)?|"
                     r"plugs?|sockets?|tv|television|blinds?|curtains?|kettle|purifier|humidifier)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _tokens(s: str) -> List[str]:
    return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in _norm(s).split()
            if t not in ("the", "my", "a")]


@dataclass
class SmartHomeConfig:
    enabled: bool = True
    url: str = ""                    # e.g. http://homeassistant.local:8123
    token: str = ""                  # a long-lived access token
    timeout_s: float = 5.0
    states_ttl_s: float = 300.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SmartHomeConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k in ("url", "token"):
                setattr(c, k, str(v or "").strip())
            elif k == "enabled":
                c.enabled = bool(v)
            elif k in ("timeout_s", "states_ttl_s"):
                try:
                    setattr(c, k, float(v))
                except (TypeError, ValueError):
                    pass
        return c


Http = Callable[[str, str, Optional[Dict[str, Any]], Dict[str, str], float], Any]


def _http(method: str, url: str, body: Optional[Dict[str, Any]], headers: Dict[str, str], timeout: float) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:        # noqa: S310 -- the user's own server
        return json.loads(r.read(2_000_000).decode("utf-8") or "null")


class SmartHome:
    def __init__(self, cfg: Optional[SmartHomeConfig] = None, http: Http = _http,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or SmartHomeConfig()
        self.http = http
        self.clock = clock
        self._states: List[Dict[str, Any]] = []
        self._states_at = -1e18
        self.stats = {"done": 0, "refused": 0, "errors": 0}
        self.last_choices: List[str] = []

    NOT_SET_UP = ("I'm not connected to your smart home yet. Add your Home Assistant address and token in the console's "
                  "settings (smarthome.url and smarthome.token), and I'll switch things for you.")

    @property
    def ready(self) -> bool:
        return self.cfg.enabled and bool(self.cfg.url) and bool(self.cfg.token)

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        return self.http(method, self.cfg.url.rstrip("/") + path, body,
                         {"Authorization": f"Bearer {self.cfg.token}", "Content-Type": "application/json"},
                         self.cfg.timeout_s)

    def states(self, force: bool = False) -> List[Dict[str, Any]]:
        now = self.clock()
        if force or now - self._states_at > self.cfg.states_ttl_s:
            got = self._call("GET", "/api/states")
            self._states = [s for s in (got or []) if isinstance(s, dict) and "entity_id" in s]
            self._states_at = now
        return self._states

    def find(self, name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """The entity a spoken name means, or (None, why not)."""
        if NEVER.search(name):
            return None, "refused"
        want = set(_tokens(name))
        if not want:
            return None, "unknown"
        best, best_score = None, 0.0
        tied: List[Dict[str, Any]] = []
        for s in self.states():
            dom = s["entity_id"].split(".", 1)[0]
            fname = str((s.get("attributes") or {}).get("friendly_name") or s["entity_id"].split(".", 1)[1])
            have = set(_tokens(fname)) | set(_tokens(s["entity_id"].split(".", 1)[1].replace("_", " ")))
            if dom == "light":
                have |= {"light", "lamp"}
            elif dom == "fan":
                have.add("fan")
            score = len(want & have) / len(want)
            if score > best_score:
                best, best_score, tied = s, score, [s]
            elif score == best_score and score >= 0.99:
                tied.append(s)
        if best is None or best_score < 0.99:
            return None, "unknown"
        # "turn off the light" with five lights switched whichever came first. Two that fit equally well are asked about.
        safe_tied = [t for t in tied if t["entity_id"].split(".", 1)[0] in SAFE_DOMAINS and not self._never(t)]
        if len(safe_tied) > 1:
            self.last_choices = [str((t.get("attributes") or {}).get("friendly_name") or t["entity_id"]) for t in safe_tied][:4]
            return None, "ambiguous"
        # The name the person said, the friendly name AND the entity id ("switch.garage_door_opener" is "Opener" to a person).
        if best["entity_id"].split(".", 1)[0] not in SAFE_DOMAINS or self._never(best):
            return None, "refused"
        return best, ""

    @staticmethod
    def _never(s: Dict[str, Any]) -> bool:
        ident = s["entity_id"].split(".", 1)[1].replace("_", " ")
        return bool(NEVER.search(ident) or NEVER.search(str((s.get("attributes") or {}).get("friendly_name") or "")))

    def hear(self, text: str) -> Optional[str]:
        """A device command or question: what to say back, or None."""
        t = (text or "").strip()
        if not t or len(t) > 80 or not self.cfg.enabled:
            return None
        if not self.ready:
            # Not connected: "turn on the lights" is answered plainly, not handed to a model that would say "Done!"
            # and do nothing. Anything that is not about a home device is left alone.
            if (_ON_OFF.match(t) or _DIM.match(t) or _STATE.match(t)) and _DEVICE.search(t):
                return self.NOT_SET_UP
            return None
        m = _ON_OFF.match(t)
        d = _DIM.match(t)
        q = _STATE.match(t)
        if not (m or d or q):
            return None
        name = (m.group("t") or m.group("t2")) if m else (d.group("t") if d else q.group("t"))
        try:
            ent, why = self.find(name)
        except Exception as e:
            self.stats["errors"] += 1
            logger.info("Home Assistant unreachable: %s", type(e).__name__)
            return "I can't reach Home Assistant right now."
        if why == "refused":
            self.stats["refused"] += 1
            return f"I don't operate {name.strip()} by voice. Use Home Assistant for that one."
        if why == "ambiguous":
            a = self.last_choices
            return "Which one -- " + ", ".join(a[:-1]) + (" or " if len(a) > 1 else "") + a[-1] + "?"
        if ent is None:
            if not m and not d:
                return None
            if not _DEVICE.search(name):
                return None                 # "turn off the monitor", "turn off the wifi": not a home device, not for this
            return f"I couldn't find anything called {name.strip()} in Home Assistant."
        friendly = (ent.get("attributes") or {}).get("friendly_name") or ent["entity_id"]
        if q:
            return f"{friendly} is {ent.get('state', 'unknown')}."
        dom = ent["entity_id"].split(".", 1)[0]
        try:
            if d:
                pct = max(0, min(100, int(d.group("p"))))
                if dom != "light":
                    return f"{friendly} can't be dimmed."
                self._call("POST", "/api/services/light/turn_on",
                           {"entity_id": ent["entity_id"], "brightness_pct": pct})
                self.stats["done"] += 1
                return f"{friendly} set to {pct}%."
            action = (m.group("a") or m.group("a2")).lower()
            svc = "turn_on" if action == "on" else "turn_off"
            self._call("POST", f"/api/services/{dom}/{svc}", {"entity_id": ent["entity_id"]})
            self.stats["done"] += 1
            self._states_at = -1e18                      # its state just changed
            return f"Turning {action} {friendly}."
        except Exception as e:
            self.stats["errors"] += 1
            logger.info("Home Assistant call failed: %s", type(e).__name__)
            return f"Home Assistant didn't take that, so {friendly} is unchanged."

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "configured": bool(self.cfg.url and self.cfg.token), **self.stats}
