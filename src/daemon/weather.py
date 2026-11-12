"""
src/daemon/weather.py

The weather, for the briefing (and "what's the weather like" when asked).

Open-Meteo's forecast API needs no account or key. Nothing is fetched until a
place is set in daemon.json (`weather.city`, or `weather.lat` and
`weather.lon`): then the place -- and only the place -- is sent, at most
once an hour, on its own thread, and the forecast is kept until the next one.
A city name is turned into coordinates once, with Open-Meteo's geocoder.

    morning briefing   "Weather: 18 to 26 degrees, sunny; rain likely after 3pm."
    evening briefing   tomorrow's
    asked              today's, or tomorrow's

Fails quietly: offline, the briefing simply has no weather in it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.weather")

# WMO weather codes, in words.
CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
         51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
         66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow",
         80: "showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "snow showers",
         95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail"}

_ASK = re.compile(r"\b(?:what'?s|what is|how'?s|how is)\s+the\s+weather(?:\s+like)?(?:\s+(?P<when>today|tomorrow))?\b"
                  r"|\b(?:will|is)\s+it\s+(?:going to\s+)?(?:rain|snow)\s*(?P<when2>today|tomorrow)?\b"
                  r"|\bweather\s+(?:for\s+)?(?P<when3>today|tomorrow)\b", re.I)


def _get(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ASH-local-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:          # noqa: S310 -- fixed https host
        return json.loads(r.read(1_000_000).decode("utf-8"))


@dataclass
class WeatherConfig:
    enabled: bool = True
    city: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    refresh_s: float = 3600.0
    timeout_s: float = 6.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "WeatherConfig":
        c = cls()
        for k, v in (d or {}).items():
            if k == "city":
                c.city = str(v or "").strip()
            elif k in ("lat", "lon"):
                try:
                    setattr(c, k, float(v)) if v not in (None, "") else None
                except (TypeError, ValueError):
                    pass
            elif k == "enabled":
                c.enabled = bool(v)
            elif k in ("refresh_s", "timeout_s"):
                try:
                    setattr(c, k, float(v))
                except (TypeError, ValueError):
                    pass
        return c


class Weather:
    def __init__(self, cfg: Optional[WeatherConfig] = None, fetch: Callable[[str, float], Any] = _get,
                 clock: Callable[[], float] = time.time, background: bool = True):
        self.cfg = cfg or WeatherConfig()
        self.fetch = fetch
        self.clock = clock
        self.background = background
        self._lock = threading.Lock()
        self._coords: Optional[Tuple[float, float]] = (
            (self.cfg.lat, self.cfg.lon) if self.cfg.lat is not None and self.cfg.lon is not None else None)
        self._daily: Optional[Dict[str, Any]] = None
        self._hourly: Optional[Dict[str, Any]] = None
        self._fetched_at = -1e18
        self._inflight = False
        self.stats = {"fetched": 0, "errors": 0}

    @property
    def configured(self) -> bool:
        return self.cfg.enabled and (bool(self.cfg.city) or self._coords is not None)

    def refresh(self, now: Optional[float] = None):
        if not self.configured or self._inflight:
            return
        now = self.clock() if now is None else now
        if now - self._fetched_at < self.cfg.refresh_s:
            return
        self._fetched_at = now
        self._inflight = True
        if self.background:
            threading.Thread(target=self._get_forecast, daemon=True, name="ash-weather").start()
        else:
            self._get_forecast()

    def _get_forecast(self):
        try:
            if self._coords is None:
                g = self.fetch("https://geocoding-api.open-meteo.com/v1/search?count=1&name="
                               + urllib.parse.quote(self.cfg.city), self.cfg.timeout_s)
                r = (g or {}).get("results") or []
                if not r:
                    raise ValueError("place not found")
                self._coords = (float(r[0]["latitude"]), float(r[0]["longitude"]))
            lat, lon = self._coords
            d = self.fetch("https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f"
                           "&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                           "&hourly=precipitation_probability&timezone=auto&forecast_days=2" % (lat, lon),
                           self.cfg.timeout_s)
            with self._lock:
                self._daily = d.get("daily")
                self._hourly = d.get("hourly")
            self.stats["fetched"] += 1
        except Exception as e:
            self.stats["errors"] += 1
            logger.info("Weather fetch failed: %s", type(e).__name__)
        finally:
            self._inflight = False

    def describe(self, day: int = 0) -> Optional[str]:
        """'18 to 26 degrees, sunny; rain likely after 3pm' for today (0) or
        tomorrow (1), or None if there is no forecast."""
        with self._lock:
            dd, hh = self._daily, self._hourly
        if not dd:
            return None
        try:
            lo = round(float(dd["temperature_2m_min"][day]))
            hi = round(float(dd["temperature_2m_max"][day]))
            sky = CODES.get(int(dd["weathercode"][day]), "mixed")
            rain = dd.get("precipitation_probability_max", [None, None])[day]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        s = f"{lo} to {hi} degrees, {sky}"
        if rain is not None and rain >= 50 and hh:
            date = dd["time"][day] if dd.get("time") else ""
            after = None
            for t, p in zip(hh.get("time", []), hh.get("precipitation_probability", [])):
                if t.startswith(date) and p is not None and p >= 50:
                    after = datetime.fromisoformat(t).strftime("%I%p").lstrip("0").lower()
                    break
            s += f"; rain likely" + (f" from {after}" if after else "")
        return s

    def briefing_part(self, kind: str, now: float) -> Optional[str]:
        self.refresh(now)
        d = self.describe(1 if kind == "evening" else 0)
        if not d:
            return None
        return ("Weather: " if kind != "evening" else "Weather tomorrow: ") + d + "."

    NOT_SET_UP = ("I don't know where you are, so I can't check the weather yet. Put your town in the console's "
                  "settings (weather.city) and I'll answer that straight away.")

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        if not self.cfg.enabled:
            return None
        t = (text or "").strip()
        m = _ASK.search(t) if len(t) <= 80 else None
        if not m:
            return None
        if re.search(r"\b(?:in|for|at|near)\s+(?!the\s+(?:morning|afternoon|evening)\b)(?:the\s+)?[A-Z][a-z]", t):
            return None                  # another place: the brain and its weather tool answer that
        if not self.configured:
            return self.NOT_SET_UP       # said plainly, not handed to a model that would guess a forecast
        self.refresh(now)
        when = (m.group("when") or m.group("when2") or m.group("when3") or "today").lower()
        d = self.describe(1 if when == "tomorrow" else 0)
        if not d:
            return "I don't have a forecast right now -- I couldn't reach the weather service."
        return f"{when.capitalize()}: {d}."

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "configured": self.configured, "has_forecast": self._daily is not None,
                **self.stats}
