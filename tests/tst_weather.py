"""
tst_weather.py -- the weather, for the briefing and when asked.

  W1  Not configured: nothing fetched; asked, it says how to set it up (not a model guessing a forecast)
  W2  Fetching: a city is looked up once, the forecast at most hourly, only
      the place is sent, failures are quiet
  W3  Describing: today, tomorrow, rain and from when
  W4  Asking: today, tomorrow, "will it rain"; another place is left to the
      brain
  W5  The runtime: in the morning and evening briefings, and answered

Run:  PYTHONIOENCODING=utf-8 python tests/tst_weather.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.weather import Weather, WeatherConfig  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 8, 0, 0, 0, 0, -1))

FORECAST = {
    "daily": {"time": ["2026-03-10", "2026-03-11"], "weathercode": [0, 63], "temperature_2m_max": [26.4, 19.0],
              "temperature_2m_min": [17.6, 12.2], "precipitation_probability_max": [10, 80]},
    "hourly": {"time": [f"2026-03-1{d}T{h:02d}:00" for d in (0, 1) for h in range(24)],
               "precipitation_probability": [0] * 10 + [60] + [0] * 13 + [0] * 15 + [70] * 9},   # a shower today at 10
}


class Net:
    def __init__(self):
        self.urls = []
        self.fail = False

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.fail:
            raise TimeoutError("offline")
        if "geocoding" in url:
            return {"results": [{"latitude": 31.95, "longitude": 35.93}]}
        return FORECAST


def wrig(**cfg):
    t = [T0]
    net = Net()
    w = Weather(WeatherConfig.from_dict(cfg), fetch=net, clock=lambda: t[0], background=False)
    w.t, w.net = t, net
    return w


def w1_unconfigured():
    print("\n=== W1. Not configured ===")
    w = wrig()
    w.refresh()
    check("nothing fetched", w.net.urls == [] and not w.configured)
    check("asked, it says it is not set up and how -- nothing fetched for it",
          w.hear("what's the weather like today") == w.NOT_SET_UP and "weather.city" in w.NOT_SET_UP and w.net.urls == [])
    check("...another place is still the brain's to answer", w.hear("what's the weather in Paris") is None)
    check("...and a line that is not about the weather is left alone", w.hear("what's the time") is None)
    check("no briefing part", w.briefing_part("morning", T0) is None)
    check("disabled: nothing even with a city", wrig(city="Amman", enabled=False).configured is False)


def w2_fetching():
    print("\n=== W2. Fetching ===")
    w = wrig(city="Amman")
    w.refresh()
    check("a city is looked up, then the forecast fetched", len(w.net.urls) == 2 and "geocoding" in w.net.urls[0]
          and "latitude=31.9500&longitude=35.9300" in w.net.urls[1])
    check("only the place is sent", "Amman" in w.net.urls[0] and all("user" not in u.lower() for u in w.net.urls))
    w.t[0] += 600
    w.refresh()
    check("not again within the hour", len(w.net.urls) == 2)
    w.t[0] += 3600
    w.refresh()
    check("again after it, without looking the city up again", len(w.net.urls) == 3 and "geocoding" not in w.net.urls[2])
    w = wrig(lat=31.9, lon=35.9)
    w.refresh()
    check("coordinates need no lookup", len(w.net.urls) == 1 and "geocoding" not in w.net.urls[0])
    w = wrig(city="Amman")
    w.net.fail = True
    w.refresh()
    check("offline: quiet, no forecast", w.describe() is None and w.stats["errors"] == 1)
    check("and the briefing just has no weather", w.briefing_part("morning", T0) is None)


def w3_describe():
    print("\n=== W3. Describing ===")
    w = wrig(city="Amman")
    w.refresh()
    check("today", w.describe(0) == "18 to 26 degrees, clear", str(w.describe(0)))
    check("tomorrow, with rain and from when", w.describe(1) == "12 to 19 degrees, rain; rain likely from 3pm",
          str(w.describe(1)))
    check("briefing: morning is today", w.briefing_part("morning", T0) == "Weather: 18 to 26 degrees, clear.")
    check("evening is tomorrow", w.briefing_part("evening", T0).startswith("Weather tomorrow: 12 to 19"))


def w4_asking():
    print("\n=== W4. Asking ===")
    w = wrig(city="Amman")
    check("today", w.hear("what's the weather like today?") == "Today: 18 to 26 degrees, clear.")
    check("tomorrow", w.hear("how's the weather tomorrow").startswith("Tomorrow: 12 to 19"))
    check("will it rain tomorrow", w.hear("will it rain tomorrow").startswith("Tomorrow:"))
    check("plain 'what's the weather'", w.hear("what's the weather").startswith("Today:"))
    for text in ("what's the weather like in Tokyo", "what is the weather for London tomorrow", "nice weather huh",
                 "", None):
        check(f"left to the brain: {str(text)[:30]!r}", w.hear(text) is None)
    w = wrig(city="Amman")
    w.net.fail = True
    check("no forecast: said plainly", w.hear("what's the weather") ==
          "I don't have a forecast right now -- I couldn't reach the weather service.")


def w5_runtime():
    print("\n=== W5. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, weather=True,
                             weather_cfg={"city": "Amman"}, briefings=True,
                             briefings_path=os.path.join(d, "b.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    net = Net()
    rt.weather.fetch, rt.weather.background, rt.weather.clock = net, False, (lambda: clock[0])
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="what's the weather like today",
                          addressed=True, salience=0.9, urgency=0.6))
    rt._tick()
    check("answered without the model", out[-1] == "Today: 18 to 26 degrees, clear." and ash.calls == [], str(out))
    check("in the morning briefing", "Weather: 18 to 26 degrees, clear." in rt.briefings.compose("morning", clock[0]))
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="what's the weather in Tokyo",
                          addressed=True, salience=0.9, urgency=0.6))
    rt._tick()
    check("another place goes to the brain", ash.calls == ["what's the weather in Tokyo"])
    check("status carries weather", rt.status()["weather"]["has_forecast"] is True)


if __name__ == "__main__":
    for fn in (w1_unconfigured, w2_fetching, w3_describe, w4_asking, w5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
