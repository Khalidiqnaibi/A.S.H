"""
tst_smarthome.py -- lights, plugs and the like, through Home Assistant.

  H1  Commands read: on/off both ways round, dimming, "is it on?"; and what
      is not one
  H2  Names matched: friendly names, plurals, "lamp" for a light; an unknown
      name; two words must both match
  H3  Never by voice: locks, doors, garages, alarms, heaters -- by domain or
      by name
  H4  The calls made, the token sent, and failures said plainly
  H5  Not configured: nothing taken (the brain answers)

Run:  PYTHONIOENCODING=utf-8 python tst_smarthome.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tst_wondering as tw  # noqa: E402
from src.daemon.smarthome import SmartHome, SmartHomeConfig  # noqa: E402

check = tw.check

STATES = [
    {"entity_id": "light.living_room", "state": "off", "attributes": {"friendly_name": "Living Room Lights"}},
    {"entity_id": "light.bedroom", "state": "on", "attributes": {"friendly_name": "Bedroom Lamp"}},
    {"entity_id": "switch.kettle", "state": "off", "attributes": {"friendly_name": "Kettle"}},
    {"entity_id": "fan.office", "state": "on", "attributes": {"friendly_name": "Office Fan"}},
    {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "Front Door"}},
    {"entity_id": "cover.garage", "state": "closed", "attributes": {"friendly_name": "Garage"}},
    {"entity_id": "switch.space_heater", "state": "off", "attributes": {"friendly_name": "Space Heater"}},
    {"entity_id": "sensor.temp", "state": "21", "attributes": {"friendly_name": "Temperature"}},
    {"entity_id": "cover.bedroom_blinds", "state": "open", "attributes": {"friendly_name": "Bedroom Blinds"}},
    {"entity_id": "switch.fairy", "state": "off", "attributes": {"friendly_name": "Fairy Lights"}},
]


class FakeHA:
    def __init__(self):
        self.calls = []
        self.fail = None

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append((method, url, body, headers))
        if self.fail:
            raise self.fail
        if url.endswith("/api/states"):
            return STATES
        return []


def ha(**cfg):
    cfg.setdefault("url", "http://ha.local:8123/")
    cfg.setdefault("token", "test-token")
    f = FakeHA()
    s = SmartHome(SmartHomeConfig.from_dict(cfg), http=f)
    s.f = f
    return s


def posts(s):
    return [(u.split("/api/")[1], b) for m, u, b, h in s.f.calls if m == "POST"]


def h1_reading():
    print("\n=== H1. Commands ===")
    s = ha()
    check("turn on X", s.hear("turn on the living room lights") == "Turning on Living Room Lights."
          and posts(s)[-1] == ("services/light/turn_on", {"entity_id": "light.living_room"}), str(posts(s)))
    check("turn X off", s.hear("turn the kettle off") == "Turning off Kettle."
          and posts(s)[-1] == ("services/switch/turn_off", {"entity_id": "switch.kettle"}))
    check("switch off the fan", s.hear("switch off the office fan") == "Turning off Office Fan."
          and posts(s)[-1][0] == "services/fan/turn_off")
    check("dim to 30%", s.hear("dim the bedroom lamp to 30%") == "Bedroom Lamp set to 30%."
          and posts(s)[-1] == ("services/light/turn_on", {"entity_id": "light.bedroom", "brightness_pct": 30}))
    check("a switch cannot be dimmed", s.hear("set the kettle to 50%") == "Kettle can't be dimmed.")
    check("over 100% is 100%", s.hear("set the bedroom lamp to 150%") == "Bedroom Lamp set to 100%."
          and posts(s)[-1][1]["brightness_pct"] == 100)
    check("is it on?", s.hear("is the bedroom lamp on?") == "Bedroom Lamp is on.")
    for text in ("turn left at the lights", "what's the temperature", "open outlook", "", None, "turn on " + "x" * 90):
        check(f"not a command: {str(text)[:30]!r}", s.hear(text) is None)


def h2_names():
    print("\n=== H2. Names ===")
    s = ha()
    check("plural and singular", s.hear("turn on the living room light") == "Turning on Living Room Lights.")
    check("'lamp' finds a light", s.hear("turn off the bedroom light") == "Turning off Bedroom Lamp.")
    check("plurals match on any device", s.hear("turn on the fairy light") == "Turning on Fairy Lights.")
    check("an unknown name says so", s.hear("turn on the attic lights") ==
          "I couldn't find anything called attic lights in Home Assistant.")
    check("a question about an unknown name is left to the brain", s.hear("is the attic light on?") is None)
    check("every word must match", s.hear("turn on the office lights") ==
          "I couldn't find anything called office lights in Home Assistant.")
    n = len([c for c in s.f.calls if c[0] == "GET"])
    s.hear("turn on the kettle")
    check("states are cached, refreshed after a change", len([c for c in s.f.calls if c[0] == "GET"]) <= n + 1)


def h3_never():
    print("\n=== H3. Never by voice ===")
    s = ha()
    for text in ("unlock the front door", "turn off the front door", "turn on the garage", "turn on the space heater",
                 "turn off the alarm", "turn on the oven", "turn on the bedroom blinds"):
        r = s.hear(text)
        check(f"refused or not taken: {text!r}", r is None or "don't operate" in r, str(r))
    check("nothing was operated", posts(s) == [])
    check("refusals are counted", s.status()["refused"] >= 4)


def h4_calls():
    print("\n=== H4. Calls and failures ===")
    s = ha()
    s.hear("turn on the kettle")
    m, u, b, h = s.f.calls[-1]
    check("the token is sent as a bearer token", h["Authorization"] == "Bearer test-token")
    check("to the configured server", u == "http://ha.local:8123/api/services/switch/turn_on")
    s = ha()
    s.f.fail = TimeoutError("down")
    check("unreachable: said plainly", s.hear("turn on the kettle") == "I can't reach Home Assistant right now.")
    s = ha()
    s.states()
    s.f.fail = OSError("500")
    check("a call that fails: said plainly", s.hear("turn on the kettle") ==
          "Home Assistant didn't take that, so Kettle is unchanged.")


def h5_unconfigured():
    print("\n=== H5. Not configured ===")
    check("no URL or token: nothing taken", SmartHome(SmartHomeConfig()).hear("turn on the kettle") is None)
    check("disabled: nothing taken", ha(enabled=False).hear("turn on the kettle") is None)
    check("status says so", SmartHome(SmartHomeConfig()).status()["configured"] is False)


def h6_runtime():
    print("\n=== H6. The runtime ===")
    import tempfile
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, smarthome=True,
                             smarthome_cfg={"url": "http://ha.local:8123", "token": "t"}),
        response_sink=lambda t, m: out.append((t, m)))
    rt.smarthome.http = FakeHA()
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="turn on the kettle", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("done without the model", [o[0] for o in out] == ["Turning on Kettle."] and ash.calls == [])
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="what's the time", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("anything else goes to the model", ash.calls == ["what's the time"])
    check("status carries it", rt.status()["smarthome"]["done"] == 1)


if __name__ == "__main__":
    for fn in (h1_reading, h2_names, h3_never, h4_calls, h5_unconfigured, h6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
