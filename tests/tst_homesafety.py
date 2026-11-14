"""
tst_homesafety.py -- what the smart-home handler must not do.

Review found: the never-by-voice list was tested on the spoken name and the friendly name, never the entity's id, so
`switch.garage_door_opener` with the friendly name "Opener" was operated, and the list had no pump, valve, boiler or siren; two
devices that matched a name equally well were not told apart ("turn off the light" switched whichever came first); and a line
that was not about a home device at all ("turn off the monitor") was answered "I couldn't find anything called monitor in Home
Assistant", hijacking it from the handlers that could do it.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_homesafety.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_smarthome as th  # noqa: E402
import tst_wondering as tw  # noqa: E402
from src.daemon.smarthome import SmartHome, SmartHomeConfig  # noqa: E402

check = tw.check


def home(states):
    f = th.FakeHA()
    real = list(th.STATES)

    class H(th.FakeHA):
        def __call__(self, method, url, body, headers, timeout):
            self.calls.append((method, url, body, headers))
            return list(states) if url.endswith("/api/states") else []
    f = H()
    s = SmartHome(SmartHomeConfig.from_dict({"url": "http://ha.local:8123/", "token": "t"}), http=f)
    s.f = f
    return s


def t1_entity_id():
    print("\n=== The entity id counts ===")
    s = home([
        {"entity_id": "switch.garage_door_opener", "state": "off", "attributes": {"friendly_name": "Opener"}},
        {"entity_id": "switch.well_pump", "state": "on", "attributes": {"friendly_name": "Well"}},
        {"entity_id": "switch.boiler_relay", "state": "on", "attributes": {"friendly_name": "Hot water"}},
        {"entity_id": "switch.hall_siren", "state": "off", "attributes": {"friendly_name": "Hall"}},
        {"entity_id": "switch.desk_lamp", "state": "off", "attributes": {"friendly_name": "Desk lamp"}},
    ])
    for name, ent in (("opener", "garage door opener"), ("well", "well pump"), ("hot water", "boiler"), ("hall", "siren")):
        say = s.hear(f"turn on the {name}")
        check(f"'{name}' ({ent}) is not operated by voice", say and "don't operate" in say
              and not any(c[0] == "POST" for c in s.f.calls), str(say))
    say = s.hear("turn on the desk lamp")
    check("an ordinary switch still is", any(c[0] == "POST" for c in s.f.calls), str(say))


def t2_ambiguous():
    print("\n=== Two that fit equally ===")
    s = home([
        {"entity_id": "light.living_room", "state": "off", "attributes": {"friendly_name": "Living room light"}},
        {"entity_id": "light.desk", "state": "off", "attributes": {"friendly_name": "Desk light"}},
        {"entity_id": "light.hall", "state": "off", "attributes": {"friendly_name": "Hall light"}},
    ])
    say = s.hear("turn off the light")
    check("'turn off the light' with three lights asks which, and switches none",
          say and say.startswith("Which one") and "Desk light" in say and not any(c[0] == "POST" for c in s.f.calls), str(say))
    s.f.calls.clear()
    s.hear("turn off the desk light")
    check("a name that fits one is done", any(c[0] == "POST" for c in s.f.calls))
    s2 = home([
        {"entity_id": "light.a", "state": "off", "attributes": {"friendly_name": "Reading light"}},
        {"entity_id": "lock.front", "state": "locked", "attributes": {"friendly_name": "Front light lock"}},
    ])
    say = s2.hear("turn on the light")
    check("a refused device tied with a safe one is not a reason to ask (and is never the one chosen)",
          not (say or "").startswith("Which one") or "lock" not in (say or "").lower(), str(say))


def t3_not_a_device():
    print("\n=== Not a home device ===")
    s = home(th.STATES)
    for t in ("turn off the monitor", "turn off the wifi", "switch off the screen", "turn on the radio"):
        check(f"{t!r} is left for the handlers that can do it", s.hear(t) is None, str(s.hear(t)))
    check("a missing home device is still said plainly", "couldn't find anything called attic lights" in (s.hear("turn on the attic lights") or ""))


if __name__ == "__main__":
    t1_entity_id()
    t2_ambiguous()
    t3_not_a_device()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
