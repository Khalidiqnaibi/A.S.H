"""
tst_situation.py -- one well-timed line from everything ASH notices, by rule.

  S1  a meeting soon (and "you're still gaming"); a meeting that started while you're not working; once per event
  S2  battery low / critical, not when plugged in, not again for half an hour
  S3  memory nearly full, only after two readings in a row
  S4  birthdays from notes: today in the day, tomorrow in the evening; parsed from the ways people say dates
  S5  late at night while coding or gaming, once a night
  S6  only the strongest line at a time
  S7  in the runtime: offered through the gate (said as written, no model), not while muted or paused

Run:  PYTHONIOENCODING=utf-8 python tests/tst_situation.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.situation import Situation, World, birthday_of, birthdays_from  # noqa: E402

check = tw.check


def at(h, m=0, day=12, month=11):
    return datetime(2026, month, day, h, m).timestamp()


def sit():
    return Situation()


def s1_meetings():
    print("\n=== S1. Meetings ===")
    now = at(14, 50)
    s = sit()
    n = s.tick(World(now=now, idle_s=10, activity="gaming", events=[(at(15, 0), "Standup")]))
    check("soon, while gaming: says both", n is not None and n.say == "Heads up: Standup starts in 10 minutes -- and you're still gaming.",
          str(n and n.say))
    check("once per event", s.tick(World(now=now + 60, idle_s=10, activity="gaming", events=[(at(15, 0), "Standup")])) is None)
    n = sit().tick(World(now=now, idle_s=10, activity="coding", events=[(at(15, 0), "Standup")]))
    check("soon, while working: just the heads-up", n is not None and n.say == "Heads up: Standup starts in 10 minutes.", str(n and n.say))
    check("an hour away: nothing yet", sit().tick(World(now=at(14, 0), idle_s=10, events=[(at(15, 0), "Standup")])) is None)
    n = sit().tick(World(now=at(15, 2), idle_s=10, activity="gaming", events=[(at(15, 0), "Standup")]))
    check("started 2 minutes ago, still gaming", n is not None and n.say == "Standup started 2 minutes ago.", str(n and n.say))
    check("started, but you're working (probably in it): nothing", sit().tick(World(now=at(15, 2), idle_s=10, activity="working",
                                                                                     events=[(at(15, 0), "Standup")])) is None)
    check("started, but you're away: nothing", sit().tick(World(now=at(15, 2), idle_s=900, events=[(at(15, 0), "Standup")])) is None)


def s2_battery():
    print("\n=== S2. Battery ===")
    s = sit()
    n = s.tick(World(now=at(10), battery=(12.0, False)))
    check("low: worth plugging in", n is not None and n.say == "The battery's at 12% -- worth plugging in." and n.urgency < 0.9, str(n))
    check("not again for half an hour", s.tick(World(now=at(10, 10), battery=(11.0, False))) is None)
    n = s.tick(World(now=at(10, 31), battery=(10.0, False)))
    check("...then again", n is not None, str(n))
    n = sit().tick(World(now=at(10), battery=(4.0, False)))
    check("critical: urgent", n is not None and "plug in now" in n.say and n.urgency >= 0.9, str(n))
    check("plugged in: nothing", sit().tick(World(now=at(10), battery=(4.0, True))) is None)
    check("a desktop with no battery: nothing", sit().tick(World(now=at(10), battery=None)) is None)


def s3_memory():
    print("\n=== S3. Memory ===")
    s = sit()
    check("one high reading: nothing yet", s.tick(World(now=at(10), mem_percent=95)) is None)
    n = s.tick(World(now=at(10, 1), mem_percent=96))
    check("two in a row: said, pointing at the PC doctor", n is not None and "96% full" in n.say and "why is my computer slow" in n.say, str(n))
    s2 = sit()
    s2.tick(World(now=at(10), mem_percent=95))
    s2.tick(World(now=at(10, 1), mem_percent=60))
    check("a dip in between starts again", s2.tick(World(now=at(10, 2), mem_percent=95)) is None)


def s4_birthdays():
    print("\n=== S4. Birthdays from notes ===")
    for back, want in [("your sister's birthday is on May 3rd", ("your sister's", 5, 3)), ("your birthday is 12 March", ("your", 3, 12)),
                       ("Omar's birthday is the 5th of June", ("Omar's", 6, 5)), ("mom's birthday is on Sept 9", ("mom's", 9, 9)),
                       ("your sister's birthday is next week", None), ("the wifi password is hunter2", None)]:
        check(f"{back!r} -> {want}", birthday_of(back) == want, str(birthday_of(back)))
    b = [("your sister's", 11, 12)]
    n = sit().tick(World(now=at(10, day=12), idle_s=5, birthdays=b))
    check("today, during the day", n is not None and n.say == "It's your sister's birthday today.", str(n and n.say))
    check("today, at 3 in the morning: not then", sit().tick(World(now=at(3, day=12), idle_s=5, birthdays=b)) is None)
    n = sit().tick(World(now=at(19, day=11), idle_s=5, birthdays=b))
    check("tomorrow, in the evening", n is not None and n.say == "Your sister's birthday is tomorrow.", str(n and n.say))
    check("away from the computer: not said", sit().tick(World(now=at(10, day=12), idle_s=3600, birthdays=b)) is None)
    s = sit()
    s.tick(World(now=at(10, day=12), idle_s=5, birthdays=b))
    check("once a day", s.tick(World(now=at(12, day=12), idle_s=5, birthdays=b)) is None)
    from src.daemon.notes import Notes
    nt = Notes(None)
    nt.keep_note("my sister's birthday is on May 3rd")
    nt.keep_note("I parked on level 3")
    check("read from the notes", birthdays_from(nt.items) == [("your sister's", 5, 3)], str(birthdays_from(nt.items)))


def s5_late():
    print("\n=== S5. Late at night ===")
    s = sit()
    n = s.tick(World(now=at(1, 30), idle_s=4, activity="coding"))
    check("past 1, still coding", n is not None and n.say == "It's past 1 -- the coding can wait till morning?", str(n and n.say))
    check("once a night", s.tick(World(now=at(2, 30), idle_s=4, activity="coding")) is None)
    check("not when just browsing", sit().tick(World(now=at(1, 30), idle_s=4, activity="leisure")) is None)
    check("not in the evening", sit().tick(World(now=at(23, 30), idle_s=4, activity="coding")) is None)


def s6_one_at_a_time():
    print("\n=== S6. The strongest one ===")
    s = sit()
    w = World(now=at(14, 50), idle_s=5, activity="gaming", events=[(at(15, 0), "Standup")], battery=(4.0, False))
    n = s.tick(w)
    check("critical battery beats a meeting", n is not None and "plug in now" in n.say, str(n))
    n = s.tick(World(now=at(14, 51), idle_s=5, activity="gaming", events=[(at(15, 0), "Standup")], battery=(4.0, False)))
    check("...and the meeting comes next", n is not None and "Standup" in n.say, str(n))


def s7_runtime():
    print("\n=== S7. In the runtime ===")
    import tst_effects as te
    rt = te.rig(situation=True, situation_probe=lambda: ((9.0, False), 50.0))
    n = len(rt.ash.calls)
    for _ in range(3):
        rt.t[0] += 31
        rt._tick()
    said = [o for o in rt.out if "battery" in o]
    check("offered through the gate and said as written, no model", said == ["The battery's at 9% -- worth plugging in."]
          and len(rt.ash.calls) == n, str(rt.out))
    rt2 = te.rig(situation=True, situation_probe=lambda: ((9.0, False), 50.0))
    rt2.gate.mute(True)
    for _ in range(3):
        rt2.t[0] += 31
        rt2._tick()
    check("muted: nothing", not any("battery" in o for o in rt2.out), str(rt2.out))
    check("status counts it", rt.status()["situation"]["offered"] == 1, str(rt.status()["situation"]))


if __name__ == "__main__":
    for fn in (s1_meetings, s2_battery, s3_memory, s4_birthdays, s5_late, s6_one_at_a_time, s7_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
