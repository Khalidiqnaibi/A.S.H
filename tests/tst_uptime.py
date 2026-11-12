"""
tst_uptime.py -- were the crashes while ASH was running?

The Windows log can say WHEN the computer crashed; nothing said whether ASH was
running at the time. ASH now writes down when it starts, keeps a heartbeat, and
notes a clean stop. A run that never recorded a clean stop was cut off, and the
crash is what cut it (Windows stamps the crash at the reboot, so the run counts
until the next start). Then, with a day of history:

  U1  Runs are kept in their own file (so "first read is history" still holds)
  U2  Heartbeats update the run and are written at most once a minute; a clean
      stop records an end
  U3  A crash counts as "ASH running" inside a run, or in the gap after a run
      that was cut off -- not after a clean stop
  U4  "Is ASH causing the crashes?" answers from that, and says link, not cause
  U5  The ordinary crash answer gets the same sentence
  U6  The runtime beats and stops cleanly

Run:  PYTHONIOENCODING=utf-8 python tests/tst_uptime.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
DAY, HOUR = 86400.0, 3600.0
T0 = 1_800_000_000.0


def make(path=None, log=(), now=T0):
    from src.daemon.crashwatch import CrashWatch
    t = [now]
    c = CrashWatch(path=path, clock=lambda: t[0], reader=lambda: list(log), background=False)
    c.t = t
    return c


def u1_files():
    print("\n=== U1. Its own file ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "crashes.json")
    c = make(p, log=[{"t": T0 - 10 * DAY, "code": 0}])
    check("a run is written at once", os.path.exists(p + ".runs") and len(c.runs) == 1)
    check("the told-crashes file is NOT written yet", not os.path.exists(p))
    c2 = make(p, log=[{"t": T0 - 10 * DAY, "code": 0}], now=T0 + HOUR)
    c2.refresh(force=True)
    check("so on a first read the old crash is still history, not news", c2.briefing_part("morning", T0 + HOUR) is None)
    check("the earlier run is remembered", len(c2.runs) == 2 and c2.runs[0]["start"] == T0)
    off = make(None)
    check("no path: nothing is tracked", off.runs == [] and off.overlap(T0) is None)
    from src.daemon.crashwatch import CrashWatch
    dis = CrashWatch(path=os.path.join(d, "x.json"), enabled=False, clock=lambda: T0, reader=lambda: [], background=False)
    check("disabled: nothing is tracked", dis.runs == [] and not os.path.exists(os.path.join(d, "x.json.runs")))
    old = os.path.join(d, "o.json")
    with open(old + ".runs", "w") as fh:
        json.dump({"runs": [{"start": T0 - 200 * DAY, "beat": T0 - 199 * DAY}, {"start": T0 - 5 * DAY, "beat": T0 - 4 * DAY}]}, fh)
    c3 = make(old)
    check("runs older than 90 days are dropped", [r["start"] for r in c3.runs[:-1]] == [T0 - 5 * DAY], str(c3.runs))
    with open(old + ".runs", "w") as fh:
        fh.write("{not json")
    c4 = make(old)
    check("a damaged file is survived", len(c4.runs) == 1)


def u2_beats():
    print("\n=== U2. Heartbeat and clean stop ===")
    p = os.path.join(tempfile.mkdtemp(), "crashes.json")
    c = make(p)
    c.t[0] = T0 + 10
    c.heartbeat()
    on_disk = json.load(open(p + ".runs"))["runs"][-1]
    check("a beat within the minute is held in memory, not rewritten", c.runs[-1]["beat"] == T0 + 10
          and on_disk["beat"] == T0)
    c.t[0] = T0 + 61
    c.heartbeat()
    check("after a minute it is written", json.load(open(p + ".runs"))["runs"][-1]["beat"] == T0 + 61)
    c.t[0] = T0 + 100
    c.shutdown()
    last = json.load(open(p + ".runs"))["runs"][-1]
    check("a clean stop records its end", last["end"] == T0 + 100 and last["beat"] == T0 + 100)


def runs_of(*spec):
    return [dict(r) for r in spec]


def u3_overlap():
    print("\n=== U3. Was it running? ===")
    now = T0 + 20 * DAY
    c = make(None, now=now)
    c.runs = runs_of({"start": T0, "beat": T0 + 2 * DAY, "end": T0 + 2 * DAY},       # clean
                     {"start": T0 + 3 * DAY, "beat": T0 + 3.2 * DAY},                  # cut off (no end)
                     {"start": T0 + 5 * DAY, "beat": T0 + 6 * DAY, "end": T0 + 6 * DAY},
                     {"start": T0 + 10 * DAY, "beat": now})                            # still running
    c.events = [{"t": T0 + 1 * DAY, "code": 0},            # inside run 1
                {"t": T0 + 2.5 * DAY, "code": 0},          # after a clean stop: not running
                {"t": T0 + 4.5 * DAY, "code": 0x1A},       # the gap after the cut-off run: it was running
                {"t": T0 + 8 * DAY, "code": 0},            # after a clean stop: not running
                {"t": T0 + 12 * DAY, "code": 0}]           # inside the current run
    c.events.sort(key=lambda e: -e["t"])
    o = c.overlap(now, days=30)
    check("five crashes counted", o["crashes"] == 5, str(o))
    check("...ASH was running through three of them", o["with_ash"] == 3, str(o))
    ran = 2 + 2 + 1 + 10   # days: run1 (2), cut-off run to next start (2), run3 (1), current (10)
    check("how much of the time it runs", abs(o["share"] - ran / 20.0) < 0.01, f"{o['share']:.3f}")
    young = make(None, now=T0 + 0.5 * DAY)
    young.runs = [{"start": T0, "beat": T0 + 0.5 * DAY}]
    check("under a day of history: no answer", young.overlap(T0 + 0.5 * DAY) is None)
    c.events.insert(0, {"t": T0 - 3 * DAY, "code": 0})
    check("a crash from before ASH was keeping track is not counted", c.overlap(now)["crashes"] == 5)


def u4_cause():
    print("\n=== U4. 'Is ASH causing them?' ===")
    now = T0 + 20 * DAY
    from src.daemon.crashwatch import CrashWatch

    def watch(events, runs):
        c = CrashWatch(clock=lambda: now, reader=lambda: list(events), background=False)
        c.runs = runs
        c.refresh(force=True)
        return c
    # ASH runs 25% of the time, and every crash was in that quarter
    runs = [{"start": T0 + 10 * DAY, "beat": T0 + 15 * DAY, "end": T0 + 15 * DAY}]
    ev = [{"t": T0 + (10.5 + i) * DAY, "code": 0} for i in range(4)]
    c = watch(ev, runs)
    say = c.hear("is ash causing the crashes?", now)
    check("every crash with ASH running, and it runs half the time: worth a test", "during 4 of those 4" in say
          and "switching me off" in say, say)
    check("...and it says link, not cause", "not a cause" in say, say)
    # runs the whole time, crashes spread
    runs = [{"start": T0, "beat": now}]
    ev = [{"t": T0 + (3 + i * 3) * DAY, "code": 0} for i in range(4)]
    say = watch(ev, runs).hear("did you crash my computer", now)
    check("running all the time: nothing points at it", "nothing here points at me" in say, say)
    say = watch([], runs).hear("are you the cause of my crashes", now)
    check("no crashes: nothing to blame", say.startswith("No unexpected shutdowns"), say)
    say = watch([{"t": T0 + 5 * DAY, "code": 0}], runs).hear("is ash causing the crashes", now)
    check("one crash: too few", "too few" in say, say)
    young = CrashWatch(clock=lambda: now, reader=lambda: [{"t": now - 3600, "code": 0}, {"t": now - 7200, "code": 0}],
                       background=False)
    young.runs = [{"start": now - HOUR, "beat": now}]
    young.refresh(force=True)
    say = young.hear("is ash causing the crashes", now)
    check("no history yet: says to ask again", "ask again" in say, say)
    for text in ("how do I cause a crash in my code", "is the game causing it", "did you break my sandwich"):
        check(f"not the question: {text[:30]!r}", watch([], runs).hear(text, now) is None)


def u5_ordinary():
    print("\n=== U5. The ordinary answer ===")
    now = T0 + 20 * DAY
    from src.daemon.crashwatch import CrashWatch
    ev = [{"t": now - 3 * HOUR, "code": 0xA}, {"t": now - 5 * DAY, "code": 0}, {"t": now - 12 * DAY, "code": 0}]
    c = CrashWatch(clock=lambda: now, reader=lambda: list(ev), background=False)
    c.runs = [{"start": T0, "beat": now}]
    c.refresh(force=True)
    say = c.hear("any crashes lately", now)
    check("the usual answer gains the sentence", "ASH was running during 3 of those 3" in say, say)
    c.runs = []
    check("without history: the usual answer, unchanged", "ASH was running" not in c.hear("any crashes lately", now))


def u6_runtime():
    print("\n=== U6. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import PeripheryRegistry
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    rt = AmbientRuntime(ash=tw.Ash(), periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
                        gate=AttentionGate(), privacy=PrivacyPolicy(),
                        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, crashes=True, crashes_path=p),
                        response_sink=lambda t, m: None)
    run = rt.crashwatch.runs[-1]
    first = run["beat"]
    time.sleep(0.05)
    rt._tick()
    check("a tick is a heartbeat", run["beat"] > first)
    check("(no clean end yet)", "end" not in run)
    rt.stop(timeout=1)
    check("a stop is a clean end", "end" in rt.crashwatch.runs[-1])


if __name__ == "__main__":
    for fn in (u1_files, u2_beats, u3_overlap, u4_cause, u5_ordinary, u6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
