"""tst_startquiet.py -- 26 Nov after each restart: "How are you holding up today?" 55 s in (asked at 14:45 already), "Still curious: ..." 34 s in,
"Memory is 93% full" 90 s in (said at 20:41 already, beside the governor's own notice), and never "I'm back." Now: nothing volunteered in the first
three minutes, the mood check once a day across restarts, one memory notice kept across restarts, and "I'm back." after a restart that was asked
for. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_startquiet.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.freemoment import FreeMoments, MoodCheck  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402
from src.senses.base import Modality, SensorEvent  # noqa: E402

check = tw.check


def offer(rt, say):
    n = len(rt._out)
    rt.submit(SensorEvent(source="flow", modality=Modality.TEXT, text=say, salience=0.6, urgency=0.3, ts=rt.clock(), persistable=False,
                          meta={"offer": "situation", "offer_strength": 0.9, "say": say, "topic": "situation", "situation": "test"}))
    rt._tick()
    return rt._out[n:]


def rig():
    rt = tm.rig()
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    rt._out = out
    d = tempfile.mkdtemp()
    rt.cfg.notices_path = os.path.join(d, "notices.json")
    rt._restart_checked = True
    return rt


def main():
    print("\n=== Q1. The mood check: once a day, across restarts ===")
    p = os.path.join(tempfile.mkdtemp(), "moments.json")
    nine = time.mktime((2026, 11, 26, 9, 0, 0, 0, 0, -1))
    t = [nine]
    env = Environment(idle_s=60.0, free_moment=True)
    fm = FreeMoments(clock=lambda: t[0], path=p)
    mc = fm.register(MoodCheck(fm))
    check("offered when nothing is known", mc.propose(env, t[0]) is not None)
    fm.mark_spoken("mood_check", t[0])
    t[0] += 6 * 3600                                         # unanswered; a restart later the same day
    fm2 = FreeMoments(clock=lambda: t[0], path=p)
    mc2 = fm2.register(MoodCheck(fm2))
    check("after a restart the same day: not asked again (26 Nov 14:45 and 20:43)", mc2.propose(env, t[0]) is None and fm2._last_by.get("mood_check") == nine,
          str(fm2._last_by))
    t[0] = nine + 86400
    check("the next day: asked", mc2.propose(env, t[0]) is not None)

    print("\n=== Q2. Nothing volunteered in the first minutes ===")
    rt = rig()
    rt.started_at = time.time() - 3600
    said_late = offer(rt, "Memory is 93% full, so things may slow down.")
    rt = rig()
    rt.started_at = time.time() - 20
    said_early = offer(rt, "Memory is 93% full, so things may slow down.")
    check("an hour in, an offer is said", said_late == ["Memory is 93% full, so things may slow down."], str(said_late))
    check("twenty seconds after a start, it is held", said_early == [], str(said_early))
    used, said = rt.say("what time is it")
    check("...while what is asked is still answered", said, str(said))

    print("\n=== Q3. One memory notice, kept across restarts ===")
    rt = rig()
    rt.governor = types.SimpleNamespace(mem=95.0, cpu=20.0, status=lambda: {})
    w = rt._situation_world(rt.clock(), 0.0)
    check("with the governor running, the situation's own memory line is off", w.mem_percent is None, str(w.mem_percent))
    now = rt.clock()
    rt._notices(memory_told_at=now - 600)
    rt._mem_high_since = now - 1300
    n = len(rt._out)
    rt._memory_notice(now)
    check("said ten minutes ago, before a restart: not said again", rt._out[n:] == [], str(rt._out[n:]))
    rt = rig()
    rt.governor = types.SimpleNamespace(mem=95.0, cpu=20.0, status=lambda: {})
    rt._notices(memory_told_at=now - 4 * 3600)
    rt._mem_high_since = now - 1300
    n = len(rt._out)
    rt._memory_notice(now)
    check("four hours on: said once", len(rt._out[n:]) == 1 and rt._out[n].startswith("Memory has been above 90%"), str(rt._out[n:]))
    check("...and when is kept", abs(float(rt._notices().get("memory_told_at")) - now) < 1)

    print("\n=== Q4. 'I'm back.' after a restart that was asked for ===")
    rt = rig()
    rt.request_restart = lambda: None
    rt.say("restart yourself")
    rt.say("yes")
    at = rt._notices().get("restart_at")
    check("confirming a restart marks it", at is not None and abs(time.time() - float(at)) < 30, str(rt._notices()))
    rt2 = rig()
    rt2.cfg.notices_path = rt.cfg.notices_path
    rt2._restart_checked = False
    n = len(rt2._out)
    rt2._social_step(rt2.clock())
    check("the next start says 'I'm back.'", rt2._out[n:] == ["I'm back."], str(rt2._out[n:]))
    check("...once", rt2._notices().get("restart_at") is None)
    rt3 = rig()
    rt3._notices(restart_at=time.time() - 3600)
    rt3._restart_checked = False
    n = len(rt3._out)
    rt3._social_step(rt3.clock())
    check("a mark an hour old (a crash, a manual start) says nothing", rt3._out[n:] == [], str(rt3._out[n:]))
    rt4 = rig()
    rt4._restart_checked = False
    n = len(rt4._out)
    rt4._social_step(rt4.clock())
    check("an ordinary start says nothing", rt4._out[n:] == [], str(rt4._out[n:]))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
