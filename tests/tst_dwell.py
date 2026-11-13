"""
tst_dwell.py -- where the user is and for how long: reading, away, back, and the day's time.

  D1  the mode: typing / pausing / reading / away; a long read is not "away"
  D2  reading: the same window, hands off, long enough; never for a private window
  D3  away and back: how long, once; the hour gone is not time on the window left open
  D4  the day: time by program and window, across window changes; kept across a restart
  D5  what is said: the line for the model, "what have I been doing?", "how long was I away?"; private windows never named
  D6  in the runtime: the model is given where the user is; a return is welcomed; reading is a free moment; asked by rule

Run:  PYTHONIOENCODING=utf-8 python tests/tst_dwell.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.dwell import Dwell  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 10, 0, 0, 0, 0, -1))


def mk(**kw):
    t = [T0]
    d = Dwell(clock=lambda: t[0], **kw)
    d.t = t
    return d


def run(d, seconds, idle, step=10.0):
    """Let time pass with the keyboard idle for `idle` seconds at the start (and growing)."""
    out = None
    n = int(seconds / step)
    for i in range(n):
        d.t[0] += step
        b = d.tick(d.t[0], idle + (i + 1) * step if idle >= 0 else 0.0)
        out = b or out
    return out


def d1_modes():
    print("\n=== D1. Modes ===")
    d = mk()
    d.window("msedge.exe", "Body blueprints", now=T0)
    check("no tick yet: unknown", d.mode(T0) == "unknown")
    d.tick(T0, 1.0)
    check("input a second ago: typing", d.mode(T0) == "typing")
    d.tick(T0, 12.0)
    check("a short gap: pausing", d.mode(T0) == "pausing")
    d.tick(T0, 90.0)
    check("a minute and a half, no input: reading", d.mode(T0) == "reading")
    d.tick(T0, 580.0)
    check("nearly ten minutes: still reading, not gone", d.mode(T0) == "reading")
    d.tick(T0, 601.0)
    check("past ten minutes: away", d.mode(T0) == "away")
    d.tick(T0, None)
    check("an unknown idle changes nothing", d.mode(T0) == "away")
    short = mk(away_s=300)
    short.tick(T0, 301.0)
    check("the away line is a setting", short.mode(T0) == "away")


def d2_reading():
    print("\n=== D2. Reading ===")
    d = mk(reading_min_s=240)
    d.window("msedge.exe", "A.S.H body blueprints", now=T0)
    d.t[0] = T0 + 60
    d.tick(d.t[0], 40.0)
    check("a minute in: not yet a thing", not d.reading(d.t[0]))
    d.t[0] = T0 + 300
    d.tick(d.t[0], 60.0)
    check("five minutes in one window, hands off: reading", d.reading(d.t[0]))
    d.tick(d.t[0], 3.0)
    check("typing again: not reading", not d.reading(d.t[0]))
    d.tick(d.t[0], 700.0)
    check("gone: not reading", not d.reading(d.t[0]))
    p = mk()
    p.window("whatsapp.exe", "Omar", now=T0, private=True)
    p.t[0] = T0 + 600
    p.tick(p.t[0], 60.0)
    check("a private window is never a reading moment", not p.reading(p.t[0]))
    b = mk()
    b.window("chrome.exe", "Bank", now=T0, blocked=True)
    check("nor a blocked one, and its title is not kept", b.subject == "" and b.private)
    s = mk()
    s.window("msedge.exe", "Page one", now=T0)
    s.t[0] = T0 + 200
    s.window("msedge.exe", "Page two", now=s.t[0])
    s.t[0] += 100
    s.tick(s.t[0], 50.0)
    check("a window change restarts the clock", not s.reading(s.t[0]))


def d3_away_back():
    print("\n=== D3. Away and back ===")
    d = mk()
    d.window("msedge.exe", "Blueprints", now=T0)
    d.tick(T0 + 5, 3.0)
    d.t[0] = T0 + 5
    gone = None
    for i in range(1, 8):
        d.t[0] = T0 + 5 + i * 300
        gone = d.tick(d.t[0], i * 300.0)
        check(f"idle {i * 5} minutes: no Back yet", gone is None)
    check("away since the input stopped, not since it was noticed", d.away_from is not None and abs(d.away_from - (T0 + 5)) < 1)
    d.t[0] += 3
    b = d.tick(d.t[0], 0.5)
    check("typing again: a Back, with how long", b is not None and 2000 < b.away_s < 2200, str(b))
    check("said once", d.tick(d.t[0] + 3, 3.0) is None)
    check("the welcome in words", d.welcome_back(b).startswith("Welcome back. You were away about 35 minutes"), d.welcome_back(b))
    check("the hour gone is not time on the window", d.minutes_here(d.t[0]) < 1.0, f"{d.minutes_here(d.t[0]):.1f}")
    check("'how long was I away?' says so", "away about 35 minutes" in d.answer("away", d.t[0]), d.answer("away", d.t[0]))
    check("and the time gone was not counted on the window",
          sum(s for _a, _s, s in d.today(d.t[0], include_current=False)) < 400, str(d.today(d.t[0], False)))
    quick = mk()
    quick.tick(T0, 100.0)
    check("a coffee-length gap is not an away", quick.tick(T0 + 20, 0.5) is None)
    d2 = mk()
    d2.tick(T0, 700.0)
    check("away now: the answer says so, and that it may be reading", "away about" in d2.answer("away", T0 + 1) and "reading" in d2.answer("away", T0 + 1))


def d4_day():
    print("\n=== D4. The day ===")
    path = os.path.join(tempfile.mkdtemp(), "dwell.json")
    d = mk(path=path)
    d.window("Code.exe", "brain.py", now=T0)
    d.tick(T0 + 1, 1.0)
    d.t[0] = T0 + 40 * 60
    d.window("msedge.exe", "A.S.H blueprints", now=d.t[0])
    d.t[0] += 15 * 60
    d.tick(d.t[0], 30.0)
    rows = d.today(d.t[0])
    check("time by window, biggest first", rows[0][1] == "brain.py" and abs(rows[0][2] - 2400) < 1 and rows[1][1] == "A.S.H blueprints", str(rows))
    check("the window in front counts up to now", abs(rows[1][2] - 900) < 1)
    d.t[0] += 10
    d.window("Code.exe", "brain.py", now=d.t[0])
    rows = d.today(d.t[0])
    check("coming back to a window adds up", abs([r for r in rows if r[1] == "brain.py"][0][2] - 2400) < 15, str(rows))
    d.save()
    e = Dwell(path=path, clock=lambda: d.t[0])
    check("kept across a restart", e.today(d.t[0], include_current=False)[0][1] == "brain.py", str(e.today(d.t[0], False)))
    tiny = mk()
    tiny.window("a.exe", "x", now=T0)
    tiny.window("b.exe", "y", now=T0 + 2)
    check("a two-second glance is not kept", tiny.today(T0 + 3, include_current=False) == [])
    old = mk()
    for i in range(10):
        old.t[0] = T0 + i * 86400
        old.window("Code.exe", f"f{i}", now=old.t[0])
        old.t[0] += 600
        old.window("x.exe", "z", now=old.t[0])
    check("only a week is kept", len(old.day) <= 7, str(len(old.day)))
    old.forget()
    check("and it can all be forgotten", old.day == {} and old.recent == [])


def d5_said():
    print("\n=== D5. What is said ===")
    d = mk()
    d.window("msedge.exe", "A.S.H body blueprints", now=T0)
    d.tick(T0, 1.0)
    d.t[0] = T0 + 25 * 60
    d.tick(d.t[0], 120.0)
    c = d.context(d.t[0])
    check("the model is told where they are and for how long", "A.S.H body blueprints" in c and "Edge" in c and "25 minutes" in c, c)
    check("and that they are not typing", "reading or watching" in c, c)
    d.window("Code.exe", "brain.py", now=d.t[0])
    d.t[0] += 60
    d.tick(d.t[0], 2.0)
    c2 = d.context(d.t[0])
    check("what came before is in it", "Earlier today" in c2 and "A.S.H body blueprints" in c2, c2)
    d.window("whatsapp.exe", "Omar - Chat", now=d.t[0], private=True)
    d.t[0] += 120
    d.tick(d.t[0], 2.0)
    c3 = d.context(d.t[0])
    check("a private window is named as a program only", "Omar" not in c3 and "WhatsApp" in c3 or "Whatsapp" in c3.title(), c3)
    check("and never in the day's list", all("Omar" not in r[1] for r in d.today(d.t[0])), str(d.today(d.t[0])))
    d.tick(d.t[0], 900.0)
    check("away: nothing said about it", d.context(d.t[0]) == "")
    for text, kind in (("how long have I been on this?", "here"), ("how long have I been here", "here"), ("what have I been doing today?", "doing"),
                       ("what was I doing", "doing"), ("what did I do this morning", "doing"), ("how long was I gone?", "away"),
                       ("when did I get back", "away")):
        check(f"{text!r} is read as {kind}", d.parse(text) == kind, str(d.parse(text)))
    for text in ("how long is the movie", "what is the weather doing", "how long have I known Omar", "what was I saying about Omar"):
        check(f"{text!r} is not", d.parse(text) is None, str(d.parse(text)))
    d2 = mk()
    d2.window("Code.exe", "brain.py", now=T0)
    d2.t[0] += 3 * 60
    d2.window("msedge.exe", "A.S.H body blueprints", now=d2.t[0])
    d2.t[0] += 20 * 60
    d2.tick(d2.t[0], 30.0)
    a = d2.answer("doing", d2.t[0])
    check("'what have I been doing?' lists the day", a.startswith("So far today:") and "A.S.H body blueprints in Edge (20 minutes)" in a, a)
    check("'how long have I been here?'", d2.answer("here", d2.t[0]).startswith("About 20 minutes on A.S.H body blueprints (Edge)"), d2.answer("here", d2.t[0]))
    check("nothing seen yet: said so", mk().answer("here") == "I haven't seen you switch windows yet.")


def rig(**extra):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    cfg = dict(moments=True, moments_cfg={"jokes": False, "mood_check": False, "min_gap_s": 0}, dwell=True, dwell_path=os.path.join(d, "dwell.json"), asks=True, asks_dir=os.path.join(d, "asks"),
               wonder=True, wonder_cfg={"min_gap_s": 0}, wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json"))
    cfg.update(extra)
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=20, quiet_hours=None),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0, **cfg),
                        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9, urgency=0.6))
        rt._tick()

    def window(process, subject, private=False):
        rt.submit(SensorEvent(source="active_window", modality=Modality.TEXT, text=f"switched to {subject}", salience=0.35, ts=clock[0],
                              meta={"process": process, "subject": subject, "project": "", "blocked": False, "dwell_s": 5.0, "private": private}))
        rt._tick()
    h.step, h.say, h.window = step, say, window
    h.said = lambda: [o[0] for o in out]
    return h


def d6_runtime():
    print("\n=== D6. In the runtime ===")
    h = rig()
    seen = {}
    real = h.ash.think

    def think(query, image=None, context=""):
        seen["context"] = context
        return real(query, image=image, context=context)
    h.ash.think = think
    h.window("msedge.exe", "A.S.H body blueprints")
    h.step(25 * 60, 100.0)
    h.say("what do you think of this layout")
    check("the model is told which window they are in, and for how long", "A.S.H body blueprints" in seen.get("context", "") and "25 minutes" in seen.get("context", ""),
          seen.get("context", "")[:200])
    n = len(h.ash.calls)
    h.say("how long have I been on this?")
    check("'how long have I been on this' by rule, no model", len(h.ash.calls) == n and h.said()[-1].startswith("About 25 minutes on A.S.H body blueprints"), h.said()[-1])
    h.say("what have I been doing today?")
    check("'what have I been doing' by rule", h.said()[-1].startswith("So far today:") and len(h.ash.calls) == n, h.said()[-1])
    from src.senses import address as _address
    _address.conversation.__init__()
    h.step(900, 30.0)
    h.say("how long have I been on this", source="microphone", addressed=False)
    check("overheard: not answered", h.said()[-1].startswith("So far today:"), h.said()[-1])

    h = rig()
    h.window("msedge.exe", "A.S.H body blueprints")
    h.step(5, 2.0)
    h.step(40 * 60, 2400.0)
    mid = len(h.said())
    check("being away says nothing while away", not any("Welcome back" in s for s in h.said()))
    h.step(3, 1.0)
    check("typing again: not over the top of them", not any("Welcome back" in s for s in h.said()))
    h.step(8, 8.0)
    check("coming back after forty minutes: welcomed, once", [s for s in h.said()[mid:] if s.startswith("Welcome back")] == ["Welcome back. You were away about 40 minutes."], str(h.said()[mid:]))
    h.step(3, 2.0)
    check("only once", len([s for s in h.said() if "Welcome back" in s]) == 1)
    h = rig()
    h.window("msedge.exe", "x")
    h.step(5, 2.0)
    h.step(12 * 60, 700.0)
    h.step(3, 1.0)
    h.step(8, 8.0)
    check("twelve minutes is long enough to be noticed", any("Welcome back" in s for s in h.said()), str(h.said()))
    h = rig()
    h.window("msedge.exe", "x")
    h.step(5, 2.0)
    h.step(11 * 60, 620.0)
    h.rt.cfg.dwell_welcome_min_s = 900
    h.step(3, 1.0)
    check("the welcome has a floor", not any("Welcome back" in s for s in h.said()))
    h = rig()
    h.rt.gate.mute(True)
    h.window("msedge.exe", "x")
    h.step(5, 2.0)
    h.step(40 * 60, 2400.0)
    h.step(3, 1.0)
    h.step(8, 8.0)
    check("muted: no welcome", not any("Welcome back" in s for s in h.said()))
    h = rig(dwell=False)
    check("off: nothing built", h.rt.dwell is None)


def d7_reading_moment():
    print("\n=== D7. A long read is a free moment ===")
    h = rig()
    h.window("msedge.exe", "A.S.H body blueprints")
    for _ in range(3):
        h.step(120, 100.0)
    check("six minutes in: too soon", not any("without typing" in s for s in h.said()), str(h.said()))
    for _ in range(8):
        if any("without typing" in s for s in h.said()):
            break
        h.step(120, 100.0)
    asked = [s for s in h.said() if "without typing" in s]
    check("after a long read ASH says what it noticed and offers a note", len(asked) == 1 and "A.S.H body blueprints" in asked[0]
          and asked[0].endswith("keep a note of?"), str(h.said()))
    check("the ask is written down, with where they were", h.rt.asks.open and h.rt.asks.open[-1].ctx.get("mode") == "reading"
          and "blueprints" in h.rt.asks.open[-1].ctx.get("subject", ""), str(h.rt.asks.open))
    n = len(h.ash.calls)
    h.step(5, 100.0)
    h.say("yes")
    check("'yes' is understood as an answer to it (no model)", h.said()[-1].startswith("Go ahead") and len(h.ash.calls) == n, h.said()[-1])
    check("and settled in the ledger", h.rt.asks.recent and h.rt.asks.recent[-1].status in ("accepted", "answered"), str(h.rt.asks.recent[-1:]))
    for _ in range(10):
        h.step(120, 100.0)
    check("not twice in a row", len([s for s in h.said() if "without typing" in s]) == 1)

    p = rig()
    p.window("whatsapp.exe", "Omar - Chat", private=True)
    for _ in range(12):
        p.step(120, 100.0)
    check("a private window is never remarked on", not any("without typing" in s for s in p.said()), str(p.said()))
    m = rig()
    m.rt.gate.mute(True)
    m.window("msedge.exe", "A.S.H body blueprints")
    for _ in range(12):
        m.step(120, 100.0)
    check("muted: silent", not m.said())
    n = rig()
    n.window("msedge.exe", "A.S.H body blueprints")
    n.step(120, 100.0)
    for i in range(8):
        n.step(120, 900.0 + 120 * i)
    check("away is not reading: no remark about it", not any("without typing" in s for s in n.said()), str(n.said()))


if __name__ == "__main__":
    for fn in (d1_modes, d2_reading, d3_away_back, d4_day, d5_said, d6_runtime, d7_reading_moment):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
