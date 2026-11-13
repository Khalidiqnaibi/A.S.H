"""
tst_availability.py -- can the user talk right now? Free moments keep out of the moments they can't, and use the rest.

  V1  the judge: away, call, gaming, presenting, focus, talking, typing are CAN'T; coding and video are busy; the rest is free
  V2  how good a moment is: longer free is better
  V3  in the runtime: a user who is simply there gets a question within seconds; a gamer, a caller, a presenter, someone in
      focus mode, someone typing, someone away never does
  V4  it does not take two minutes of steady typing any more, and not every moment is used (budget, gaps, answers)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_availability.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.availability import judge  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))


def v1_judge():
    print("\n=== V1. The judge ===")
    cases = [
        (dict(process="msedge.exe", subject="Docs", idle_s=40), "free", True),
        (dict(process="msedge.exe", subject="Docs", idle_s=700), "away", False),
        (dict(process="zoom.exe", subject="Zoom", idle_s=40), "call", False),
        (dict(process="Teams.exe", subject="Weekly sync | Microsoft Teams", idle_s=40), "call", False),
        (dict(process="msedge.exe", subject="Standup - Google Meet", idle_s=40), "call", False),
        (dict(process="chrome.exe", subject="Design meeting notes - Google Docs", idle_s=40), "free", True),
        (dict(process="steam.exe", subject="Steam", idle_s=40), "gaming", False),
        (dict(process="eldenring.exe", subject="ELDEN RING", idle_s=30), "gaming", False),
        (dict(process="POWERPNT.EXE", subject="PowerPoint Slide Show - deck", idle_s=40), "presenting", False),
        (dict(process="Code.exe", subject="brain.py", idle_s=40, focus=True), "focusing", False),
        (dict(process="Code.exe", subject="brain.py", idle_s=40, conversation=True), "talking", False),
        (dict(process="Code.exe", subject="brain.py", idle_s=40, speaking=True), "talking", False),
        (dict(process="msedge.exe", subject="Docs", idle_s=2), "typing", False),
        (dict(process="Code.exe", subject="brain.py", idle_s=30), "working", True),
        (dict(process="Code.exe", subject="brain.py", idle_s=90), "free", True),
        (dict(process="winword.exe", subject="Report", idle_s=20), "working", True),
        (dict(process="vlc.exe", subject="movie.mkv", idle_s=100), "watching", True),
        (dict(process="msedge.exe", subject="Lo-fi beats - YouTube", idle_s=100), "watching", True),
        (dict(process="msedge.exe", subject="Some page", idle_s=100, video_page=True), "watching", True),
        (dict(process="msedge.exe", subject="A.S.H body blueprints", idle_s=100), "free", True),
        (dict(process="explorer.exe", subject="Downloads", idle_s=25), "free", True),
        (dict(process="msedge.exe", subject="x", idle_s=None), "unknown", False),
    ]
    for kw, state, can in cases:
        a = judge(**kw)
        check(f"{kw.get('process')} {kw.get('subject')!r} idle={kw.get('idle_s')}: {state}", a.state == state and a.can_talk == can, f"{a.state} {a.can_talk} {a.why}")
    a = judge("whatsapp.exe", "Zoom meeting with Omar", 40, private=True)
    check("a private window's title is never read (so it is not 'a meeting')", a.state == "free")
    check("only free counts as free", judge("explorer.exe", "x", 30).free and not judge("Code.exe", "x", 30).free)


def v2_score():
    print("\n=== V2. How good a moment ===")
    fresh = judge("msedge.exe", "Docs", 30, since_free_s=0)
    warm = judge("msedge.exe", "Docs", 30, since_free_s=200)
    check("longer free is a better moment", warm.score > fresh.score and warm.score == 1.0, f"{fresh.score} {warm.score}")
    check("busy is worth little", judge("Code.exe", "x", 20).score < 0.4 and judge("vlc.exe", "x", 90).score < 0.4)
    check("can't is worth nothing", judge("zoom.exe", "x", 40).score == 0.0)


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
    cfg = dict(moments=True, moments_cfg={"jokes": False, "mood_check": False, "min_gap_s": 0, "max_per_hour": 6},
               wonder=True, wonder_cfg={"min_gap_s": 0}, wonder_path=os.path.join(d, "w.json"),
               knowledge_path=os.path.join(d, "k.json"), dwell=True, dwell_path=os.path.join(d, "dwell.json"),
               asks=True, asks_dir=os.path.join(d, "asks"), free_after_s=20.0)
    cfg.update(extra)
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(max_offers_per_hour=30, quiet_hours=None),
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
    h.asked = lambda: [o for o in out if o[0] == "Who is Omar to you?"]
    return h


def seed(h):
    from src.senses import address as _address
    h.say("I told Omar something about work")
    h.step(5, 2.0)
    _address.conversation.__init__()                      # the exchange is over: they are on their own again


def v3_runtime():
    print("\n=== V3. In the runtime ===")
    h = rig()
    seed(h)
    h.window("msedge.exe", "A.S.H body blueprints")
    h.step(10, 12.0)
    check("ten seconds into a pause: not yet", not h.asked())
    h.step(15, 27.0)
    h.step(15, 42.0)
    check("a user who is simply there, hands off for 40 seconds, is asked -- no stretch of typing needed", len(h.asked()) == 1, str(h.out[-3:]))
    check("the availability is in the status", h.rt.status()["availability"]["state"] in ("free", "talking"), str(h.rt.status()["availability"]))

    def never(process, subject, label, **kw):
        g = rig(**kw)
        seed(g)
        g.window(process, subject)
        for i in range(12):
            g.step(20, 25.0 + i * 20)
        check(f"{label}: never asked", not g.asked(), str(g.out[-2:]))
        return g
    never("steam.exe", "Steam", "a game launcher in front")
    never("zoom.exe", "Zoom Meeting", "in a call")
    never("msedge.exe", "Standup - Google Meet", "in a Meet in the browser")
    never("POWERPNT.EXE", "PowerPoint Slide Show - deck", "presenting")
    g = rig()
    seed(g)
    g.window("Code.exe", "brain.py")
    for _ in range(10):
        g.step(20, 30.0)
    check("coding, hands moving in the last minute: not asked", not g.asked())
    g = rig(focus=True)
    seed(g)
    g.say("I'm focusing for an hour")
    g.window("msedge.exe", "Docs")
    for i in range(10):
        g.step(20, 25.0 + i * 20)
    check("focus mode: never", not g.asked())
    g = rig()
    seed(g)
    g.window("msedge.exe", "Docs")
    for _ in range(10):
        g.step(20, 2.0)
    check("typing all the while: never", not g.asked())
    g = rig()
    seed(g)
    g.window("msedge.exe", "Docs")
    g.step(20, 900.0)
    for i in range(6):
        g.step(30, 930.0 + i * 30)
    check("away: never", not g.asked())
    g.step(3, 1.0)
    for _ in range(4):
        g.step(20, 1.0)
    check("...and not the moment they come back typing", not g.asked())
    g = rig(free_after_s=0.0)
    seed(g)
    g.window("msedge.exe", "A.S.H body blueprints")
    for i in range(6):
        g.step(20, 25.0 + i * 20)
    check("with free_after_s off, the old rule stands (needs a stretch of work first)", not g.asked())


def v4_budget():
    print("\n=== V4. Not every moment is used ===")
    h = rig(moments_cfg={"jokes": False, "mood_check": False, "min_gap_s": 300, "max_per_hour": 2})
    seed(h)
    for name in ("Priya", "Sam", "Lee", "Noor"):
        h.say(f"I told {name} something about work")
    h.window("msedge.exe", "Docs")
    for i in range(60):
        h.step(30, 30.0 + (i % 5) * 20)
    asked = [o[0] for o in h.out if (o[1].get("attention") or {}).get("kind") == "wonder"]
    check("an hour of being free is not an hour of questions", 1 <= len(asked) <= 3, str(asked))
    h2 = rig()
    seed(h2)
    h2.window("msedge.exe", "Docs")
    h2.step(40, 40.0)
    h2.step(40, 80.0)
    first = len(h2.asked())
    h2.say("not now")
    for i in range(10):
        h2.step(30, 40.0 + i * 5)
    check("'not now' is answered and it does not ask again straight away", first == 1 and len(h2.asked()) == 1, str(h2.out[-3:]))


if __name__ == "__main__":
    for fn in (v1_judge, v2_score, v3_runtime, v4_budget):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
