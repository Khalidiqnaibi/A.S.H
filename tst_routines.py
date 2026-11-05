"""
tst_routines.py -- noticing a routine, and asking whether to keep it.

  R1  Watching a sitting: the first apps opened, in order, de-duplicated;
      shell and system windows skipped; a new sitting after being away or on
      a new day; nothing after the opening window
  R2  Noticing: the same opening on enough different days; a shorter opening
      of a longer one counts; once asked, not again
  R3  Asking and answering: yes, yes with a name, "call it ...", no
  R4  Persistence and the switches
  R5  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tst_routines.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.routines import Routines, RoutinesConfig  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check, rig, kinds = tw.check, tw.rig, tw.kinds
DAY, HOUR = tw.DAY, tw.HOUR
FREE = dict(idle_s=60.0, free_moment=True)
MORNING = time.mktime((2026, 3, 10, 8, 30, 0, 0, 0, -1))


def win(process, **meta):
    return SensorEvent(source="active_window", modality=Modality.TEXT, text="",
                       meta={"process": process, **meta})


def sitting(r, start, apps, gap=60):
    t = start
    for a in apps:
        r.observe(win(a), t)
        t += gap
    r.tick(start + r.cfg.window_s + 1)
    return t


def rrig(**cfg):
    w = rig()
    r = Routines(RoutinesConfig.from_dict(cfg), wonder=w, clock=lambda: w.t[0])
    return r, w


def r1_watching():
    print("\n=== R1. Watching a sitting ===")
    r, _ = rrig(min_days=99)
    sitting(r, MORNING, ["outlook.exe", "teams.exe", "outlook.exe", "jira.exe"])
    check("the opening, in order, de-duplicated", "Outlook > Teams > Jira" in r.seen, str(list(r.seen)))
    check("its shorter openings too", "Outlook > Teams" in r.seen)
    r, _ = rrig(min_days=99)
    sitting(r, MORNING, ["explorer.exe", "outlook.exe", "SearchHost.exe", "teams.exe", "python.exe"])
    check("shell and system windows are not steps", list(r.seen) == ["Outlook > Teams"], str(list(r.seen)))
    r, _ = rrig(min_days=99)
    r.observe(win("outlook.exe"), MORNING)
    r.observe(win("teams.exe"), MORNING + 700)
    r.tick(MORNING + 1300)
    check("after the opening window, nothing more is a step", r.seen == {}, str(r.seen))
    r, _ = rrig(min_days=99)
    r.observe(win("outlook.exe", blocked=True), MORNING)
    r.observe(win("teams.exe"), MORNING + 30)
    r.observe(win("jira.exe"), MORNING + 60)
    r.tick(MORNING + 700)
    check("a withheld window is not a step", list(r.seen) == ["Teams > Jira"], str(list(r.seen)))
    r, _ = rrig(min_days=99)
    sitting(r, MORNING, ["outlook.exe", "teams.exe"])
    r.observe(win("slack.exe"), MORNING + 900)
    r.observe(win("zoom.exe"), MORNING + 950)
    r.tick(MORNING + 2000)
    check("without being away, it is the same sitting", r.stats["sittings"] == 1)
    r.observe(win("slack.exe"), MORNING + 950 + 1801)
    r.observe(win("zoom.exe"), MORNING + 950 + 1850)
    r.tick(MORNING + 950 + 1801 + 700)
    check("after being away, a new one", r.stats["sittings"] == 2 and "Slack > Zoom" in r.seen, str(list(r.seen)))
    r, _ = rrig(min_days=99)
    r.observe(win("notepad.exe"), MORNING)
    r.tick(MORNING + 700)
    check("one app is not a routine", r.seen == {} and r.stats["recorded"] == 0)
    r.observe(win("a.exe"), MORNING + 1)
    check("a non-window event is ignored", r.observe(SensorEvent(source="microphone", text="hi"), MORNING) is None)


def r2_noticing():
    print("\n=== R2. Noticing ===")
    r, w = rrig()
    for d in range(2):
        sitting(r, MORNING + d * DAY, ["outlook.exe", "teams.exe", "jira.exe"])
    check("two days: not yet", kinds(w) == [])
    sitting(r, MORNING + 2 * DAY, ["outlook.exe", "teams.exe", "jira.exe"])
    check("three days: asked about", kinds(w) == ["routine"])
    q = w.wonders[next(iter(w.wonders))]
    check("the longest opening is the one asked", q.data["steps"] == ["Outlook", "Teams", "Jira"])
    check("worded with the time of day", q.question == "You often start your morning by opening Outlook, "
                                                         "then Teams, then Jira. Want me to remember that as a routine?",
          q.question)
    sitting(r, MORNING + 3 * DAY, ["outlook.exe", "teams.exe", "jira.exe"])
    check("not raised twice", len(w.wonders) == 1)
    w.housekeeping(MORNING + 30 * DAY)
    check("(the unasked question expired)", kinds(w) == [])
    sitting(r, MORNING + 31 * DAY, ["outlook.exe", "teams.exe", "jira.exe"])
    check("and a routine is offered once only, even after that", kinds(w) == [])

    r, w = rrig()
    sitting(r, MORNING, ["outlook.exe", "teams.exe", "jira.exe"])
    sitting(r, MORNING + DAY, ["outlook.exe", "teams.exe"])
    sitting(r, MORNING + 2 * DAY, ["outlook.exe", "teams.exe", "slack.exe"])
    check("a shared start counts across days", kinds(w) == ["routine"]
          and w.wonders[next(iter(w.wonders))].data["steps"] == ["Outlook", "Teams"])

    r, w = rrig()
    for d in range(3):
        sitting(r, MORNING + d * DAY + 7 * HOUR, ["code.exe", "chrome.exe"])
    check("an afternoon one says 'a session'", "start a session" in w.wonders[next(iter(w.wonders))].question)

    r, w = rrig()
    sitting(r, MORNING, ["outlook.exe", "teams.exe"])
    sitting(r, MORNING + 3 * HOUR, ["outlook.exe", "teams.exe"])
    sitting(r, MORNING + 6 * HOUR, ["outlook.exe", "teams.exe"])
    check("the same day three times is one day", kinds(w) == [])


def asked(r, w):
    for d in range(3):
        sitting(r, MORNING + d * DAY, ["outlook.exe", "teams.exe", "jira.exe"])
    w.t[0] = MORNING + 3 * DAY
    q = w.next_question(Environment(**FREE), w.t[0])
    w.mark_asked(q.id, w.t[0])
    return q


def r3_answers():
    print("\n=== R3. Asking and answering ===")
    r, w = rrig()
    asked(r, w)
    rep = w.hear("yes", w.t[0] + 5)
    check("yes keeps it under a default name", w.kb.procedure("usual start")["steps"] == ["Outlook", "Teams", "Jira"],
          str(w.kb.procedures))
    check("and says what it kept", rep.say == 'Got it -- I\'ll remember "usual start": Outlook, then Teams, then Jira.',
          rep.say)
    r, w = rrig()
    asked(r, w)
    w.hear("yes, morning check", w.t[0] + 5)
    check("yes with a name", w.kb.procedure("morning check") is not None, str(w.kb.procedures))
    r, w = rrig()
    asked(r, w)
    w.hear("sure, call it my standup prep", w.t[0] + 5)
    check("'call it ...'", w.kb.procedure("standup prep") is not None, str(w.kb.procedures))
    r, w = rrig()
    asked(r, w)
    rep = w.hear("no", w.t[0] + 5)
    check("no: nothing kept, and it is closed", w.kb.procedures == {} and rep.outcome == "answered"
          and kinds(w) == [], str(rep))
    r, w = rrig()
    asked(r, w)
    rep = w.hear("hmm", w.t[0] + 5)
    check("unclear: one follow-up", rep.outcome == "followup" and "yes or no" in rep.say)
    r, w = rrig()
    asked(r, w)
    w.t[0] += 2 * HOUR
    w.hear("sure, outlook then teams then jira again today", w.t[0])
    check("a later mention is not an answer", w.kb.procedures == {})


def r4_persist():
    print("\n=== R4. Persistence and switches ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "r.json")
    w = rig()
    r = Routines(RoutinesConfig(), path=p, wonder=w, clock=lambda: w.t[0])
    for i in range(2):
        sitting(r, MORNING + i * DAY, ["outlook.exe", "teams.exe"])
    r2 = Routines(RoutinesConfig(), path=p, wonder=w, clock=lambda: w.t[0])
    sitting(r2, MORNING + 2 * DAY, ["outlook.exe", "teams.exe"])
    check("days survive a restart", kinds(w) == ["routine"])
    text = open(p, encoding="utf-8").read()
    check("only app names are kept", "Outlook" in text and "title" not in text)
    with open(p, "w") as fh:
        fh.write("{x")
    check("a broken file is set aside", Routines(RoutinesConfig(), path=p).seen == {} and os.path.exists(p + ".bad"))
    r, w = rrig(enabled=False)
    sitting(r, MORNING, ["outlook.exe", "teams.exe"])
    check("off: nothing watched", r.seen == {})
    r, w = rrig(ask=False)
    for i in range(3):
        sitting(r, MORNING + i * DAY, ["outlook.exe", "teams.exe"])
    check("ask off: noticed but not asked", len(r.seen) == 1 and kinds(w) == [])
    r, w = rrig()
    w.cfg.kinds["routine"] = False
    for i in range(3):
        sitting(r, MORNING + i * DAY, ["outlook.exe", "teams.exe"])
    check("the wonder kind can be switched off", kinds(w) == [])
    r.forget()
    check("forget", r.seen == {})


def r5_runtime():
    print("\n=== R5. The runtime ===")
    clock = [MORNING]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=True, wonder_cfg={"min_gap_s": 0}, routines=True,
                             routines_path=os.path.join(d, "r.json"),
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    for day in range(3):
        clock[0] = MORNING + day * DAY
        for app in ("outlook.exe", "teams.exe", "jira.exe"):
            rt.submit(win(app))
            step(30, 0.0, ticks=1)
        step(700, 0.0, ticks=1)
    check("the runtime noticed it", rt.wonder is not None and "routine" in kinds(rt.wonder), str(kinds(rt.wonder)))
    for _ in range(15):
        step(10, 0.0, ticks=1)
    step(15, 25.0)
    said = [o[0] for o in out]
    check("and asks at a free moment", any("Want me to remember that as a routine?" in s for s in said), str(said))
    step(5, 5.0)
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="yes, morning check",
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("the answer keeps it", rt.wonder.kb.procedure("morning check") is not None, str(rt.wonder.kb.procedures))
    check("status carries routines", rt.status()["routines"]["openings"] >= 1)


if __name__ == "__main__":
    for fn in (r1_watching, r2_noticing, r3_answers, r4_persist, r5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
