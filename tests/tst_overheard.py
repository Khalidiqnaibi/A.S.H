"""
tst_overheard.py -- speech from the room does not run things.

ASH hears everything the microphone hears: the TV, a call, someone else in the room. Opening, closing and switching things
already only acted on a line meant for ASH. Found in review that the rest did not: overheard "open X then close Y" made a plan
and the next overheard "ok" ran it (closing the app, with the confirmation auto-given); TV saying "remind me to..." or "wake me
at 3" set reminders and alarms; "cancel my reminders" cancelled them; a guest's bare "yes" confirmed "Close Word?".

  O1  Not meant for ASH: no plan, no reminder, no watch, no arrival reminder
  O2  Meant for ASH: all of it still works
  O3  A guest: cannot plan, set reminders, cancel watches, or confirm what is pending
  O4  A confirmation is the user's, in the conversation the question opened

Run:  PYTHONIOENCODING=utf-8 python tests/tst_overheard.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_desktop as td  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
T0 = tw.T0


def make():
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=tw.Ash(), periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, desktop=True, reminders=True,
                             reminders_path=os.path.join(d, "r.json"), tasks=True, tasks_path=os.path.join(d, "t.json"),
                             watch=True, watch_path=os.path.join(d, "w.json"), arrivals=True,
                             arrivals_path=os.path.join(d, "a.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.desktop.ex = td.FakeEx()

    def hear(text, addressed=False, source="microphone", guest=False, likely=False):
        clock[0] += 3
        meta = {}
        if guest:
            meta["guest"] = True
        if likely:
            meta["address"] = {"likely": True}
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9,
                              urgency=0.6, meta=meta))
        rt._tick()
    rt.hear, rt.out, rt.t = hear, out, clock
    return rt


def o1_overheard():
    print("\n=== O1. Said in the room, not to ASH ===")
    rt = make()
    rt.hear("remind me to call mum at five")
    check("'remind me to...' from the room sets nothing", not rt.reminders.items, str(rt.reminders.items))
    rt.hear("wake me up at 3 am")
    check("an alarm either", not rt.reminders.items)
    rt.hear("set a timer for ten minutes")
    check("a timer either", not rt.reminders.items)
    rt.hear("open chrome, close word and set a 25 minute timer")
    check("a plan is not made from the room", rt.tasks.pending is None and not any("plan" in o for o in rt.out), str(rt.out))
    rt.hear("ok")
    rt.hear("sure")
    check("...so an overheard 'ok' has nothing to run", rt.desktop.ex.closed == [] and rt.desktop.ex.opened == [])
    rt.hear("when I say get ready, open the agenda and mute the sound")
    check("a macro is not taught from the room", not rt.tasks.saved, str(rt.tasks.saved))
    rt.hear("tell me when the download finishes")
    check("a watch is not set from the room", not rt.watcher.watches)
    rt.hear("remind me to buy milk when I get home")
    check("an arrival reminder is not set from the room", not (rt.arrivals and rt.arrivals.items))


def o2_meant():
    print("\n=== O2. Said to ASH ===")
    rt = make()
    rt.hear("remind me to call mum in 20 minutes", addressed=True)
    check("addressed: a reminder is set", len(rt.reminders.items) == 1, str(rt.out[-1:]))
    rt.hear("set a timer for 10 minutes", source="console")
    check("typed: a timer is set", len(rt.reminders.items) == 2)
    rt.hear("open chrome, close word and set a 25 minute timer", addressed=True)
    check("addressed: the plan is made", rt.tasks.pending is not None and "plan" in rt.out[-1], str(rt.out[-1:]))
    rt.hear("yes", addressed=True)
    check("...and a yes runs it", rt.desktop.ex.opened == ["chrome"], str(rt.desktop.ex.opened))
    rt.hear("remind me to stretch in 30 minutes", likely=True)
    check("a line the address model thinks is for ASH counts too", len(rt.reminders.items) >= 3)
    rt.hear("tell me when the download finishes", addressed=True)
    check("addressed: a watch is set", rt.watcher.watches)


def o3_guest():
    print("\n=== O3. A guest ===")
    rt = make()
    rt.hear("remind me to call mum at five", addressed=True, guest=True)
    check("a guest cannot set a reminder (and is told so)", not rt.reminders.items and "only do that for" in rt.out[-1], str(rt.out[-1:]))
    rt.hear("remind me to call mum in 20 minutes", addressed=True)
    n = len(rt.reminders.items)
    rt.hear("cancel my reminders", addressed=True, guest=True)
    check("a guest cannot cancel the user's reminders", len(rt.reminders.items) == n and "only do that for" in rt.out[-1],
          str(rt.out[-1:]))
    rt.hear("stop watching", addressed=True, guest=True)
    check("...nor watches", "only do that for" in rt.out[-1], str(rt.out[-1:]))
    rt.hear("open chrome and close word", addressed=True, guest=True)
    check("a guest cannot make a plan", rt.tasks.pending is None)


def o4_confirm():
    print("\n=== O4. A confirmation is the user's ===")
    rt = make()
    rt.hear("close word", addressed=True)
    check("'Close Word?' is asked", rt.desktop._pending is not None, str(rt.out[-1:]))
    rt.hear("yes", addressed=True, guest=True)
    check("a guest's 'yes' does not close it", rt.desktop.ex.closed == [], str(rt.desktop.ex.closed))
    check("...and the question is dropped, not left for the next voice", rt.desktop._pending is None)
    rt.hear("close word", addressed=True)
    rt.hear("yes", addressed=True)
    check("the user's 'yes' closes it", rt.desktop.ex.closed == ["winword.exe"], str(rt.desktop.ex.closed))

    rt = make()
    rt.hear("open chrome, close word and set a 25 minute timer", addressed=True)
    rt.hear("yes", addressed=True, guest=True)
    check("a guest's 'yes' does not run a plan", rt.desktop.ex.opened == [] and rt.desktop.ex.closed == [], str(rt.out[-2:]))
    rt = make()
    rt.hear("open chrome and set a timer for 5 minutes", addressed=True)
    rt.t[0] += 120
    rt.hear("yes", addressed=False)
    check("a plan that waited two minutes is gone, whoever says yes", rt.desktop.ex.opened == [])


if __name__ == "__main__":
    o1_overheard()
    o2_meant()
    o3_guest()
    o4_confirm()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
