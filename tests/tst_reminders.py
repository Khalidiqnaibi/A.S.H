"""
tst_reminders.py -- reminders, timers and alarms that actually go off.

  R1  Setting: reminders (time before or after the words), timers, alarms;
      what is said back; what it is not
  R2  Going off: on time, once, in order; late after being off
  R3  Snooze, cancel, list
  R4  Persistence, and the briefing part
  R5  The runtime: set and said without the model; said in quiet hours; held
      while muted

Run:  PYTHONIOENCODING=utf-8 python tests/tst_reminders.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.reminders import Reminders  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))      # Tuesday 14:00
MIN = 60.0


def at(day, h, m=0):
    return time.mktime((2026, 3, 10 + day, h, m, 0, 0, 0, -1))


def rig(path=None):
    t = [T0]
    r = Reminders(path=path, clock=lambda: t[0])
    r.t = t
    return r


def due(r):
    return sorted(x.due for x in r.items.values() if x.status == "set")


def r1_setting():
    print("\n=== R1. Setting ===")
    r = rig()
    s = r.hear("remind me to call mom in 20 minutes")
    check("a reminder, time after", s == "Okay, I'll remind you to call mom at 14:20." and due(r) == [T0 + 20 * MIN], s)
    r = rig()
    s = r.hear("remind me in 2 hours to check the oven")
    check("time before", s == "Okay, I'll remind you to check the oven at 16:00.", s)
    r = rig()
    s = r.hear("remind me tomorrow at 9am to send the report")
    check("tomorrow at 9", s == "Okay, I'll remind you to send the report at Wed 09:00." and due(r) == [at(1, 9)], s)
    r = rig()
    s = r.hear("hey ash, remind me to stretch at 5pm")
    check("at 5pm today", due(r) == [at(0, 17)] and "stretch" in s, s)
    r = rig()
    s = r.hear("set a timer for 5 minutes")
    check("a timer", s == "Okay, a 5-minute timer, starting now." and due(r) == [T0 + 5 * MIN], s)
    r = rig()
    check("half an hour", r.hear("timer for half an hour") == "Okay, a 30-minute timer, starting now.")
    r = rig()
    check("an hour and a half", r.hear("set a timer for an hour and a half") ==
          "Okay, a 1-hour 30-minute timer, starting now.")
    r = rig()
    s = r.hear("wake me up at 7")
    check("an alarm at 7 is tomorrow morning", s == "Okay, I'll wake you at 07:00 tomorrow." and due(r) == [at(1, 7)], s)
    r = rig()
    s = r.hear("set an alarm for 6:30pm")
    check("an alarm later today", due(r) == [at(0, 18, 30)] and s.endswith("today."), s)
    r = rig()
    check("no time: asked for one", r.hear("remind me to call mom").startswith("When should I remind you?")
          and r.items == {})
    check("no subject: asked for one", r.hear("remind me in 10 minutes") == "What should I remind you about?")
    check("a timer with no length", r.hear("set a timer for later").startswith("How long"))
    check("a timer for a day, not a length", r.hear("set a timer for tomorrow").startswith("How long") and r.items == {})
    for text in ("what's the time", "I set a timer earlier", "the alarm went off", "", None, "x" * 200):
        check(f"not a request: {str(text)[:25]!r}", r.hear(text) is None)
    check("off: nothing", Reminders(enabled=False).hear("set a timer for 5 minutes") is None)


def r2_firing():
    print("\n=== R2. Going off ===")
    r = rig()
    r.hear("remind me to call mom in 20 minutes")
    r.hear("set a timer for 5 minutes")
    r.t[0] += 4 * MIN
    check("not before", r.tick() == [])
    r.t[0] += 2 * MIN
    check("the timer, on time", r.tick() == ["Your 5-minute timer is done."])
    check("once", r.tick() == [])
    r.t[0] = T0 + 20 * MIN + 1
    check("then the reminder", r.tick() == ["Reminder: call mom."])
    r = rig()
    r.hear("wake me up at 7")
    r.t[0] = at(1, 7)
    check("the alarm", r.tick() == ["Good morning -- it's 07:00."])
    r = rig()
    r.hear("remind me to stretch in 10 minutes")
    r.t[0] += 3 * 3600
    out = r.tick()
    check("late after being off: said, and said to be late",
          out == ["I was off when this was due at 14:10. Reminder: stretch."] and r.stats["late"] == 1, str(out))
    r = rig()
    r.hear("set a timer for 10 minutes")
    r.hear("set a timer for 5 minutes")
    r.t[0] += 10 * MIN + 1
    check("two due: in order, the overdue one said to be late",
          r.tick() == ["I was off when this was due at 14:05. Your 5-minute timer is done.",
                       "Your 10-minute timer is done."])


def r3_controls():
    print("\n=== R3. Snooze, cancel, list ===")
    r = rig()
    r.hear("set a timer for 5 minutes")
    r.t[0] += 5 * MIN
    r.tick()
    check("snooze", r.hear("snooze") == "Okay, again in 5 minutes.")
    r.t[0] += 5 * MIN
    check("and it comes back", r.tick() == ["Your 5-minute timer is done."])
    check("snooze for a set time", r.hear("snooze for 10 minutes") == "Okay, again in 10 minutes."
          and due(r) == [r.t[0] + 10 * MIN])
    r.t[0] += 10 * MIN
    r.tick()
    r.t[0] += 10 * MIN
    check("too late to snooze", r.hear("snooze") is None)
    r = rig()
    check("snooze with nothing gone off", r.hear("snooze") is None)
    r.hear("set a timer for 5 minutes")
    r.hear("remind me to call mom in 20 minutes")
    r.hear("remind me to stretch in 30 minutes")
    s = r.hear("what are my reminders?")
    check("listed, by time", s == "You have a 5-minute timer ending at 14:05, call mom at 14:20 and stretch at 14:30.", s)
    check("cancel the timer", r.hear("cancel the timer") == "Cancelled your timer." and len(due(r)) == 2)
    check("cancel all reminders", r.hear("cancel all my reminders") == "Cancelled your 2 reminders." and due(r) == [])
    check("nothing left to cancel", r.hear("cancel the alarm") == "You don't have any alarms set.")
    check("nothing to list", r.hear("do I have any reminders") == "You don't have any reminders, timers or alarms set.")
    r.t[0] += 3600
    check("cancelled ones never go off", r.tick() == [])
    r = rig()
    r.hear("remind me to call mom in 20 minutes")
    r.hear("remind me to stretch in 30 minutes")
    check("cancel one by what it says", r.hear("cancel my reminder to call mom") == "Cancelled your reminder."
          and len(due(r)) == 1)
    check("a reminder that isn't there", r.hear("cancel the reminder about the dentist") == "I don't have a reminder like that."
          and len(due(r)) == 1)
    print("\n=== R3b. Times said other ways ===")
    r = rig()
    r.hear("set a timer for 5 minutes")
    r.t[0] += 5 * MIN
    r.tick()
    check("'1 minute', not '1 minutes'", r.hear("snooze for 1 minute") == "Okay, again in 1 minute.")
    # The brain's set_reminder tool sets it for real (it used to return a due time that nothing scheduled).
    from tools import extra_tools
    r = rig()
    try:
        extra_tools.SCHEDULER = r.schedule
        out = extra_tools.reminder_tool("don't let me forget to call dad in 20 minutes")
        check("the tool schedules it and says so", out.get("ok") and "remind you" in out.get("result", ""), str(out))
        check("it is in the list, and goes off", len(due(r)) == 1 and (r.t.__setitem__(0, time.time() + 21 * MIN) or True)
              and any("dad" in x for x in r.tick()))
        extra_tools.SCHEDULER = False
        out = extra_tools.reminder_tool("remind me to call dad in 20 minutes")
        check("with no reminders to set it, the tool says it didn't", out.get("ok") is False, str(out))
    finally:
        extra_tools.SCHEDULER = None
    r = rig()
    s = r.hear("wake me up tomorrow at 7")
    s = lead(s)
    check("an alarm 'tomorrow at 7' is seven in the morning, not seven at night", s == "i'll wake you at 07:00 tomorrow.", s)
    s = r.hear("set an alarm for 6 tomorrow")
    s = lead(s)
    check("'for 6 tomorrow' too", s == "i'll wake you at 06:00 tomorrow.", s)
    s = r.hear("remind me in 20 minutes to plan tomorrow")
    s = lead(s)
    check("'in 20 minutes' wins over a day word in what it is about", s == "i'll remind you to plan tomorrow at 14:20.", s)
    s = r.hear("remind me in 10 minutes to put sun cream on")
    s = lead(s)
    check("'sun' is not Sunday", s == "i'll remind you to put sun cream on at 14:10.", s)
    s = r.hear("set a 30 second timer")
    s = lead(s)
    check("a timer in seconds", s == "a 30-second timer, starting now.", s)
    s = r.hear("a twenty minute timer")
    s = lead(s)
    check("a timer with the number in words", s == "a 20-minute timer, starting now.", s)
    s = r.hear("remind me in fifteen minutes to call one friend")
    s = lead(s)
    check("number words in the time, not in what it is about", s == "i'll remind you to call one friend at 14:15.", s)
    s = r.hear("timer for 1 hour 30 minutes")
    s = lead(s)
    check("an hour and minutes is both", s == "a 1-hour 30-minute timer, starting now.", s)


def lead(s):
    """What was said, without the way it opened ('Okay,' / 'Done --' / 'Got it --')."""
    import re
    s = re.sub(r"^[A-Za-z ']+(?:, | -- )", "", s)
    return s[:1].lower() + s[1:]


def r4_persist():
    print("\n=== R4. Persistence and the briefing ===")
    p = os.path.join(tempfile.mkdtemp(), "r.json")
    r = rig(p)
    r.hear("remind me tomorrow at 9am to send the report")
    r.hear("set a timer for 5 minutes")
    r2 = Reminders(path=p, clock=lambda: T0)
    check("a restart keeps them", len(r2.items) == 2)
    r2.t = [T0 + 6 * MIN]
    r2.clock = lambda: r2.t[0]
    r2.tick()
    r3 = Reminders(path=p, clock=lambda: T0)
    check("one that went off is gone after a restart", len(r3.items) == 1)
    check("tomorrow's reminders in the evening briefing",
          r3.briefing_part("evening", at(0, 20)) == "Reminders tomorrow: 09:00 send the report.")
    check("today's in the morning one", r3.briefing_part("morning", at(1, 7)) == "Reminders today: 09:00 send the report.")
    check("nothing: no part", r3.briefing_part("morning", at(0, 7)) is None)
    with open(p, "w") as fh:
        fh.write("{x")
    check("a broken file is set aside", Reminders(path=p).items == {} and os.path.exists(p + ".bad"))


def r6_wording():
    print("\n=== R6. My is your; and asking what is set ===")
    r = rig()
    s = r.hear("remind me to call my mom in 20 minutes")
    check("my becomes your, in the answer", s == "Okay, I'll remind you to call your mom at 14:20.", s)
    check("...and in what is said when it goes off", r.tick(T0 + 21 * MIN) == ["Reminder: call your mom."], str(r.status()))
    r = rig()
    s = r.hear("remind me in an hour to tell Sam I'm late and that I've left my keys")
    check("I'm, I've, my", s == "Okay, I'll remind you to tell Sam you're late and that you've left your keys at 15:00.", s)
    r = rig()
    r.hear("remind me in 10 minutes to email the professor")
    for q in ("do I have reminders", "do I have any reminders", "what reminders do I have", "what reminders are set",
              "show my reminders", "check my reminders", "list my reminders"):
        check(f"{q!r} lists them", "email the professor" in (r.hear(q) or ""), str(r.hear(q)))
    check("none set says so", "any reminders" in rig().hear("do I have reminders"))
    for q in ("do I have time", "what is a reminder", "remind"):
        check(f"{q!r} is not a list request", "email the professor" not in (r.hear(q) or ""), str(r.hear(q)))


def r7_snooze_words():
    print("\n=== R7. Asking for it again, said naturally ===")
    for said, mins in [("again in 10 minutes", 10), ("remind me again in 15 minutes", 15), ("give me five more minutes", 5),
                       ("10 more minutes", 10), ("just 3 minutes", 3), ("later", 5), ("not now", 5), ("remind me later", 5),
                       ("okay, ten minutes", 10), ("no, again in 20 min", 20)]:
        r = rig()
        r.hear("remind me to call mom in 5 minutes")
        r.t[0] += 5 * MIN
        r.tick()
        s = r.hear(said)
        check(f"{said!r} snoozes {mins} minutes", s == f"Okay, again in {mins} minutes." and due(r) == [r.t[0] + mins * MIN], str(s))
    r = rig()
    check("with nothing gone off, 'later' and 'ten minutes' are nobody's business", r.hear("later") is None and r.hear("ten minutes") is None
          and r.hear("not now") is None)
    r = rig()
    r.hear("remind me to call mom in 5 minutes")
    r.t[0] += 5 * MIN
    r.tick()
    r.hear("snooze")
    check("once snoozed, 'later' does not snooze it a second time", r.hear("later") is None)
    r = rig()
    s = r.hear("remind me again in 10 minutes to call mom")
    check("'remind me again in ...' with a thing to do is still a new reminder", s is not None and "call mom" in s, str(s))


def r5_runtime():
    print("\n=== R5. The runtime ===")
    clock = [T0]
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(quiet_hours=(13, 23)), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, reminders=True, briefings=True,
                             briefings_path=os.path.join(tempfile.mkdtemp(), "b.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.reminders.clock = lambda: clock[0]

    def say(text):
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("set a timer for 5 minutes")
    check("set without the model", [o[0] for o in out] == ["Okay, a 5-minute timer, starting now."] and ash.calls == [])
    clock[0] += 5 * MIN + 1
    rt._tick()
    check("said when due, even in quiet hours", out[-1][0] == "Your 5-minute timer is done."
          and out[-1][1]["pathway"] == "reminder", str(out))
    say("set a timer for 1 minutes")
    rt.gate.mute(True)
    clock[0] += 2 * MIN
    rt._tick()
    check("held while muted", out[-1][0] != "Your 1-minute timer is done.")
    rt.gate.mute(False)
    rt._tick()
    check("and said when unmuted", out[-1][0] == "Your 1-minute timer is done.")
    say("set a timer for 1 minutes")
    rt.gate.mute(True)
    clock[0] += 2 * MIN
    rt._tick()
    check("one that comes due while muted is still 'set' (so a restart does not lose it)",
          sorted(x.status for x in rt.reminders.items.values() if x.kind == "timer")[-1] == "set"
          and any(x.status == "set" for x in rt.reminders.items.values()))
    rt.gate.mute(False)
    rt._tick()
    check("...and is said once unmuted", out[-1][0] == "Your 1-minute timer is done.")
    say("snooze")
    check("snooze works through the loop", out[-1][0] == "Okay, again in 5 minutes.")
    check("status carries reminders", rt.status()["reminders"]["fired"] == 3)
    say("remind me tomorrow at 9am to send the report")
    check("reminders reach the evening briefing", "Reminders tomorrow: 09:00 send the report."
          in rt.briefings.compose("evening", clock[0]))


if __name__ == "__main__":
    for fn in (r1_setting, r2_firing, r3_controls, r4_persist, r5_runtime, r6_wording, r7_snooze_words):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
