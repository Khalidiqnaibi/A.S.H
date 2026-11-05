"""
tst_tasks.py -- multi-step tasks, with a plan to approve.

  K1  Splitting: commands joined by commas, "and", "then"; a part that is not
      a command goes back where it was; questions and chat are not plans
  K2  Named tasks: taught, listed, forgotten, kept across a restart
  K3  Approving: yes runs, no declines, anything else drops the plan and is
      heard as itself, and a plan not answered in a minute is gone
  K4  The runtime: the plan is said and nothing runs; on yes each step goes
      to the part that does it (apps, timers, the closing question answered
      by the approval), one line said back; a step it can't do is named;
      "undo that" still works on a timer it set; the console shows the plan
      as waiting

Run:  PYTHONIOENCODING=utf-8 python tst_tasks.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.tasks import Tasks, describe, split_steps  # noqa: E402

check = tw.check
T0 = tw.T0


def k1_split():
    print("\n=== K1. Splitting ===")
    cases = {
        "open Outlook, set a timer for 25 minutes and turn off the desk lamp":
            ["open Outlook", "set a timer for 25 minutes", "turn off the desk lamp"],
        "open spotify then play music": ["open spotify", "play music"],
        "ash, please open word and close excel": ["open word", "close excel"],
        "turn on the lights, and then set a timer for 5 minutes.": ["turn on the lights", "set a timer for 5 minutes"],
        "tell me when the download finishes and open chrome": ["tell me when the download finishes", "open chrome"],
        "open chrome; mute the sound": ["open chrome", "mute the sound"],
        "remind me to buy milk and eggs, then open notepad": ["remind me to buy milk and eggs", "open notepad"],
    }
    for text, want in cases.items():
        got = split_steps(text)
        check(f"{text[:40]!r}", got == want, str(got))
    for text in ("remind me to buy milk and eggs", "open chrome", "what is up and how are you", "open the door and the window",
                 "open chrome and close word?", "I opened chrome and closed word", "", None,
                 "open a, " * 20 + "open b", "x" * 300):
        check(f"not a plan: {str(text)[:30]!r}", split_steps(text) is None)
    check("described in order", describe(["a", "b"]) == "1) a, 2) b")


def k2_named():
    print("\n=== K2. Named tasks ===")
    p = os.path.join(tempfile.mkdtemp(), "tasks.json")
    t = [T0]
    k = Tasks(path=p, clock=lambda: t[0])
    r = k.hear("when I say get me ready for the meeting, open the agenda, mute the sound and set a 25 minute timer")
    check("taught", "get me ready for the meeting" in r["say"] and "3) set a 25 minute timer" in r["say"], r["say"])
    r = k.hear("whenever I say \"start work\" open outlook")
    check("one step is fine for a named task", k.saved["start work"]["steps"] == ["open outlook"], str(k.saved))
    r = k.hear("when I say hello there, tell me a story about dragons")
    check("not steps it can do: said so, nothing kept", "only keep steps" in r["say"] and "hello there" not in k.saved)
    r = k.hear("Get me ready for the meeting.")
    check("saying the name shows the plan", r.get("asking") and "1) open the agenda" in r["say"], str(r))
    k.pending = None
    r = k.hear("ash, get me ready for the meeting")
    check("with ASH's name in front too", r.get("asking"))
    k.pending = None
    check("listed", "start work" in k.hear("what tasks do you know?")["say"])
    k2 = Tasks(path=p)
    check("kept across a restart", sorted(k2.saved) == ["get me ready for the meeting", "start work"])
    r = k.hear("forget the start work task")
    check("forgotten", "start work" not in k.saved and "Forgotten" in r["say"])
    check("and gone after a restart", "start work" not in Tasks(path=p).saved)
    check("'forget Omar' is not ours", k.hear("forget Omar") is None)
    k3 = Tasks(clock=lambda: t[0])
    check("nothing taught: said how", "haven't taught" in k3.hear("list my tasks")["say"])
    open(p, "w").write("{bad")
    check("a broken file: none, no crash", Tasks(path=p).saved == {})
    check("off: hears nothing", Tasks(enabled=False).hear("open chrome and close word") is None)


def k3_approve():
    print("\n=== K3. Approving ===")
    t = [T0]
    k = Tasks(clock=lambda: t[0])
    r = k.hear("open chrome and close word", t[0])
    check("a plan, asked", r.get("asking") and k.awaiting(t[0]))
    t[0] += 5
    r = k.hear("yes please", t[0])
    check("yes: run", r == {"run": ["open chrome", "close word"], "name": None}, str(r))
    check("and no longer waiting", k.awaiting(t[0]) is None)
    k.hear("open chrome and close word", t[0])
    r = k.hear("no", t[0])
    check("no: not run", r == {"say": "Okay, I won't."} and k.stats["declined"] == 1)
    k.hear("open chrome and close word", t[0])
    r = k.hear("what time is it", t[0])
    check("anything else: the plan is dropped, the line is not ours", r is None and k.awaiting(t[0]) is None)
    k.hear("open chrome and close word", t[0])
    t[0] += 61
    check("a minute later it is gone", k.awaiting(t[0]) is None and k.hear("yes", t[0]) is None)
    k.hear("when I say wrap up, close word and turn off the lamp", t[0])
    k.hear("wrap up", t[0])
    k.hear("yes", t[0])
    check("a named task's runs are counted", k.saved["wrap up"]["runs"] == 1)
    check("summary: what was said, then what wasn't done",
          Tasks.summary([{"step": "a", "said": "Opening Chrome."}, {"step": "mute slack", "said": ""},
                         {"step": "c", "said": "Okay, a 5-minute timer, starting now."}])
          == "Opening Chrome. Okay, a 5-minute timer, starting now. I couldn't do: mute slack.")
    check("summary of nothing", Tasks.summary([]) == "Nothing came of that.")


def k4_runtime():
    print("\n=== K4. The runtime ===")
    import tst_desktop as td
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    d = tempfile.mkdtemp()
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, desktop=True, reminders=True,
                             reminders_path=os.path.join(d, "r.json"), tasks=True,
                             tasks_path=os.path.join(d, "t.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    ex = td.FakeEx()
    rt.desktop.ex = ex

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("open chrome, close word and set a 25 minute timer")
    check("the plan is said", out[-1] == "Here's the plan: 1) open chrome, 2) close word, 3) set a 25 minute timer. "
                                         "Go ahead?", str(out[-1:]))
    from src.senses import address
    check("asked as a question, so a bare 'yes' is heard as the answer", address.conversation._asked)
    check("nothing has run yet", ex.opened == [] and ex.closed == [] and not rt.reminders.items)
    from src.ui import settings as st
    w = st.waiting(rt)
    check("the console shows it waiting", w and w["kind"] == "plan" and w["replies"] == ["Yes", "No"], str(w))
    n = len(out)
    say("yes")
    check("all three done", ex.opened == ["chrome"] and ex.closed == ["winword.exe"]
          and any(r.kind == "timer" for r in rt.reminders.items.values()), f"{ex.opened} {ex.closed}")
    check("the closing question answered by the approval", rt.desktop._pending is None)
    check("one line said back", len(out) == n + 1 and "opening chrome" in out[-1].lower() and "closing word" in out[-1].lower()
          and "25-minute timer" in out[-1], str(out[n:]))
    check("no model asked", ash.calls == [])
    say("undo that")
    check("'undo that' still takes back the timer", all(r.status == "cancelled" for r in rt.reminders.items.values()),
          str(out[-1:]))
    say("open chrome and mute slack")
    say("yes")
    check("a step it can't do is named", out[-1].endswith("I couldn't do: mute slack."), str(out[-1:]))
    say("when I say start work, open chrome and set a timer for 50 minutes")
    check("taught through the runtime", "start work" in out[-1])
    say("start work")
    check("said back as a plan", out[-1].startswith("Here's the plan: 1) open chrome"), str(out[-1:]))
    say("no thanks")
    check("declined: nothing run", out[-1] == "Okay, I won't." and ex.opened.count("chrome") == 2)
    say("open chrome")
    check("a single command is not a plan", out[-1] == "Opening chrome." and ex.opened.count("chrome") == 3,
          str(out[-1:]))
    check("status", rt.status()["tasks"]["saved"] == ["start work"])
    check("kept on disk", "start work" in json.load(open(os.path.join(d, "t.json")))["tasks"])


if __name__ == "__main__":
    for fn in (k1_split, k2_named, k3_approve, k4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
