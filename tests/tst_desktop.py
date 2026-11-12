"""
tst_desktop.py -- opening, switching to and closing apps, and running a routine.

  D1  Opening: known apps, extra ones from the config, and what is left to
      the brain ("open the readme", "start the tests")
  D2  Switching: only to something running
  D3  Closing: asked first; yes within a minute does it, no or anything else
      does not; a polite close; nothing closed that is not open
  D4  Routines kept through wondering, run by name
  D5  The runtime: said and done without the model; "hey ash" in front works
  D6  The real executor does not crash here (opening nothing)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_desktop.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.desktop import Desktop, DesktopConfig, Executor  # noqa: E402
from src.daemon.wondering import Knowledge  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0 = tw.T0


class FakeEx(Executor):
    def __init__(self):
        self.opened, self.closed, self.focused = [], [], []
        self.procs = {"chrome.exe", "winword.exe"}
        self.fail = False

    def open(self, target):
        if self.fail:
            return False
        self.opened.append(target)
        return True

    def running(self, process):
        return process in self.procs

    def close(self, process):
        self.closed.append(process)
        return True

    def focus(self, process):
        self.focused.append(process)
        return True


def drig(**cfg):
    t = [T0]
    ex = FakeEx()
    kb = Knowledge()
    d = Desktop(DesktopConfig.from_dict(cfg), executor=ex, knowledge=kb, clock=lambda: t[0])
    d.t, d.ex_, d.kb_ = t, ex, kb
    return d


def d1_open():
    print("\n=== D1. Opening ===")
    d = drig()
    check("open Outlook", d.hear("open Outlook") == "Opening Outlook." and d.ex_.opened == ["outlook"])
    check("launch spotify (a protocol)", d.hear("launch spotify please") == "Opening spotify." and
          d.ex_.opened[-1] == "spotify:")
    check("can you open VS Code", d.hear("can you open VS Code") == "Opening VS Code." and d.ex_.opened[-1] == "code")
    check("open the calculator app", d.hear("open the calculator app") == "Opening calculator.")
    for text in ("open the readme", "open readme.md", "start the tests", "run the build", "open my email",
                 "what is outlook", "", None, "open " + "x" * 100):
        check(f"left to the brain: {str(text)[:30]!r}", d.hear(text) is None)
    d = drig(apps={"jira": "https://jira.example.com", "blender": ["blender", "blender.exe"]})
    check("an app from the config", d.hear("open jira") == "Opening jira." and d.ex_.opened == ["https://jira.example.com"])
    check("with its process", d.apps["blender"] == ("blender", "blender.exe"))
    d.ex_.fail = True
    check("a failure is said", d.hear("open outlook") == "I couldn't open outlook.")
    check("off: nothing", drig(enabled=False).hear("open outlook") is None)


def d2_switch():
    print("\n=== D2. Switching ===")
    d = drig()
    check("to something running", d.hear("switch to chrome") == "Switching to chrome." and d.ex_.focused == ["chrome.exe"])
    check("'go back to word'", d.hear("go back to word") == "Switching to word.")
    check("not running: says so, opens nothing", d.hear("switch to outlook").startswith("outlook isn't open")
          and d.ex_.opened == [])
    check("an unknown name is left to the brain", d.hear("switch to the other tab") is None)


def d3_close():
    print("\n=== D3. Closing ===")
    d = drig()
    r = d.hear("close word")
    check("asked first", r == "Close word? Anything unsaved there may be lost. Say yes to close it." and d.ex_.closed == [])
    d.t[0] += 10
    check("yes closes it", d.hear("yes") == "Closing word." and d.ex_.closed == ["winword.exe"])
    check("and only once", d.hear("yes") is None and d.ex_.closed == ["winword.exe"])
    d = drig()
    d.hear("close word")
    check("no leaves it", d.hear("no") == "Okay, I'll leave it open." and d.ex_.closed == [])
    d = drig()
    d.hear("close word")
    check("something else cancels it", d.hear("what's the time") is None and d.ex_.closed == [])
    check("(and a later yes is not taken)", d.hear("yes") is None and d.ex_.closed == [])
    d = drig()
    d.hear("close word")
    d.t[0] += 120
    check("too late a yes is not taken", d.hear("yes") is None and d.ex_.closed == [])
    d = drig()
    check("not open: nothing to close", d.hear("close outlook") == "outlook isn't open.")
    d = drig()
    d.ex_.procs = {"olk.exe"}                          # the new Outlook for Windows
    check("the new Outlook (olk.exe) is Outlook too", d.hear("close outlook").startswith("Close outlook?"))
    d.hear("yes")
    check("and closing it closes olk.exe", d.ex_.closed == ["olk.exe"], str(d.ex_.closed))
    d = drig()
    d.ex_.procs = {"olk.exe"}
    check("switching to it finds olk.exe", d.hear("switch to outlook") == "Switching to outlook."
          and d.ex_.focused == ["olk.exe"])
    d = drig()
    d.ex_.procs = {"outlook.exe"}
    d.hear("close outlook")
    d.hear("yes")
    check("the classic one still works", d.ex_.closed == ["outlook.exe"])
    d = drig(confirm_close=False)
    check("confirm off: closes at once", d.hear("quit chrome") == "Closing chrome." and d.ex_.closed == ["chrome.exe"])


def d4_routines():
    print("\n=== D4. Routines ===")
    d = drig()
    d.kb_.store_procedure("morning check", ["Outlook", "Teams", "Chrome"])
    r = d.hear("run my morning check")
    check("a kept routine runs, in order", r == "Running morning check: opening Outlook, then Teams, then Chrome."
          and d.ex_.opened == ["outlook", "ms-teams:", "chrome"], str((r, d.ex_.opened)))
    d.hear("do my morning check routine")
    check("'... routine' works too", d.ex_.opened[-3:] == ["outlook", "ms-teams:", "chrome"])
    d.kb_.store_procedure("design work", ["Figma"])
    d.hear("start my design work")
    check("a step that is not a known app is started by its name", d.ex_.opened[-1] == "Figma")
    check("no such routine: left to the brain", d.hear("run my evening thing") is None)
    d.ex_.fail = True
    check("nothing opened: said", d.hear("run my morning check") == "I couldn't open anything for morning check.")


def rt_rig():
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, desktop=True, wonder=True,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    ex = FakeEx()
    rt.desktop.ex = ex

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.clock, h.ex = rt, ash, out, clock, ex

    def say(text, source="console", addressed=True):
        clock[0] += 2
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    h.say = say
    h.said = lambda: [o[0] for o in out]
    return h


def d5_runtime():
    print("\n=== D5. The runtime ===")
    h = rt_rig()
    h.say("hey ash, open outlook")
    check("done and said, without the model", h.ex.opened == ["outlook"] and h.said() == ["Opening outlook."]
          and h.ash.calls == [], str(h.said()))
    h.say("close word")
    h.say("yes", source="microphone", addressed=False)
    check("closing asks, and a yes heard without the name closes it", h.ex.closed == ["winword.exe"], str(h.said()))
    h.rt.wonder.kb.store_procedure("morning check", ["Outlook", "Teams"])
    h.say("run my morning check")
    check("a kept routine runs", h.said()[-1] == "Running morning check: opening Outlook, then Teams.")
    h.say("open the readme")
    check("anything else goes to the model", h.ash.calls == ["open the readme"])
    h.say("open outlook", source="screen", addressed=False)
    check("text on the screen does nothing", h.ex.opened.count("outlook") == 2)
    n = len(h.ex.opened)
    h.say("open outlook", source="microphone", addressed=False)
    check("speech overheard (not said to ASH) opens nothing", len(h.ex.opened) == n)
    h.say("close chrome", source="microphone", addressed=False)
    check("nor closes anything", h.ex.closed == ["winword.exe"] and h.rt.desktop._pending is None)
    h.clock[0] += 2
    h.rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="open outlook", addressed=False,
                            salience=0.9, urgency=0.6, meta={"address": {"likely": True}}))
    h.rt._tick()
    check("probably said to ASH (in a conversation) still works", len(h.ex.opened) == n + 1)
    h.say("open outlook", source="microphone", addressed=True)
    check("said to ASH by name works", len(h.ex.opened) == n + 2)
    check("status carries desktop", h.rt.status()["desktop"]["opened"] == 5)


def d6_executor():
    print("\n=== D6. The real executor ===")
    ex = Executor()
    try:
        r1 = ex.running("no-such-process-xyz.exe")
        r2 = ex.focus("no-such-process-xyz.exe")
        ok = r1 is False and r2 is False
    except Exception as e:
        ok = False
        print("   ", e)
    check("asking about a program that is not there is safe", ok)


if __name__ == "__main__":
    for fn in (d1_open, d2_switch, d3_close, d4_routines, d5_runtime, d6_executor):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
