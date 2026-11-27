"""tst_dial.py -- how often ASH volunteers ("be less chatty", "fewer jokes"), care for how the body feels ("I have a headache"), what matters in the
email, the memory notice shown first and said at a free moment, and the console's thinking timer. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_dial.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import dial  # noqa: E402
from src.daemon.inbox import Inbox, InboxConfig  # noqa: E402

check = tw.check


def rig():
    rt = tm.rig()
    rt.cfg.notices_path = os.path.join(tempfile.mkdtemp(), "notices.json")
    rt._restart_checked = True
    return rt


def main():
    print("\n=== T1. The dial: how often, never who ===")
    for t, want in (("be less chatty", ("chat", "quiet")), ("Ash, be chattier", ("chat", "chatty")), ("back to normal chattiness", ("chat", "normal")),
                    ("no more jokes", ("jokes", "none")), ("fewer jokes please", ("jokes", "fewer")), ("more jokes", ("jokes", "more")),
                    ("you can tell jokes again", ("jokes", "normal")), ("keep it short", None), ("tell me a joke", None)):
        check(f"{t!r} -> {want}", dial.parse(t) == want, str(dial.parse(t)))
    rt = rig()
    used, said = rt.say("be less chatty")
    check("quiet: the gate's hourly caps go down, said so, no model", not used and rt.gate.max_unsolicited_per_hour == 2 and rt.gate.max_offers_per_hour == 1
          and said[-1].startswith("Okay -- I'll bring things up less"), f"{used} {said}")
    rt.say("no more jokes")
    joke = rt.moments.activities.get("joke")
    check("no jokes: the joke activity is off", joke is not None and joke.weight == 0.0)
    rt.say("more jokes")
    check("more: back on, every three hours", joke.weight > 0 and joke.cooldown_s == 3 * 3600.0)
    rt2 = rig()
    rt2.cfg.notices_path = rt.cfg.notices_path
    rt2._apply_dial()
    check("kept across a restart", rt2.gate.max_unsolicited_per_hour == 2 and rt2.moments.activities["joke"].cooldown_s == 3 * 3600.0)

    print("\n=== T2. Care ===")
    line, mins = dial.care("I have a headache", 2 * 3600 + 600, 15)
    check("a headache: the time at the screen, water, an offer of quiet", line.startswith("You've been at the screen 2 hours 10 minutes without a real break.")
          and "Water" in line and mins == 15, line)
    line, mins = dial.care("my eyes hurt", 1800, 15)
    check("eyes: twenty-twenty-twenty", "twenty-twenty-twenty" in line and mins == 10, line)
    line, mins = dial.care("I'm exhausted", 600, 1)
    check("exhausted at 1 am: sleep", "It's late -- sleep is the fix" in line and mins == 30, line)
    line, mins = dial.care("I have chest pain", 0, 12)
    check("an emergency: the emergency number, nothing else", "emergency number" in line and mins == 0, line)
    check("not about the body", dial.care("I have a meeting", 0, 12) is None and dial.care("my code hurts", 0, 12) is None)
    rt = rig()
    used, said = rt.say("I have a headache")
    check("in the runtime: by rule", not used and said and "keep quiet for fifteen minutes" in said[-1], f"{used} {said}")
    used, said = rt.say("yes")
    check("a yes: quiet for fifteen minutes", rt.gate.muted and said[-1] == "Okay -- quiet for 15 minutes. I'll still answer if you ask."
          and abs(rt._quiet_until - rt.clock() - 900) < 30, f"{said} {rt._quiet_until}")

    print("\n=== T3. What matters in the email ===")
    ib = Inbox(InboxConfig.from_dict({"enabled": True, "host": "imap.x", "user": "u", "password": "p"}), background=False,
               fetch=lambda cfg: (3, [("Omar", "dinner?"), ("GitHub", "[A.S.H] new star"), ("University", "Exam timetable change")]))
    ib.who = lambda n: "your cousin" if n == "Omar" else ""
    ib.refresh(force=True)
    time.sleep(0.05)
    s = ib.hear("anything important in my email?")
    check("someone known, and an exam -- not the newsletter", s == "Worth a look: Omar, your cousin -- dinner?; University -- Exam timetable change.", str(s))
    ib2 = Inbox(InboxConfig.from_dict({"enabled": True, "host": "imap.x", "user": "u", "password": "p"}), background=False,
                fetch=lambda cfg: (1, [("Shop", "Weekly deals")]))
    ib2.refresh(force=True)
    time.sleep(0.05)
    check("nothing stands out: said so", (ib2.hear("any important emails") or "").startswith("Nothing that stands out"), str(ib2.hear("any important emails")))

    print("\n=== T4. The memory notice: a card and a glance first, said at a free moment ===")
    rt = rig()
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    rt.governor = types.SimpleNamespace(mem=95.0, cpu=20.0, status=lambda: {}, tick=lambda now: None)
    now = rt.clock()
    rt._mem_high_since = now - 1300
    rt._mem_told_at = -1e18
    rt._memory_notice(now)
    from src.daemon import headsup
    cards = headsup.cards(rt, now)
    check("a card under the face", any(c.get("title") == "Memory at 95%" for c in cards), str(cards))
    check("not spoken over whatever they were doing", not any(o.startswith("Memory has been above") for o in out), str(out))
    rt._tick()
    check("offered through the gate, so it is said when the moment allows", any(o.startswith("Memory has been above 90%") for o in out), str(out))

    print("\n=== T5. The console shows how long an answer is taking ===")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(here, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    check("the thinking row counts the seconds", "cues.thinking.started" in html and 'padStart(2, "0")' in html and "setInterval(() => { if (cues.thinking) renderThinking(); }, 1000);" in html)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
