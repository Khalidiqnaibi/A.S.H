"""
tst_asks.py -- what ASH asked, what the user said back, and what that teaches about timing.

  A1  an ask is kept open; a reply settles it as answered / yes / no / later / moved on; nothing asked: nothing to settle
  A2  "I have a question for you when you have a minute" stays standing for hours; "go ahead", "what questions do you have?"
      and a plain yes take it up; something unrelated leaves it standing
  A3  the ledger: one line per ask, a line per outcome; survives a restart; a guest's words are not kept
  A4  the pair goes into episodic memory (and not a guest's)
  A5  timing: a no lowers a situation, a yes raises it; finer situations count more once they have evidence; it persists
  A6  what ASH tells the model; the report in words

Run:  PYTHONIOENCODING=utf-8 python tests/tst_asks.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.asks import Asks, TimingModel, context_of, keys_of  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0                      # a Thursday morning, local time does not matter for these


def make(directory=None, episodic=None, **kw):
    t = [T0]
    a = Asks(directory or tempfile.mkdtemp(), clock=lambda: t[0], episodic=episodic, user="Immortal", **kw)
    a.t = t
    return a


class Epi:
    def __init__(self):
        self.rows = []

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.rows.append((summary, event_type, importance))


def a1_settling():
    print("\n=== A1. Settling an ask ===")
    a = make()
    check("nothing asked: a reply is nothing to settle", a.hear("yes", T0) is None and a.current(T0) is None)
    q = a.asked("wonder", "What's your father's name?", T0, expects="answer", topic="role")
    check("it is open, and current", a.current(T0) is q and q.status == "open")
    r = a.hear("Afif", T0 + 8)
    check("a short reply is the answer", r.outcome == "answered" and r.reply == "Afif" and q.status == "answered", str(r))
    check("and it is no longer open", a.current(T0 + 9) is None)
    for said, want in (("no", "declined"), ("not now", "later"), ("later", "later"), ("no thanks", "declined"),
                       ("what time is it?", "moved_on"), ("I don't know really", "answered")):
        a2 = make()
        a2.asked("wonder", "What is brain.py for?", T0)
        got = a2.hear(said, T0 + 5)
        check(f"{said!r} -> {want}", got is not None and got.outcome == want, str(got and got.outcome))
    a3 = make()
    a3.asked("moment", "How are you holding up today?", T0)
    check("not said to ASH (overheard): not an answer", a3.hear("yes", T0 + 3, addressed=False) is None and a3.current(T0 + 3) is not None)
    check("two minutes on nobody has answered: it is settled as ignored",
          a3.current(T0 + 400) is None and a3.stats["ignored"] == 1)
    yn = make()
    yn.asked("moment", "Want me to look into that?", T0, expects="yes_no")
    check("a yes to a yes/no is accepted", yn.hear("yes please", T0 + 4).outcome == "accepted")
    many = make(max_open=3)
    for i in range(5):
        many.asked("moment", f"q{i}?", T0 + i)
    check("only a few stay open; the oldest are settled as ignored", len(many.open) == 3 and many.stats["ignored"] == 2)
    check("the latest is the one a reply answers", many.hear("blue", T0 + 6).ask.text == "q4?")


def a2_standing():
    print("\n=== A2. 'I have a question for you when you have a minute' ===")
    a = make()
    q = a.asked("briefing", "Good morning. I have a question for you when you have a minute.", T0, expects="permission")
    check("still standing three hours later", a.standing(T0 + 3 * 3600) is q)
    for said in ("yes", "sure", "go ahead", "What questions do you have", "what's the question?", "ask me", "okay, shoot",
                 "what is it", "I'm listening", "now is fine", "what do you want to know"):
        b = make()
        b.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
        r = b.hear(said, T0 + 2 * 3600)
        check(f"{said!r} takes it up", r is not None and r.outcome == "accepted", str(r and r.outcome))
    b = make()
    b.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    r = b.hear("open spotify", T0 + 60)
    check("an unrelated request leaves it standing", r is None and b.standing(T0 + 61) is not None)
    check("'not now' puts it off", b.hear("not now", T0 + 90).outcome == "later" and b.standing(T0 + 91) is None)
    c = make()
    c.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    check("'no' declines it", c.hear("no", T0 + 5).outcome == "declined")
    d = make()
    d.asked("briefing", "I have a question for you when you have a minute.", T0, expects="permission")
    check("after a day it is gone", d.standing(T0 + 9 * 3600) is None)


def a3_ledger():
    print("\n=== A3. The ledger ===")
    d = tempfile.mkdtemp()
    a = make(d)
    ctx = context_of(T0, "msedge.exe", "A.S.H blueprints", idle_s=300, dwell_min=22)
    a.asked("wonder", "What's your father's name?", T0, topic="role", ctx=ctx)
    a.hear("Afif", T0 + 12)
    rows = [json.loads(x) for x in open(os.path.join(d, "asks.jsonl"), encoding="utf-8")]
    check("an 'asked' line and a 'settled' line", [r["event"] for r in rows] == ["asked", "settled"], str([r["event"] for r in rows]))
    check("the situation is in it: the program, how long, reading or typing",
          rows[0]["ctx"]["app"] == "Edge" and rows[0]["ctx"]["mode"] == "reading" and rows[0]["ctx"]["dwell"] == "a_while", str(rows[0]["ctx"]))
    check("the reply, and how long it took", rows[1]["reply"] == "Afif" and rows[1]["latency_s"] == 12.0 and rows[1]["status"] == "answered")
    b = Asks(d, clock=lambda: T0 + 100)
    check("after a restart it still knows what was asked", b.recent and b.recent[-1].text == "What's your father's name?" and b.recent[-1].reply == "Afif")
    n = b.asked("moment", "Another?", T0 + 100)
    check("ids carry on, not restart", n.id == 2, str(n.id))
    g = make()
    g.asked("moment", "How are you?", T0)
    g.hear("fine thanks", T0 + 4, guest=True)
    check("a guest's words are not kept, and teach nothing", g.recent[-1].reply == "" and g.timing.updates == 0)
    nr = make(keep_replies=False)
    nr.asked("moment", "How are you?", T0)
    nr.hear("fine", T0 + 4)
    check("with replies switched off only the outcome is kept", nr.recent[-1].reply == "" and nr.recent[-1].status == "answered")


def a4_episodic():
    print("\n=== A4. Episodic memory ===")
    e = Epi()
    a = make(episodic=e)
    ctx = context_of(T0, "msedge.exe", "Body blueprints", idle_s=10, dwell_min=3)
    a.asked("moment", "How are you holding up today?", T0, ctx=ctx)
    a.hear("tired but fine", T0 + 6)
    check("the question and the answer are one episode", len(e.rows) == 1 and "How are you holding up today?" in e.rows[0][0]
          and "tired but fine" in e.rows[0][0] and "Immortal answered" in e.rows[0][0], str(e.rows))
    check("typed as an ask", e.rows[0][1] == "ask")
    check("where the user was", "Body blueprints" in e.rows[0][0], e.rows[0][0])
    a.asked("moment", "Want a joke?", T0 + 50)
    a.hear("no", T0 + 52)
    check("a no is remembered too", len(e.rows) == 2 and "said no" in e.rows[1][0], str(e.rows[1:]))
    a.asked("moment", "Another?", T0 + 100)
    a.expire(T0 + 1000)
    check("one nobody answered is not an episode", len(e.rows) == 2)
    g = make(episodic=e)
    g.asked("moment", "Hi?", T0)
    g.hear("hello", T0 + 1, guest=True)
    check("nor is a guest's", len(e.rows) == 2)


def a5_timing():
    print("\n=== A5. Timing ===")
    d = tempfile.mkdtemp()
    tm = TimingModel(os.path.join(d, "timing.json"))
    morning_code = {"part": "morning", "app": "Code", "mode": "pausing", "dwell": "long", "hour": 10}
    night_edge = {"part": "night", "app": "Edge", "mode": "reading", "dwell": "long", "hour": 1}
    check("no evidence: neutral", tm.fit(morning_code) == 1.0)
    for _ in range(6):
        tm.update(night_edge, "declined")
        tm.update(morning_code, "accepted")
    check("said no six times at night in Edge: much less welcome there", tm.fit(night_edge) < 0.6, f"{tm.fit(night_edge):.2f}")
    check("said yes six times in the code editor: more welcome", tm.fit(morning_code) > 1.2, f"{tm.fit(morning_code):.2f}")
    other = {"part": "night", "app": "Code", "mode": "typing", "dwell": "just_started", "hour": 2}
    check("another program at night is only somewhat affected", 0.4 < tm.fit(other) <= 1.2, f"{tm.fit(other):.2f}")
    keys = keys_of(night_edge)
    check("the situation is kept at several levels", "part:night|app:Edge" in keys and "*" in keys, str(keys))
    tm2 = TimingModel(os.path.join(d, "timing.json"))
    check("it persists", abs(tm2.fit(night_edge) - tm.fit(night_edge)) < 1e-9 and tm2.updates == 12)
    words = tm.describe()
    check("and says it in words", "leave you alone" in words and "Edge" in words, words)
    later = TimingModel()
    for _ in range(5):
        later.update(night_edge, "later")
    check("'later' counts against too", later.fit(night_edge) < 0.7, f"{later.fit(night_edge):.2f}")
    ign = TimingModel()
    for _ in range(5):
        ign.update(night_edge, "ignored")
    check("being ignored counts only a little against", 0.7 < ign.fit(night_edge) < 1.0, f"{ign.fit(night_edge):.2f}")
    bad = TimingModel(os.path.join(d, "nope", "x.json"))
    check("a missing or unwritable file is no trouble", bad.fit(night_edge) == 1.0)


def a6_context_and_report():
    print("\n=== A6. Context and report ===")
    a = make()
    check("no report before anything", a.report() == "I haven't asked you anything yet.")
    a.asked("moment", "How are you holding up today?", T0)
    check("what the model is told while an ask is waiting", "How are you holding up today?" in a.context_line(T0 + 4)
          and "may be their answer" in a.context_line(T0 + 4))
    a.hear("fine", T0 + 5)
    check("nothing once it is settled", a.context_line(T0 + 6) == "")
    a.asked("wonder", "Q2?", T0 + 20)
    a.hear("no", T0 + 21)
    rep = a.report()
    check("the report counts them", "asked 2 things" in rep and "answered 1" in rep and "said no to 1" in rep, rep)
    check("status", a.status()["asked"] == 2)
    h = a.history()
    check("history is oldest to newest and has the reply", [r["text"] for r in h] == ["How are you holding up today?", "Q2?"] and h[0]["reply"] == "fine")


if __name__ == "__main__":
    for fn in (a1_settling, a2_standing, a3_ledger, a4_episodic, a5_timing, a6_context_and_report):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
