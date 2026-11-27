"""tst_teach.py -- what the person taught, used: "Branch, your first novel" in the day's summary and "how's the novel going?"; "call me Khalid" by
voice (it was only in the console's settings), saved and undoable; "Lieth is pronounced leeth" for how ASH says it; and an exam the person
mentioned counted down in the morning briefing. Fakes only; the config file is a temp file.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_teach.py"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import working  # noqa: E402
from src.daemon.activity import Topic  # noqa: E402
from src.senses import actuators as act  # noqa: E402

check = tw.check


def with_branch(rt, secs_by_day):
    rt.wonder.kb.store_term("Branch", "im working on my first novel")
    t = Topic(key="project:branch", name="Branch", kind="project", by_day=dict(secs_by_day), last=time.time())
    rt.activity.topics["project:branch"] = t
    return t


def main():
    print("\n=== E1. What it was taught, in the summaries ===")
    rt = tm.rig()
    today = time.strftime("%Y-%m-%d")
    days = [time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400)) for i in range(4)]
    with_branch(rt, {d: 3600.0 for d in days})
    check("taught: said back as what it is", working.taught(rt.wonder.kb, "Branch") == "your first novel")
    rows = working.top(rt.activity, time.time(), 1, kb=rt.wonder.kb)
    check("the day's summary says it with the name", rows and rows[0][0] == "Branch, your first novel", str(rows))
    used, said = rt.say("how's the novel going?")
    check("'how's the novel going?' -- found by what it is: the week and the streak, no model", not used and said
          and said[-1] == "4 hours on Branch this week, 4 days in a row.", f"{used} {said}")
    used, said = rt.say("how is the thesis going")
    check("something never taught is not guessed", not (said and "on Branch" in said[-1]), str(said))

    print("\n=== E2. Call me ... ===")
    rt = tm.rig()
    rt.ash.user = "Immortal"
    rt.ash.user_aliases = set()
    rt.ash.set_user_name = lambda n: (setattr(rt.ash, "user", n), n)[1]
    cfg = os.path.join(tempfile.mkdtemp(), "daemon.json")
    with open(cfg, "w") as f:
        json.dump({"keep": 1}, f)
    rt.config_path = cfg
    used, said = rt.say("call me Khalid")
    saved = json.load(open(cfg))
    check("set live and saved, the old name kept as one used before, no model", not used and rt.ash.user == "Khalid" and said[-1] == "Okay -- I'll call you Khalid."
          and saved["identity"]["user_name"] == "Khalid" and "Immortal" in saved["identity"]["previous_names"] and saved["keep"] == 1, f"{said} {saved}")
    used, said = rt.say("undo that")
    check("undone", rt.ash.user == "Immortal", f"{rt.ash.user} {said}")
    for t in ("call me back in 5 minutes", "call me later", "call me when it's done"):
        rt2 = tm.rig()
        rt2.ash.user = "Immortal"
        rt2.ash.set_user_name = lambda n: (setattr(rt2.ash, "user", n), n)[1]
        rt2.say(t)
        check(f"{t!r} is not a name", rt2.ash.user == "Immortal", rt2.ash.user)

    print("\n=== E3. How a word is said ===")
    rt = tm.rig()
    rt.pronounce_path = os.path.join(tempfile.mkdtemp(), "pronounce.json")
    act.LEXICON.clear()
    used, said = rt.say("Lieth is pronounced leeth")
    check("taught by rule", not used and said[-1] == "Got it -- I'll say Lieth like “leeth”.", f"{used} {said}")
    check("said that way from now on (whole words only)", act.speakable("Lieth called. Liethson too.") == "leeth called. Liethson too.", act.speakable("Lieth called. Liethson too."))
    check("kept on disk", json.load(open(rt.pronounce_path)) == {"lieth": "leeth"})
    rt.say("undo that")
    check("undone", "lieth" not in act.LEXICON and act.speakable("Lieth called.") == "Lieth called.")
    rt.say("say Iqnaibi like ick-nay-bee")
    check("'say X like Y' too", act.LEXICON.get("iqnaibi") == "ick-nay-bee", str(act.LEXICON))
    act.LEXICON.clear()

    print("\n=== E4. An exam counted down in the morning ===")
    from src.daemon.followups import FollowUp
    rt = tm.rig()
    if rt.followups is None:
        from src.daemon.followups import FollowUps
        rt.followups = FollowUps(clock=lambda: rt.clock()) if "clock" in FollowUps.__init__.__code__.co_varnames else FollowUps()
    now = time.time()
    rt.followups.items["x"] = FollowUp(id="x", key="nosql exam", what="your NoSQL exam", noun="exam", start=now + 3 * 86400 + 60, clock=True,
                                       due=now + 3 * 86400 + 7200, created=now)
    line = rt._exam_briefing("morning", now)
    check("in the morning: how many days", line == "Your NoSQL exam is in 3 days.", str(line))
    check("not in the evening", rt._exam_briefing("evening", now) is None)
    rt.followups.items["y"] = FollowUp(id="y", key="dentist", what="the dentist", noun="dentist", start=now + 86400, clock=True, due=now + 90000, created=now)
    check("only exams", rt._exam_briefing("morning", now) == "Your NoSQL exam is in 3 days.")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
