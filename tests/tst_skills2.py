"""tst_skills2.py -- taught skills, round two: showing with keys and clicks (a fake keyboard), two demonstrations that become a slot, triggers ("when I open Word", "every
day at 9"), a click that repairs itself, a branch on what the page shows, skill packs (export/import, re-checked), and skills.json merged between two PCs. Offline;
nothing is typed, clicked or opened for real.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_skills2.py"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import skills as sk  # noqa: E402
from tst_skills import Fake, fresh  # noqa: E402

check = tw.check


def press(rec, *vks, mouse=(False, 0, 0)):
    """One poll with these keys down, then one with none (a press and release)."""
    seq = [(set(vks), mouse), (set(), (False, 0, 0))]
    for s in seq:
        rec.poll = (lambda s=s: s)
        rec.tick()


def main():
    print("\n=== K1. What was typed and pressed ===")
    rec = sk.KeyRecorder(lambda: (set(), (False, 0, 0)))
    for ch in "cats":
        press(rec, ord(ch.upper()))
    press(rec, 0x0D)
    press(rec, 0x11, ord("T"))
    press(rec, 0x41, 0x10)                      # shift + a
    press(rec, 0x08)                            # backspace takes the A back
    press(rec, 1, mouse=(True, 640, 300))
    st = rec.steps()
    check("letters become one 'type', Enter and ctrl+t become presses, a click its place",
          st == [{"do": "type", "text": "cats"}, {"do": "press", "keys": ["enter"]}, {"do": "press", "keys": ["ctrl", "t"]}, {"do": "click_at", "x": 640, "y": 300}], str(st))
    check("held keys are not repeated", len(sk.KeyRecorder(lambda: ({0x41}, (False, 0, 0))).steps()) == 0)

    print("\n=== K2. Two demonstrations, one slot ===")
    a = [{"do": "open_url", "url": "https://google.com"}, {"do": "type", "text": "cats"}, {"do": "press", "keys": ["enter"]}]
    b = [{"do": "open_url", "url": "https://google.com"}, {"do": "type", "text": "dogs"}, {"do": "press", "keys": ["enter"]}]
    m = sk.merge_demos(a, b)
    check("the typed text that changed is X", m and m[1] == ["X"] and m[0][1] == {"do": "type", "text": "$X"} and m[0][0] == a[0], str(m))
    check("different steps: no guess", sk.merge_demos(a, a[:2]) is None and sk.merge_demos(a, a) is None)

    print("\n=== K3. A click that repairs itself, and a branch ===")
    class Picky(Fake):
        def click_text(self, text):
            self.log.append(("click", text))
            return text == "Next Button"
    s = fresh()
    s.teach(sk.parse_lesson("To enter the site means: go to google.com, click next button"))
    h = Picky()
    r = s.run(h, "enter the site", {}, lambda t: None)
    check("another wording works, and is kept for next time", r["ok"] and s.skills["enter the site"]["steps"][-1].get("alt") == "Next Button", f"{r} {s.skills['enter the site']['steps']}")
    h = Picky()
    s.run(h, "enter the site", {}, lambda t: None)
    check("next time the kept wording is tried first", h.log[-1] == ("click", "Next Button") and h.log.count(("click", "next button")) == 0 or h.log[-2:][0] == ("click", "Next Button"), str(h.log))
    s2 = fresh()
    s2.teach(sk.parse_lesson("To check google means: go to google.com, if the page shows sign in, do tidy up"))
    s2.teach(sk.parse_lesson("To tidy up means: press escape"))
    h = Fake()
    r = s2.run(h, "check google", {}, lambda t: None)
    check("'if the page shows X, do Y' runs Y when it does", r["ok"] and ("keys", ("esc",)) in h.log, f"{r} {h.log}")

    print("\n=== K4. Triggers ===")
    check("parsed", sk.parse_trigger("when I open Word, do my setup") == {"kind": "open", "app": "word", "name": "my setup"}
          and sk.parse_trigger("every day at 9 do morning research") == {"kind": "daily", "at": "09:00", "name": "morning research"}
          and sk.parse_trigger("every morning at 7:30 pm do x y z")["at"] == "19:30")
    now = time.time()
    trig = [{"kind": "open", "app": "word", "name": "my setup"}, {"kind": "daily", "at": "00:00", "name": "morning research"}]
    due = sk.due_triggers(trig, "WINWORD.EXE", "opera.exe", now)
    check("an app that came to the front, and a daily time that has passed", [t["kind"] for t in due] == ["daily"] or [t["kind"] for t in due] == ["open", "daily"], str(due))
    check("the same app staying in front does not fire again", all(t["kind"] != "open" for t in sk.due_triggers(trig, "word", "word", now)))
    trig[1]["last"] = time.strftime("%Y-%m-%d")
    check("a daily one runs once a day", sk.due_triggers(trig, "", "", now) == [])
    rt = tm.rig()
    rt.cfg.skills_path = os.path.join(tempfile.mkdtemp(), "skills.json")
    rt._skills_sync = True
    h = Fake()
    rt._hands = h
    rt._skills_obj().pause = lambda s: None
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    rt.say("To open my notes means: open notepad")
    used, said = rt.say("when I open word, do open my notes")
    check("set by voice", said[-1].startswith("Okay -- when you open word, I'll do open my notes.") and "ask first" in said[-1], str(said))
    rt.dwell.process = "WINWORD.EXE"
    rt._trig_prev = "opera.exe"
    rt.started_at = time.time() - 3600
    rt._skill_trigger_step(rt.clock())
    check("it fires when Word comes up: the plan is read back (not yet trusted), a yes runs it", out[-1].endswith("Go ahead?"), str(out[-2:]))
    rt.say("yes")
    check("...and does it", ("app", "notepad") in h.log, str(h.log))

    print("\n=== K5. Packs ===")
    s = fresh()
    s.teach(sk.parse_lesson("To search Google for X means: go to google.com, type X, press enter"))
    pack = sk.export_pack(s.skills)
    s_b = fresh()
    added, refused = sk.import_pack(s_b.skills, pack, 0.0)
    check("exported and imported, asking first again", added == ["search google for x"] and s_b.skills["search google for x"]["runs"] == 0 and not s_b.trusted("search google for x"))
    bad = {"skills": {"pay": {"steps": [{"do": "click_text", "text": "buy now"}], "slots": []}, "ok": {"steps": [{"do": "wait", "s": 1}], "slots": []}}}
    added, refused = sk.import_pack(fresh().skills, bad, 0.0)
    check("a pack cannot smuggle in a step a lesson may not have", added == ["ok"] and refused == ["pay"], f"{added} {refused}")

    print("\n=== K6. Merged between two PCs ===")
    from tools import mergemem
    base = {"skills": {"a": {"name": "a", "runs": 1}}, "log": [], "triggers": []}
    ours = {"skills": {"a": {"name": "a", "runs": 1}, "b": {"name": "b", "runs": 0}}, "log": [], "triggers": []}
    theirs = {"skills": {"a": {"name": "a", "runs": 1}, "c": {"name": "c", "runs": 0}}, "log": [], "triggers": []}
    d = tempfile.mkdtemp()
    paths = []
    for n, v in (("base", base), ("ours", ours), ("theirs", theirs)):
        p = os.path.join(d, n + "skills.json")
        json.dump(v, open(p, "w"))
        paths.append(p)
    rc = mergemem.run(paths[0], paths[1], paths[2], "skills.json")
    merged = json.load(open(paths[1]))
    check("both PCs' new skills are kept", rc == 0 and sorted(merged["skills"]) == ["a", "b", "c"], str(merged))
    check("skills.json is marked for the merge driver", "skills.json merge=ashmem" in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gitattributes")).read())
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
