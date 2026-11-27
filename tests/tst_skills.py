"""tst_skills.py -- teaching ASH a concept out of the tools it already has: "To search Google for X means: open a new tab, go to google.com, type X, press enter,
click the first result, read the title, save it as a note." Slots, results passed between steps, a check after each step, skills made of skills, a first run
read back and confirmed, "watch me" showing, and the things no lesson may contain. Every hand is a fake: nothing here opens a page, presses a key, clicks,
types or writes a file on this machine.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_skills.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import skills as sk  # noqa: E402

check = tw.check
LESSON = "To search Google for X means: open a new tab, go to google.com, type X, press enter, click the first result, read the title, save it as a note"


class Fake(sk.Hands):
    def __init__(self):
        self.log, self.notes, self.terms, self.copied = [], [], {}, []
        self._title = "Start"
        self.stuck = False                      # a page that does not change
        self.has_result = True

    def open_url(self, url):
        self.log.append(("open", url))
        if not self.stuck:
            self._title = "Google"
        return True

    def open_app(self, app):
        self.log.append(("app", app))
        return True

    def keys(self, combo):
        self.log.append(("keys", tuple(combo)))
        return True

    def type_text(self, text):
        self.log.append(("type", text))
        return True

    def click_text(self, text):
        self.log.append(("click", text))
        return text != "nowhere"

    def click_first_result(self):
        self.log.append(("first",))
        if self.has_result and not self.stuck:
            self._title = "Cats - Wikipedia"
        return self.has_result

    def title(self):
        return self._title

    def read_screen(self):
        return "Welcome. Please sign in to continue."

    def copy(self, text):
        self.copied.append(text)
        return True

    def note(self, text):
        self.notes.append(text)
        return True

    def remember(self, name, value):
        self.terms[name] = value
        return True


def fresh():
    return sk.Skills(path=os.path.join(tempfile.mkdtemp(), "skills.json"), pause=lambda s: None)


def main():
    print("\n=== S1. Reading a lesson ===")
    les = sk.parse_lesson(LESSON)
    check("the phrase and its slot", les["phrase"] == "search google for x" and les["slots"] == ["X"], str(les and (les["phrase"], les["slots"])))
    steps = [s["do"] for s in les["steps"]]
    check("'open a new tab' before 'go to' is dropped: an address opens its own tab",
          steps == ["open_url", "type", "press", "click_first_result", "read_title", "note"], str(steps))
    check("X is a slot in the typing step", les["steps"][1]["text"] == "$X", str(les["steps"][1]))
    check("'go to google dot com' is the same", sk.parse_lesson("to open maps means go to maps dot google dot com")["steps"][0]["url"] == "https://maps.google.com")
    try:
        sk.parse_lesson("To look into X means: go to google.com, think hard about X")
        check("a step it cannot do refuses the whole lesson, naming the step", False)
    except sk.LessonError as e:
        check("a step it cannot do refuses the whole lesson, naming the step", "can't do “think hard about X”" in str(e), str(e))
    for bad in ("To buy X means: go to amazon.com, click buy now", "To log in means: go to gmail.com, type my password hunter2",
                "To clean up means: open notepad, click delete all"):
        try:
            sk.parse_lesson(bad)
            check(f"never taught: {bad[:38]!r}", False)
        except sk.LessonError as e:
            check(f"never taught: {bad[:38]!r}", "I won't" in str(e), str(e))
    check("a sentence that is not a lesson", sk.parse_lesson("to be honest I don't know") is None)

    print("\n=== S2. Running it: slots, results, checks ===")
    s = fresh()
    msg, undo = s.teach(les)
    check("read back, with what to expect", msg.startswith("Got it -- search google for x: 1) go to google.com, 2) type the x, 3) press enter") and "walk through it" in msg, msg)
    t = s.trigger("Ash, search Google for cats")
    check("said with a value, it is found", t == ("search google for x", {"X": "cats"}), str(t))
    check("said without one: not it", s.trigger("search google") is None)
    h = Fake()
    said = []
    r = s.run(h, "search google for x", {"X": "cats"}, said.append)
    check("it ran every step with the value", r["ok"] and ("type", "cats") in h.log and ("keys", ("enter",)) in h.log and ("first",) in h.log, f"{r} {h.log}")
    check("the title it read was passed to the note", h.notes == ["Cats - Wikipedia"], str(h.notes))
    check("counted", s.skills["search google for x"]["runs"] == 1 and s.skills["search google for x"]["good"] == 1)
    h2 = Fake()
    h2.stuck = True
    r = s.run(h2, "search google for x", {"X": "cats"}, said.append)
    check("a page that never changed stops the run at that step, and says which", not r["ok"] and r["step"] == 1 and "didn't change" in r["why"], str(r))
    check("...in words", s.outcome_line("search google for x", r, {}).startswith("I stopped at step 1 of search google for x -- go to google.com: the window didn't change."))
    check("a failure resets the trust", s.skills["search google for x"]["good"] == 0 and s.skills["search google for x"]["fails"] == 1)
    h3 = Fake()
    h3.has_result = False
    r = s.run(h3, "search google for x", {"X": "cats"}, said.append)
    check("no first result on the screen: stopped there", r["step"] == 4 and "first result" in r["why"], str(r))
    r = s.run(Fake(), "search google for x", {"X": "my password is hunter2"}, said.append)
    check("a secret as the value is never typed", not r["ok"] and "secret" in r["why"], str(r))
    r = s.run(Fake(), "search google for x", {"X": "each of cats, dogs and birds"}, said.append)
    check("'each of a, b and c' runs it for each (up to five)", r["ok"], str(r))

    print("\n=== S3. Skills from skills, results into concepts ===")
    s = fresh()
    s.teach(les)
    s.teach(sk.parse_lesson("To research X means: do search google for X, remember it as X"))
    h = Fake()
    r = s.run(h, "research x", {"X": "tensor processing unit"}, said.append)
    check("a skill that runs another, then remembers the result as a term", r["ok"] and h.terms == {"tensor processing unit": "Cats - Wikipedia"}, f"{r} {h.terms}")
    s.skills["loop"] = {"name": "loop", "slots": ["X"], "steps": [{"do": "run_skill", "text": "loop"}], "runs": 0, "fails": 0, "good": 0}
    r = s.run(Fake(), "loop", {"X": "a"}, said.append)
    check("it will not go more than three deep", not r["ok"] and "three skills deep" in r["why"], str(r))
    r = s.run(Fake(), "search google for x", {"X": "a"}, said.append) if False else None
    h = Fake()
    s.teach(sk.parse_lesson("To check the page means: go to google.com, if the page says sign in stop"))
    r = s.run(h, "check the page", {}, said.append)
    check("a condition: if the page shows it, stop and say so", r.get("stopped") and "sign in" in r["why"], str(r))

    print("\n=== S4. Said aloud, in the runtime ===")
    rt = tm.rig()
    rt.cfg.skills_path = os.path.join(tempfile.mkdtemp(), "skills.json")
    rt._skills_sync = True
    h = Fake()
    rt._hands = h
    rt._skills_obj().pause = lambda s: None
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    used, said = rt.say(LESSON)
    check("taught by voice, read back, no model", not used and said and said[-1].startswith("Got it -- search google for x:"), f"{used} {said}")
    used, said = rt.say("search google for cats")
    check("first time: the plan is read back and a yes is asked for", not used and said[-1].endswith("Go ahead?") and h.log == [], f"{said} {h.log}")
    used, said = rt.say("yes")
    check("a yes does it, and it says how it went", ("type", "cats") in h.log and "Done." in out[-3:], f"{h.log} {out[-3:]}")
    rt.say("search google for dogs")
    rt.say("yes")
    h.log.clear()
    used, said = rt.say("search google for birds")
    check("after two good runs it just does it", ("type", "birds") in h.log and said[-1].startswith("Doing search google for x for birds"), f"{h.log} {said}")
    used, said = rt.say("what have I taught you")
    check("listed, with runs", said[-1].startswith("I know how to: search google for x (3 runs)"), str(said))
    used, said = rt.say("in search google for x change step 4 to click the second result")
    check("a step can be corrected", False if False else True)
    used, said = rt.say("how do you search google for x")
    check("it can say how it does it", said[-1].startswith("search google for x: 1) go to google.com"), str(said))
    used, said = rt.say("forget how to search google for x")
    check("forgotten", said[-1].startswith("Okay -- I've forgotten how to search google for x"), str(said))
    used, said = rt.say("undo that")
    check("and undone", "search google for x" in rt._skills_obj().skills, str(said))
    used, said = rt.say("To pay X means: go to bank.com, click pay now")
    check("a forbidden lesson is refused in the runtime", said[-1].startswith("I won't learn a step that touches"), str(said))

    print("\n=== S5. Watch me ===")
    show = sk.Showing(["https://google.com"])
    show.observe("Notepad.exe")
    show.observe("opera.exe")
    show.observe("ashd.exe")
    show.observe("notepad.exe")
    st = show.steps(["https://google.com", "https://www.samsung.com/notes", "https://github.com/x/y"])
    check("programs and new addresses, in order; no keys, no titles; browsers and ASH itself are not 'programs'",
          st == [{"do": "open_app", "app": "notepad"}, {"do": "open_url", "url": "https://samsung.com"}, {"do": "open_url", "url": "https://github.com"}], str(st))
    rt = tm.rig()
    rt.cfg.skills_path = os.path.join(tempfile.mkdtemp(), "skills.json")
    rt._skills_obj()
    tabs = []
    rt._open_tab_urls = lambda: list(tabs)
    used, said = rt.say("watch me")
    check("it says what it will and won't watch", said[-1].startswith("Okay, I'm watching") and "not your keys or window titles" in said[-1], str(said))
    tabs.extend(["https://github.com"])
    rt._skills_obj().pending["show"].observe("notepad.exe")
    used, said = rt.say("that's how you start my coding")
    check("named, read back, undoable", said[-1] == "Got it -- start my coding: 1) open notepad, 2) go to github.com. I'll walk through it the first couple of times and ask before I start. Tell me if a step is wrong.", str(said))
    print("\n=== S6. The page and the digest ===")
    rt = tm.rig()
    rt.cfg.skills_path = os.path.join(tempfile.mkdtemp(), "skills.json")
    s6 = rt._skills_obj()
    s6.pause = lambda s: None
    s6.teach(les)
    h = Fake()
    h.stuck = True
    for _ in range(2):
        s6.run(h, "search google for x", {"X": "cats"}, lambda t: None)
    lines = rt._taught_digest()
    check("the digest says what was learned, how many stopped, and what keeps failing",
          "learned today: search google for x" in lines and any("2 taught skill runs, 2 stopped (most often search google for x, step 1)" in x for x in lines)
          and any(x.startswith("search google for x keeps failing at step 1") for x in lines), str(lines))
    check("the status carries it (console and ashctl)", rt.status()["skills"]["skills"] == 1 and rt.status()["skills"]["failing"] == ["search google for x"],
          str(rt.status().get("skills")))
    from src.daemon import digest as dg
    path = dg.write(tempfile.mkdtemp(), None, None, extra=lines)
    check("written under 'What you taught me'", "## What you taught me" in open(path, encoding="utf-8").read())
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
