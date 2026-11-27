"""tst_presence.py -- reopening what was open after a shutdown that wasn't clean, "I was updated", a read-only self test, banter, and more cut-off words. Offline; git is a
fake, nothing is opened for real.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_presence.py"""
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon import presence as pr  # noqa: E402
from src.daemon.crashwatch import CrashWatch  # noqa: E402

check = tw.check


def main():
    print("\n=== P1. The run before this one did not say goodbye ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "crashes.json")
    c1 = CrashWatch(path=p, clock=lambda: 1000.0, reader=lambda: [], background=False)
    check("a first run: nothing before it", c1.unclean_before is False)
    c1.heartbeat(1100.0)
    c2 = CrashWatch(path=p, clock=lambda: 5000.0, reader=lambda: [], background=False)
    check("the second start finds the first one never ended", c2.unclean_before is True)
    c2.shutdown(5100.0)
    c3 = CrashWatch(path=p, clock=lambda: 9000.0, reader=lambda: [], background=False)
    check("after a clean stop it does not", c3.unclean_before is False)

    print("\n=== P2. What was open ===")
    oa = pr.OpenApps(os.path.join(d, "o.json"), clock=lambda: 1000.0)
    for a in ("opera.exe", "WINWORD.EXE", "python.exe", "notepad.exe", "opera.exe"):
        oa.see(a)
    oa.save(1000.0, force=True)
    check("programs, most recent last, ASH's own and the system's left out", oa.seen == ["winword", "notepad", "opera"], str(oa.seen))
    check("read back after a crash", pr.OpenApps(os.path.join(d, "o.json"), clock=lambda: 2000.0).before_crash() == ["winword", "notepad", "opera"])
    check("not if it is old", pr.OpenApps(os.path.join(d, "o.json"), clock=lambda: 1000.0 + 3 * 86400).before_crash() == [])
    check("said as a question", pr.recovery_line(["winword", "opera"], str.title) == "It looks like the computer shut down unexpectedly. You had Winword and Opera open -- want me to reopen them?")
    rt = tm.rig()
    rt.desktop = td.drig()
    P = rt._presence()
    P.pending = {"apps": ["notepad", "opera"], "at": rt.clock()}
    used, said = rt.say("yes")
    check("a yes reopens them", rt.desktop.ex_.opened == ["notepad", "opera"] and said[-1] == "Reopening notepad, opera.", f"{rt.desktop.ex_.opened} {said}")

    print("\n=== P3. 'I was updated' ===")
    log = "Gemini chats are chats, not projects: a title in Gemini\nSpeed, all on this machine (speedups.py): the same factual question\nSmall bugs from 27 Nov: 'Under a minute'"
    s = pr.updated_line("aaa", "bbb", run=lambda cmd: log)
    check("from the real commit subjects", s == "I was updated: Gemini chats are chats, not projects; Speed, all on this machine (speedups.py); Small bugs from 27 Nov.", s)
    check("nothing changed: nothing said", pr.updated_line("aaa", "aaa") == "" and pr.updated_line("a", "") == "")
    check("git cannot say: nothing said", pr.updated_line("a", "b", run=lambda cmd: (_ for _ in ()).throw(OSError("no git"))) == "")

    print("\n=== P4. A self test, read-only ===")
    rows = pr.smoke({"memory": lambda: (True, "61% used"), "volume": lambda: (True, "18%"), "notifications": lambda: (False, "no notification database"),
                     "screen reader": lambda: (_ for _ in ()).throw(RuntimeError("no OCR engine"))})
    check("each one ok or not, with why", pr.smoke_say(rows) == "2 of 4 work here: memory, volume. Not working: notifications (no notification database); screen reader (RuntimeError: no OCR engine).", pr.smoke_say(rows))
    rt = tm.rig()
    rt._smoke_checks = {"memory": lambda: (True, "ok"), "git": lambda: (True, "abc")}
    used, said = rt.say("run a smoke test")
    check("by voice, no model", not used and said[-1] == "2 of 2 work here: memory, git.", f"{used} {said}")
    real = pr.smoke(pr.default_checks(tm.rig()))
    check("the real checks run here without changing anything (they report, whatever they find)", len(real) == 8 and all(isinstance(r[1], bool) for r in real), str([(r[0], r[1]) for r in real]))

    print("\n=== P5. Banter ===")
    b = pr.Banter(rng=random.Random(2), clock=lambda: 86400.0 * 3)
    f1 = b.hear("tell me something interesting")
    check("a fact, not the same twice in a row", f1 in pr.FACTS and all(b.hear("fun fact") != f1 for _ in range(1)))
    check("would you rather", b.hear("would you rather") in pr.WYR)
    check("word of the day is the same all day", b.hear("what's the word of the day") == "Today's word is “" + pr.WORDS[3][0] + "”: " + pr.WORDS[3][1] + ".")
    check("a tongue twister", b.hear("give me a tongue twister").endswith("Try it three times fast."))
    check("not banter", b.hear("what time is it") is None)
    rt = tm.rig()
    used, said = rt.say("tell me something interesting")
    check("in the runtime, no model", not used and said[-1] in pr.FACTS, f"{used} {said}")

    print("\n=== P6. More cut-off lines are asked about ===")
    for line, want in (("can you tell me what", "I only caught part of that -- say the rest?"), ("I wonder if", "I only caught part of that -- say the rest?"), ("show me how", "I only caught part of that -- say the rest?")):
        rt = tm.rig()
        used, said = rt.say(line, source="microphone")
        check(f"{line!r}", not used and said and said[-1] == want, f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("what is that", source="microphone")
    check("a whole short question is not cut", not (said and said[-1].startswith("I only caught part")), f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
