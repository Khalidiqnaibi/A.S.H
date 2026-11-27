"""tst_doing.py -- a calendar ASH can write to, tidy downloads (plan, yes, undo; nothing deleted), duplicates, scripts by voice, a research write-up from what is on this
computer, and reading an email's text. Offline; every folder is a temp folder, the script is a fake, the mail fetch is a fake.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_doing.py"""
import os
import sys
import tempfile
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import doing  # noqa: E402
from src.daemon.calendar_ics import Calendar, CalendarConfig  # noqa: E402
from src.daemon.inbox import Inbox, InboxConfig  # noqa: E402

check = tw.check


def main():
    print("\n=== D1. Writing to the calendar ===")
    now = datetime(2026, 11, 27, 10, 0).timestamp()                     # a Friday
    e = doing.parse_event("put dinner with Omar on Saturday at 8", now)
    check("'dinner at 8' is the evening, an hour", e["title"] == "dinner with Omar" and e["start"] == datetime(2026, 11, 28, 20, 0) and e["end"] == datetime(2026, 11, 28, 21, 0), str(e))
    e = doing.parse_event("schedule a meeting with the team tomorrow at 9:30 am for half an hour", now)
    check("morning, half an hour", e["start"] == datetime(2026, 11, 28, 9, 30) and (e["end"] - e["start"]).seconds == 1800, str(e))
    check("not an event", doing.parse_event("put the kettle on", now) is None)
    path = os.path.join(tempfile.mkdtemp(), "ash.ics")
    undo = doing.append_event(path, "Dinner, with Omar", datetime(2026, 11, 28, 20, 0), datetime(2026, 11, 28, 21, 0))
    cal = Calendar(CalendarConfig.from_dict({"files": [path]}), background=False)
    cal.refresh()
    evs = cal.events_on(datetime(2026, 11, 28).date())
    check("the calendar reader reads what was written", [x.summary for x in evs] == ["Dinner, with Omar"], str(evs))
    check("a clash is found", len(doing.conflicts(evs, datetime(2026, 11, 28, 20, 30), datetime(2026, 11, 28, 21, 30))) == 1 and not doing.conflicts(evs, datetime(2026, 11, 28, 9, 0), datetime(2026, 11, 28, 10, 0)))
    undo()
    cal._mtimes.clear()
    cal._texts.clear()
    cal.refresh()
    check("undone: exactly that event is gone", not cal.events_on(datetime(2026, 11, 28).date()))
    rt = tm.rig()
    rt.cfg.life_dir = tempfile.mkdtemp()
    used, said = rt.say("put dinner with Omar on Saturday at 8")
    check("by voice: asked first, no model", not used and said[-1].endswith("Add it?"), f"{used} {said}")
    used, said = rt.say("yes")
    check("a yes writes it", said[-1].startswith("Added dinner with Omar on Saturday at 20:00.") and os.path.exists(os.path.join(rt.cfg.life_dir, "ash-calendar.ics")), str(said))

    print("\n=== D2. Tidy downloads ===")
    d = tempfile.mkdtemp()
    old = time.time() - 3 * 86400
    for n, age in (("holiday.JPG", old), ("paper.pdf", old), ("setup.exe", old), ("fresh.pdf", time.time()), ("movie.mp4.crdownload", old), ("notes", old), ("data.xyz", old)):
        p = os.path.join(d, n)
        open(p, "w").write("x")
        os.utime(p, (age, age))
    plan = doing.tidy_plan(d, time.time())
    check("old, known kinds only; new ones, partials, unknown kinds and extensionless stay", sorted(os.path.basename(a) for a, _ in plan) == ["holiday.JPG", "paper.pdf", "setup.exe"], str(plan))
    check("said before it is done", doing.tidy_say(plan) == "I'd move 1 to Documents, 1 to Images, 1 to Installers. Nothing is deleted, and “undo that” puts them back. Go ahead?", doing.tidy_say(plan))
    n, undo = doing.tidy_do(plan)
    check("moved into folders", n == 3 and os.path.exists(os.path.join(d, "Images", "holiday.JPG")) and not os.path.exists(os.path.join(d, "paper.pdf")))
    check("undone: every one back", undo() == 3 and os.path.exists(os.path.join(d, "paper.pdf")))
    rt = tm.rig()
    rt._downloads_dir = d
    used, said = rt.say("tidy my downloads")
    used, said2 = rt.say("yes")
    check("by voice", said[-1].startswith("I'd move") and said2[-1].startswith("Done -- moved 3 files."), f"{said} {said2}")
    rt.say("undo that")
    check("'undo that' puts them back", os.path.exists(os.path.join(d, "setup.exe")))

    print("\n=== D3. Duplicates ===")
    d2 = tempfile.mkdtemp()
    for n, body in (("a.txt", "same"), ("a (1).txt", "same"), ("b.txt", "other"), ("c.txt", "oth")):
        open(os.path.join(d2, n), "w").write(body)
    g = doing.find_dupes(d2)
    check("same size and same content only", len(g) == 1 and [os.path.basename(p) for p in g[0]] == ["a (1).txt", "a.txt"], str(g))
    n, undo = doing.move_dupes_aside(g, d2)
    check("the extra copy is moved aside, never deleted", n == 1 and os.path.exists(os.path.join(d2, "_duplicates", "a.txt")) and os.path.exists(os.path.join(d2, "a (1).txt")), str(os.listdir(d2)))

    print("\n=== D4. Scripts ===")
    sdir = tempfile.mkdtemp()
    bat = os.path.join(sdir, "backup.bat")
    open(bat, "w").write("echo ok")
    ran = []
    S = doing.Scripts(path=os.path.join(sdir, "s.json"), run=lambda p: (ran.append(p), (0, "copied 3 files\nDone"))[1])
    r = S.register(f"my script back up my project is {bat}")
    check("registered, with how to run it", r and r[0].startswith("Okay -- “back up my project” runs backup.bat."))
    check("found by name", S.which("run back up my project") == "back up my project" and S.which("run something else") is None)
    out = S.run("back up my project")
    check("run, said how it ended, logged", out == "back up my project finished. Last line: Done" and ran == [bat] and S.last["code"] == 0, out)
    check("'how did that script go'", S.last_say("how did that script go") == "back up my project: it worked. Last line: Done")
    rt = tm.rig()
    rt._script_book = S
    used, said = rt.say("run back up my project")
    check("by voice it asks first", said[-1].startswith("Run backup.bat?") and len(ran) == 1, str(said))
    rt.say("yes")
    check("...and a yes runs it", len(ran) == 2)

    print("\n=== D5. A research write-up ===")
    check("parsed", doing.parse_writeup("write up what you have on NPUs") == "NPUs" and doing.parse_writeup("write up that") is None)
    rt = tm.rig()
    rt.wonder.kb.store_term("NPU", "a neural processing unit on my laptop")
    rt.say("remember that the NPU on my laptop is an Intel AI Boost")
    rt._life().screen.add("Neural processing units accelerate matrix multiplication; the NPU shares memory with the CPU on laptops like this one", "opera.exe", "NPU explained", rt.clock())
    home = tempfile.mkdtemp()
    os.environ["USERPROFILE"], os.environ["HOME"] = home, home
    used, said = rt.say("write up what you have on NPU")
    docs = os.path.join(home, "Documents", "ASH")
    files = os.listdir(docs) if os.path.isdir(docs) else []
    body = open(os.path.join(docs, files[0]), encoding="utf-8").read() if files else ""
    check("a markdown file from terms, notes and the screen, with where each came from", not used and files == ["research-NPU.md"] and "## What I have" in body and "## From your notes" in body
          and "## Seen on screen" in body and "Wikipedia" not in body, f"{said} {files}")

    print("\n=== D6. Reading an email ===")
    ib = Inbox(InboxConfig.from_dict({"enabled": True, "host": "imap.x", "user": "u", "password": "p"}), background=False,
               fetch=lambda cfg: (1, [("Omar", "dinner?", "omar@x.com", "<m1@x>")]))
    ib.refresh(force=True)
    time.sleep(0.05)
    ib.body_fetch = lambda cfg, mid: "Hi! Are you free Saturday at 8 for dinner? Bring the laptop." if mid == "<m1@x>" else ""
    s = ib.hear("what does Omar's email say")
    check("the text of the newest one from him", s == "Omar wrote, “dinner?”: Hi! Are you free Saturday at 8 for dinner? Bring the laptop.", str(s))
    ib.body_fetch = lambda cfg, mid: (_ for _ in ()).throw(OSError("no route"))
    check("no connection: still says the subject", "the subject is “dinner?”" in ib.hear("read me the latest email"), str(ib.hear("read me the latest email")))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
