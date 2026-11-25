"""
tst_samsungnotes.py -- Samsung Notes: read, write a new note, add to a plain text note -- safely.

The app has no API and keeps notes in a format ASH must not touch, so writing is done through the app (UI Automation,
clipboard) and then CHECKED by reading the app's database again; reading is from that database, read-only. These tests use a
database with the app's real table layout and a fake driver; the real thing was driven on this machine by hand (a new note, and
text added to an existing one, both confirmed from the database).

  S1  What is said: new note, titled, dictated, add to a note, read, find, list -- and what is not about Samsung Notes
  S2  Reading from the database
  S3  Writing: read back first, a yes, the steps, the check afterwards, the failure that leaves the text on the clipboard
  S4  Adding to an existing note: only plain text, only one match; never a PDF, handwriting or pictures
  S5  Dictation
  S6  On the runtime: a guest, the overheard room, and the manual

Run:  PYTHONIOENCODING=utf-8 python tests/tst_samsungnotes.py
"""

import os
import sqlite3
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.samsungnotes import NotesDB, SamsungNotes  # noqa: E402

check = tw.check
NOW = 1_800_000_000.0

SCHEMA = """
create table NoteDB (Id integer primary key, UUID text, CategoryUUID text, Title text, DeletedStatus integer default 0,
                     LastModifiedAt integer, CreatedAt integer);
create table TextSearchDB (UUID text, StrippedContent text, PDFTextContents text, FilePath text, HWTextContent text);
create table CategoryTreeDB (UUID text, DisplayName text);
create table CategoryDB (UUID text, DisplayName text);
create table document_extra (UUID text, hasSticky integer, hasImage integer, hasStroke integer, hasPdf integer);
"""


def make_db(path):
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    c.execute("insert into CategoryTreeDB values ('c1','Nosql'), ('c2','Uni')")
    notes = [
        ("n1", "c1", "Groceries", "milk eggs bread", 0, 0, 0, 0, 1000, 900),
        ("n2", "c1", "NoSQL lecture", "key value stores and document databases", 0, 0, 0, 0, 3000, 2000),
        ("n3", "c1", "Chapter 5 slides_aa11bb22cc33dd44ee55", "", 0, 0, 0, 1, 4000, 3000),
        ("n4", "c2", "Chapter 5 sketch", "", 0, 0, 1, 0, 5000, 4000),
        ("n5", "c2", "Chapter 5 summary", "wide column databases and cassandra", 0, 0, 0, 0, 6000, 5000),
        ("n6", "c2", "Old stuff", "gone", 1, 0, 0, 0, 7000, 6500),
    ]
    for u, cat, title, text, st, im, stroke, pdf, mod, created in notes:
        c.execute("insert into NoteDB (UUID,CategoryUUID,Title,DeletedStatus,LastModifiedAt,CreatedAt) values (?,?,?,?,?,?)",
                  (u, cat, title, st, mod * 1000, created * 1000))
        c.execute("insert into TextSearchDB values (?,?,?,?,?)", (u, text, "", "", "recognised handwriting about birds" if u == "n4" else ""))
        c.execute("insert into document_extra values (?,?,?,?,?)", (u, 0, im, stroke, pdf))
    c.commit()
    c.close()


class FakeDriver:
    def __init__(self, ok=True, app=True):
        self.log = []
        self.ok = ok
        self.app = app
        self.launched_by_us = False
        self.clip = "what you had copied"
        self.db = None
        self.written = None

    def available(self):
        return self.app

    def clipboard_get(self):
        return self.clip

    def clipboard_set(self, t):
        self.clip = t

    def foreground(self):
        return 77

    def restore_foreground(self, h):
        self.log.append(("fg", h))

    def launch(self):
        self.log.append("launch")
        return self.ok

    def dismiss_dialog(self):
        self.log.append("dismiss")

    def to_list(self, tries=5):
        self.log.append("to_list")
        return self.ok

    def close_app(self):
        self.log.append("close")


def rig(driver=None, **kw):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "Storage.sqlite")
    make_db(p)
    drv = driver or FakeDriver()
    said = []
    sn = SamsungNotes(db=NotesDB(p), driver=drv, clock=lambda: NOW, backup_dir=os.path.join(d, "backup"), run_async=False,
                      notify=said.append, **kw)
    sn.said, sn.d, sn.dbpath = said, drv, p
    # the steps that really need the app are replaced; everything around them is the real code
    sn.write_new = lambda title, body: (drv.log.append(("new", title, body)) or
                                        ({"ok": True, "verified": True, "title": title or "Note from ASH", "folder": "Nosql"} if drv.ok else
                                         {"ok": False, "verified": False, "why": "the new note didn't open"}))
    sn.append = lambda uuid, title, body: (drv.log.append(("append", uuid, body)) or
                                           ({"ok": True, "verified": True} if drv.ok else
                                            {"ok": False, "verified": False, "why": "the note didn't open"}))
    return sn


def s1_hearing():
    print("\n=== S1. What is said ===")
    sn = rig()
    for t, kind in [("write buy milk in Samsung Notes", "new"), ("put this in Samsung Notes: call Omar at five", "new"),
                    ("new Samsung note called Lecture 5 saying hash tables are fast", "new"),
                    ("make a note in my Samsung Notes called Ideas: build a robot", "new"),
                    ("hey ash, save remember the milk to Samsung Notes", "new"),
                    ("add finish the slides to my NoSQL lecture note in Samsung Notes", "append"),
                    ("add to my Groceries Samsung note: butter", "append")]:
        sn.pending = None
        say = sn.hear(t, NOW)
        check(f"{t!r} -> {kind}", sn.pending is not None and sn.pending["kind"] == kind and "yes" in (say or "").lower(),
              f"{say} {sn.pending}")
    sn = rig()
    sn.hear("new Samsung note called Lecture 5 saying hash tables are fast", NOW)
    check("title and body are told apart", sn.pending["title"] == "Lecture 5" and sn.pending["body"] == "hash tables are fast",
          str(sn.pending))
    sn.pending = None
    sn.hear("write buy milk in Samsung Notes", NOW)
    check("a plain write has the words and no title", sn.pending["body"] == "buy milk" and sn.pending["title"] == "", str(sn.pending))
    for t in ("remember that I parked on level 3", "what time is it", "write a poem", "open Samsung", "what's on my calendar",
              "I take notes in Samsung Notes every day", "put the kettle on"):
        check(f"not for it: {t!r}", rig().hear(t, NOW) in (None,) or "yes" not in (rig().hear(t, NOW) or "").lower(),
              str(rig().hear(t, NOW)))


def s2b_activity():
    print("\n=== S2b. What did I do in Samsung Notes ===")
    import sqlite3
    sn = rig()
    c = sqlite3.connect(sn.dbpath)
    mid = time.mktime(time.localtime(NOW)[:3] + (0, 0, 0, 0, 0, -1))
    for u, ts in (("n5", NOW - 600), ("n4", mid + 60), ("n2", mid - 3600), ("n1", NOW - 3 * 86400)):
        c.execute("update NoteDB set LastModifiedAt=? where UUID=?", (int(ts * 1000), u))
    c.commit()
    c.close()
    for t in ("what did i work on in samsung notes", "what about samsung notes what did i do on it", "what did I do in my Samsung Notes today"):
        say = sn.hear(t, NOW)
        check(f"{t!r} -> today's two", say == "Today you changed 2 notes: Chapter 5 summary and Chapter 5 sketch.", str(say))
    say = sn.hear("what did I work on in Samsung Notes yesterday", NOW)
    check("yesterday", say == "Yesterday you changed 1 note: NoSQL lecture.", str(say))
    say = sn.hear("what have I done in Samsung Notes this week", NOW)
    check("this week: all four, newest first", say.startswith("In the last week you changed 4 notes: Chapter 5 summary, Chapter 5 sketch, NoSQL lecture and Groceries"), str(say))
    say = sn.hear("what did I do on Samsung Notes last week", NOW)
    check("nothing then: said so, with the last one changed", say.startswith("I don't see any Samsung notes changed the week before") and "The last one changed was" in say, str(say))
    for t in ("Lasted note on Samsung Notes.", "the latest note on Samsung Notes", "last note in my Samsung Notes"):
        say = sn.hear(t, NOW)
        check(f"{t!r} reads the last note", say and "Chapter 5 summary" in say, str(say))
    check("not an activity question: what did I write about X is a search", "mention" in (sn.hear("what did i write about cassandra in samsung notes", NOW) or ""))


def s2_reading():
    print("\n=== S2. Reading ===")
    sn = rig()
    check("the notes are counted without the deleted one", sn.db.count() == 5, str(sn.db.count()))
    check("the latest first, with no deleted ones", [n["title"] for n in sn.db.recent(3)][:2] == ["Chapter 5 summary", "Chapter 5 sketch"],
          str([n["title"] for n in sn.db.recent(3)]))
    say = sn.hear("list my Samsung notes", NOW)
    check("'list' says how many and the latest", "5 notes" in say and "Chapter 5 summary" in say, say)
    say = sn.hear("read my last Samsung note", NOW)
    check("the last note's text is read", "wide column databases and cassandra" in say and "Chapter 5 summary" in say, say)
    say = sn.hear("what's in my NoSQL lecture Samsung note", NOW)
    check("a note by name", "key value stores" in say, say)
    say = sn.hear("find cassandra in my Samsung notes", NOW)
    check("a search in the text", "Chapter 5 summary" in say and "1 note" in say, say)
    say = sn.hear("find birds in my Samsung notes", NOW)
    check("...and in recognised handwriting", "Chapter 5 sketch" in say, say)
    say = sn.hear("what's in my Chapter 5 Samsung note", NOW)
    check("two notes fit a name: it asks which", say.startswith("Which one") and "summary" in say, say)
    say = sn.hear("what's in my slides Samsung note", NOW)
    check("a PDF note has nothing to read aloud, and says so (and the id on its name is not spoken)",
          "no text I can read" in say and "aa11bb22" not in say, say)
    check("a note that is not there", "can't find" in sn.hear("what's in my Zebra Samsung note", NOW))
    check("the deleted note is not found", "didn't find" in sn.hear("find gone in my Samsung notes", NOW))
    off = NotesDB(os.path.join(tempfile.mkdtemp(), "none.sqlite"))
    check("without the app: it says so", "can't find Samsung Notes" in SamsungNotes(db=off, driver=FakeDriver()).hear("list my Samsung notes", NOW))
    say = sn.hear("What notes do I have in the NoSQL folder in Samsung Notes?", NOW)
    check("a folder's notes (by the folder's name, any case)", "3 notes" in say and "Groceries" in say and "NoSQL lecture" in say, say)
    say = sn.hear("show me my notes in the Uni folder", NOW)
    check("a folder asked for without the app's name, after 'Samsung notes' was mentioned", say is None or "Uni" in say, str(say))
    check("a folder that is not there", "can't find any notes" in sn.hear("what notes do I have in the Zebra folder in Samsung Notes", NOW))
    say = sn.hear("What can you do with Samsung Notes?", NOW)
    check("what it can do is answered from what it can do, not from memory", "read your notes" in say and "new note" in say, say)
    check("'search for X in Samsung Notes' searches for X (not for 'for X')", "about projects" in sn.hear("Search for projects in Samsung Notes.", NOW))
    check("...and with the place first", "Chapter 5 sketch" in sn.hear("In Samsung Notes search for birds.", NOW))
    check("'open up the last note in Samsung Notes'", "Your last note is" in sn.hear("Open up the last note in Samsung notes.", NOW))
    check("...and the misheard 'and Samsung notes'", "Your last note is" in sn.hear("Open up the last note and Samsung notes.", NOW))
    check("'what notes do I have in Samsung Notes'", "5 notes" in sn.hear("What notes do I have in Samsung Notes", NOW))
    check("'can you write in Samsung Notes' too", "read your notes" in sn.hear("can you write in Samsung Notes?", NOW))
    before = open(sn.dbpath, "rb").read()
    sn.hear("read my last Samsung note", NOW)
    check("reading never changes the database", open(sn.dbpath, "rb").read() == before)


def s3_writing():
    print("\n=== S3. Writing a new note ===")
    sn = rig()
    ask = sn.hear("new Samsung note called Lecture 5 saying hash tables are fast", NOW)
    check("it reads back what it will write, where, and that it takes the mouse", "hash tables are fast" in ask and "Samsung note called Lecture 5" in ask
          and "mouse" in ask, ask)
    check("nothing has happened yet", not [x for x in sn.d.log if isinstance(x, tuple)])
    out = sn.hear("yes", NOW + 5)
    check("on a yes it writes, with the title and the text", ("new", "Lecture 5", "hash tables are fast") in sn.d.log, str(sn.d.log))
    check("...and says it is done, where it landed", out == "Done -- it's in Samsung Notes in Nosql as \"Lecture 5\".", out)
    check("...counted", sn.stats["written"] == 1)

    sn = rig()
    sn.hear("write buy milk in Samsung Notes", NOW)
    out = sn.hear("no", NOW + 3)
    check("'no' writes nothing", "won't write" in out and not [x for x in sn.d.log if isinstance(x, tuple)] and sn.pending is None)
    sn.hear("write buy milk in Samsung Notes", NOW)
    check("a yes after two minutes is not taken", sn.hear("yes", NOW + 120) is None and not [x for x in sn.d.log if isinstance(x, tuple)])

    sn = rig(FakeDriver(ok=False))
    sn.hear("write buy milk in Samsung Notes", NOW)
    out = sn.hear("yes", NOW + 2)
    check("when it cannot confirm the note, it says why and where the text is", "couldn't get that into Samsung Notes" in out and
          "clipboard" in out and "one paste away" in out, out)
    check("...the text IS on the clipboard", sn.d.clip == "buy milk", sn.d.clip)
    check("...and in ASH's own notes folder", any(f.endswith(".txt") for f in os.listdir(os.path.join(os.path.dirname(sn.dbpath), "backup"))))
    check("...counted as a failure", sn.stats["failed"] == 1 and sn.stats["written"] == 0)
    check("...and it offers to try again", "try again" in out, out)
    sn.d.ok = True
    out = sn.hear("Try again Ash.", NOW + 30)
    check("'try again' writes the same note once more, with no second read-back", out.startswith("Done") and
          [x for x in sn.d.log if isinstance(x, tuple)].count(("new", "", "buy milk")) == 2, f"{out!r} {sn.d.log}")
    check("...and after it worked, 'try again' is nothing of ours", sn.hear("try again", NOW + 40) is None)
    sn = rig(FakeDriver(ok=False))
    sn.hear("write buy milk in Samsung Notes", NOW)
    sn.hear("yes", NOW + 2)
    check("a 'try again' long after the failure is not taken", sn.hear("try again", NOW + 600) is None)
    sn = rig()
    check("'open Samsung Notes' opens the app", sn.hear("Ash, open Samsung Notes.", NOW) == "Opening Samsung Notes." and "launch" in sn.d.log, str(sn.d.log))
    check("...'just open the app, open Samsung Notes' too", rig().hear("just open Samsung Notes", NOW) == "Opening Samsung Notes.")
    check("'open my notes' (ASH's own) is not this", rig().hear("open my notes", NOW) is None)

    sn = rig(write=False)
    check("with writing switched off it refuses before reading anything back", "switched off" in sn.hear("write buy milk in Samsung Notes", NOW))
    check("no automation available (not Windows, no comtypes): it says so", "can't drive Samsung Notes" in
          rig(FakeDriver(app=False)).hear("write buy milk in Samsung Notes", NOW))
    sn = rig()
    check("a very long text starts nothing", sn.hear("write " + "word " * 1500 + "in Samsung Notes", NOW) is None and sn.pending is None)

    # the real write_new/append: restoring what the person had
    real = SamsungNotes(db=NotesDB(sn.dbpath), driver=FakeDriver(ok=False), clock=lambda: NOW)
    res = real.write_new("T", "body")
    check("the real write_new gives up cleanly if the app will not open, and the clipboard is put back",
          res["ok"] is False and real.driver.clip == "what you had copied", str(res))
    check("...and the focus is given back", ("fg", 77) in real.driver.log)


def s4_append():
    print("\n=== S4. Adding to an existing note ===")
    sn = rig()
    ask = sn.hear("add finish the slides to my NoSQL lecture note in Samsung Notes", NOW)
    check("a plain text note is offered the addition", "end of \"NoSQL lecture\"" in ask and sn.pending["uuid"] == "n2", ask)
    out = sn.hear("yes", NOW + 3)
    check("on a yes it is added", ("append", "n2", "finish the slides") in sn.d.log and out.startswith("Added to"), f"{sn.d.log} {out}")

    sn = rig()
    ask = sn.hear("add more to my Chapter 5 slides note in Samsung Notes", NOW)
    check("a PDF note is refused, and a new note is offered beside it", "a PDF" in ask and "new Samsung note" in ask and sn.pending is None, ask)
    ask = sn.hear("add more to my Chapter 5 sketch note in Samsung Notes", NOW)
    check("a handwritten note too", "handwritten" in ask and sn.pending is None, ask)
    ask = sn.hear("add more to my Chapter 5 note in Samsung Notes", NOW)
    check("two notes fit the name: it asks which, and writes nowhere", ask.startswith("Which one") and sn.pending is None, ask)
    ask = sn.hear("add more to my Zebra note in Samsung Notes", NOW)
    check("no such note: it offers a new one", "can't find a Samsung note called Zebra" in ask and sn.pending is None, ask)
    ask = sn.hear("add more to my Chapter 5 summary note in Samsung Notes", NOW)
    check("an exact title wins over a longer list of partial matches", sn.pending and sn.pending["uuid"] == "n5", f"{ask} {sn.pending}")


def s5_dictation():
    print("\n=== S5. Dictation ===")
    sn = rig()
    say = sn.hear("start a Samsung note called Novel", NOW)
    check("it starts, with the title", "Go ahead" in say and sn.dictating["title"] == "Novel", say)
    check("what is said next is taken down quietly", sn.hear("It was a dark night.", NOW + 5) == "" and sn.hear("The wind howled.", NOW + 9) == "")
    ask = sn.hear("that's the note", NOW + 12)
    check("at the end it reads back and asks", "dark night" in ask and "Samsung note called Novel" in ask and sn.pending is not None, ask)
    sn.hear("yes", NOW + 15)
    check("...and writes all of it, in order", ("new", "Novel", "It was a dark night.\nThe wind howled.") in sn.d.log, str(sn.d.log))

    sn = rig()
    sn.hear("start a Samsung note", NOW)
    sn.hear("something", NOW + 2)
    check("a dictation can be dropped", "dropped" in sn.hear("scratch that note", NOW + 4) and sn.dictating is None and sn.pending is None)
    sn = rig()
    sn.hear("start a Samsung note", NOW)
    check("an empty one writes nothing", "nothing in it" in sn.hear("that's the note", NOW + 3))
    sn = rig()
    sn.hear("start a Samsung note", NOW)
    sn.hear("a line", NOW + 2)
    check("a dictation left for fifteen minutes is gone, and the next sentence is nobody's business",
          sn.hear("what time is it", NOW + 16 * 60) is None and sn.dictating is None)


def s6_runtime():
    print("\n=== S6. On the runtime ===")
    import tst_effects as te
    from src.daemon import manual
    from src.senses.base import Modality, SensorEvent
    on = te.rig(samsung=True, samsung_cfg={"write": False})
    check("switched on by config, the runtime builds it (writing off as set)", on.samsung is not None and on.samsung.write is False)
    check("...and its status is in the runtime's", "samsung_notes" in on.status() and on.status()["samsung_notes"] is not None,
          str(list(on.status())[:5]))
    rt = te.rig()
    sn = rig()
    check("the runtime has a place for it", hasattr(rt, "samsung"))
    rt.samsung = sn
    rt.say("write buy milk in Samsung Notes", source="console")
    check("typed: it reads back and asks", "yes to go ahead" in rt.out[-1], rt.out[-1])
    rt.say("yes", source="console")
    check("...and on a yes it writes (the loop never calls the model)", ("new", "", "buy milk") in sn.d.log and not rt.ash.calls, str(sn.d.log))
    sn.d.log.clear()
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="write secret plans in Samsung Notes", addressed=True, salience=0.9,
                          urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice cannot write into your notes", not sn.pending and not [x for x in sn.d.log if isinstance(x, tuple)] and "only do that for" in rt.out[-1],
          rt.out[-1])
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="write the lecturer's joke in Samsung Notes", addressed=False,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("speech from the room (not meant for ASH) does not start a write", not sn.pending)
    rt2 = te.rig(notes=True, notes_path=os.path.join(tempfile.mkdtemp(), "notes.json"))
    rt2.samsung = rig()
    rt2.say("remember that Samsung Notes is my favorite note taking app and I keep my lectures on it", source="console")
    rt2.say("What can you do with Samsung Notes?", source="console")
    check("a remembered remark about Samsung Notes does not answer 'what can you do with it' (the notes rule used to come first)",
          "read your notes" in rt2.out[-1] and "You told me" not in rt2.out[-1], rt2.out[-1])
    ids = {e[0]: e[2] for e in manual.ENTRIES}
    check("the manual in core memory has it", "samsung" in ids and "new Samsung note" in ids["samsung"] and "Never into a PDF" in ids["samsung"], str(list(ids)))
    check("...and fits", len(ids["samsung"]) < 760, str(len(ids.get("samsung", ""))))


if __name__ == "__main__":
    s1_hearing()
    s2_reading()
    s2b_activity()
    s3_writing()
    s4_append()
    s5_dictation()
    s6_runtime()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
