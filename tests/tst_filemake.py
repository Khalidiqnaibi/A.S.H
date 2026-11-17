"""
tst_filemake.py -- "create a new empty desktop text file called notes.txt": done by rule, checked, never over an existing file.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_filemake.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import filemake  # noqa: E402

check = tw.check


def f1_parse():
    print("\n=== F1. What is understood ===")
    p = filemake.parse("Ash create a new empty desktop text file called livetest.txt")
    check("the live line", p == {"place": "desktop", "name": "livetest.txt", "ext": ""}, str(p))
    p = filemake.parse("make a text file called ideas in my documents")
    check("a kind and a folder", p and p["place"] == "documents" and p["name"] == "ideas" and p["ext"] == ".txt", str(p))
    p = filemake.parse("create a new file")
    check("no name: a default", p and p["name"] == "new file" and p["place"] == "desktop", str(p))
    check("a markdown file", filemake.parse("create a markdown file named todo")["ext"] == ".md")
    p = filemake.parse(r"create a file called ..\..\evil.txt")
    check("path characters are cleaned from a name", p is None or ("\\" not in p["name"] and "/" not in p["name"]), str(p))
    for t in ["I created a file yesterday", "where is the file", "can you read the file notes.txt", "delete the file notes.txt",
              "create a reminder for five"]:
        check(f"not a request: {t!r}", filemake.parse(t) is None)


def f2_make():
    print("\n=== F2. Making it ===")
    home = tempfile.mkdtemp()
    os.makedirs(os.path.join(home, "Desktop"))
    old = filemake._home
    filemake._home = lambda: home
    try:
        say = filemake.make(filemake.parse("create an empty desktop text file called a.txt"))
        f = os.path.join(home, "Desktop", "a.txt")
        check("it is there, empty, and it says so", os.path.isfile(f) and os.path.getsize(f) == 0 and "a.txt" in say and "Done" in say, say)
        say = filemake.make(filemake.parse("create an empty desktop text file called a.txt"))
        check("never over one that exists", "already" in say, say)
        say = filemake.make(filemake.parse("create a text file called b"))
        check("an extension is added", os.path.isfile(os.path.join(home, "Desktop", "b.txt")), say)
        say = filemake.make(filemake.parse("create a file in my documents"))
        check("a folder that does not exist falls back to the home folder", os.path.isfile(os.path.join(home, "new file.txt")), say)
        filemake.make(filemake.parse("create a text file called where1"))
        w = filemake.where("Where did you create the file? I don't see it.", __import__("time").time())
        check("where it was made is answered from what was made", w and w.endswith("where1.txt.") and "Desktop" in w, str(w))
        check("another question is not it", filemake.where("where is my phone", __import__("time").time()) is None)
        check("an hour later it is not remembered", filemake.where("where did you create the file", __import__("time").time() + 7200) is None)
        say = filemake.make(filemake.parse("create a file called c.txt"), write=lambda p, c, o: {"ok": False, "error": "disk full"})
        check("a failure is told, not claimed", "couldn't create" in say and "disk full" in say, say)
    finally:
        filemake._home = old


if __name__ == "__main__":
    f1_parse()
    f2_make()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
