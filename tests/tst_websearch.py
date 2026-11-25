"""
tst_websearch.py -- "look up cat pics" opens the search in the user's own browser.

It was answered "I can't browse the web" or by the model offering to discuss. Opening a search is something ASH can do, and the nearest thing to
"look it up": a URL through the desktop executor, nothing read or sent by ASH itself, the results in front of the person in their own browser.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_websearch.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import websearch  # noqa: E402

check = tw.check


def s1_parse():
    print("\n=== S1. What is asked ===")
    for t, want in [("Look up cat pics", ("images", "cat pics")), ("search for the best ramen near me", ("google", "the best ramen near me")),
                    ("search YouTube for lo-fi", ("youtube", "lo-fi")), ("google segue", ("google", "segue")), ("search on youtube for cats", ("youtube", "cats")),
                    ("search the web for python 3.13 release", ("google", "python 3.13 release")), ("look up cat pics on youtube", ("youtube", "cat pics")),
                    ("find me pictures of cats", ("images", "cats")), ("search wikipedia for segue", ("wikipedia", "segue")),
                    ("search for a good pizza place in amman please", ("google", "a good pizza place in amman")),
                    ("Look up a picture of a cat.", ("images", "cat")), ("Look up Google images for potatoes.", ("images", "potatoes")),
                    ("look up youtube for lo-fi", ("youtube", "lo-fi")), ("look up some photos of the sea", ("images", "sea")),
                    ("Look up potatoes.", ("google", "potatoes"))]:
        check(f"{t!r} -> {want}", websearch.parse(t) == want, str(websearch.parse(t)))
    check("the default engine can be moved", websearch.parse("google segue", "duckduckgo") == ("duckduckgo", "segue") or websearch.parse("search for segue", "duckduckgo") == ("duckduckgo", "segue"))
    for t in ("search my files for the report", "look up my notes", "find the report", "search memory for omar", "google it", "search for it", "find a file called x",
              "look up", "", None, "search for " + "x " * 60, "look up my schedule", "what is a search engine"):
        check(f"{str(t)[:40]!r} is not a web search", websearch.parse(t) is None)
    check("the URL carries only the query, encoded", websearch.url_for("images", "cat pics & dogs") == "https://www.google.com/search?tbm=isch&q=cat%20pics%20%26%20dogs")
    check("said back", websearch.say("youtube", "lo-fi") == "Searching YouTube for lo-fi.")


def s2_runtime():
    print("\n=== S2. In the runtime ===")
    import tst_modellast as tm
    import tst_desktop as td
    rt = tm.rig()
    rt.desktop = td.drig()
    used, said = rt.say("Look up cat pics")
    check("the browser is opened on the images, nothing else is done", not used and rt.desktop.ex_.opened == ["https://www.google.com/search?tbm=isch&q=cat%20pics"],
          f"{used} {rt.desktop.ex_.opened}")
    check("said back", said and said[-1] == "Searching Google Images for cat pics.", str(said))
    used, said = rt.say("search YouTube for lo-fi beats")
    check("YouTube", rt.desktop.ex_.opened[-1] == "https://www.youtube.com/results?search_query=lo-fi%20beats" and not used)
    n = len(rt.desktop.ex_.opened)
    rt.say("search my files for the report")
    check("'search my files' is not a web search", len(rt.desktop.ex_.opened) == n)
    rt.desktop.ex_.fail = True
    used, said = rt.say("google segue")
    check("a browser that will not open is said plainly", said and said[-1] == "I couldn't open the browser for that.", str(said))
    plain = tm.rig()
    used, said = plain.say("look up cat pics")
    check("with no desktop control it goes where it did", used or said, str(said))
    from src.daemon.desktop import Executor
    import inspect
    check("the real executor opens a page with startfile (cmd's 'start' splits a URL at '&')", "startswith((\"http://\", \"https://\"))" in inspect.getsource(Executor.open))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    s1_parse()
    s2_runtime()
