"""
tst_errorwords.py -- failures are told in words, not in class names.

ASH said "I couldn't check your email just now (OSError)." and "It didn't send (SMTPServerDisconnected).", "That didn't
work: unknown error", "I stopped the plan before 'get_weather'" and "...was blocked by safety_board". Those are the log's
words. JARVIS would say what happened, in a few plain ones; the class names stay in the log.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_errorwords.py
"""

import os
import re
import smtplib
import socket
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check


def t1_mail():
    print("\n=== Mail ===")
    from src.daemon.inbox import why_it_failed
    for exc, want in [(OSError("x"), "the mail server wouldn't answer"), (ConnectionRefusedError(), "the mail server wouldn't answer"),
                      (socket.timeout(), "it took too long to answer"), (TimeoutError(), "it took too long to answer"),
                      (EOFError(), "the mail server wouldn't answer"), (KeyError("k"), "something went wrong")]:
        got = why_it_failed(exc)
        check(f"{type(exc).__name__} -> {want}", got == want, got)
        check(f"...no class name in it", type(exc).__name__ not in got)

    import tst_inbox as ti
    from src.daemon import inbox
    d = ti.make() if hasattr(ti, "make") else None
    if d is not None:
        d.last_error = why_it_failed(OSError())
        d.cfg.enabled = True
        say = d.hear("any new email?")
        check("the email answer carries the words", say and "OSError" not in say and "mail server" in say, str(say))

    from src.daemon.drafts import Drafts
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "daemon", "drafts.py"),
               encoding="utf-8").read()
    check("a failed send does not read a class name aloud", "type(e).__name__" not in src)
    check("...and says the draft is still there", "The draft is still here" in src)


def t2_plan_and_tools():
    print("\n=== Plans and tools ===")
    from src.brain import brain as b
    check("'unknown error' is gone", "unknown error" not in open(b.__file__, encoding="utf-8").read())
    out = b._fmt_tool_output({"ok": False}) if hasattr(b, "_fmt_tool_output") else None
    if out is not None:
        check("a failure with no reason is still plain", out == "That didn't work out.", out)
        check("one with a reason keeps it", b._fmt_tool_output({"ok": False, "error": "boom"}) == "That didn't work: boom")
        said = b._fmt_tool_output({"ok": True, "cpu_percent": 27.5, "ram_used_percent": 82.7, "ram_used_mb": 13391, "ram_total_mb": 16202,
                                   "uptime": "8:56:35", "cpu_temp_c": None})
        check("the machine's numbers are said as a sentence, not a dump of keys",
              said == "CPU at 28%, memory at 83% (13.1 of 16 GB); on for 8 h 56 min." and "_percent" not in said and "None" not in said, said)
        check("an empty reading is left out of a plain template", b._fmt_tool_output({"ok": True, "a": 1, "b": None}) == "a: 1")
        f = b._fmt_tool_output
        s = f({"ok": True, "operation": "list_directory", "path": "src", "entry_count": 7,
               "entries": [{"name": n, "size_bytes": 10} for n in "abcdefg"], "truncated": False})
        check("a folder listing is said as names, not a dump", s == "7 items in there: a, b, c, d, e, and 2 more.", s)
        check("an empty folder", f({"ok": True, "operation": "list_directory", "entry_count": 0, "entries": []}) == "That folder is empty.")
        s = f({"ok": True, "words": 1, "characters": 5, "estimated_reading_time_sec": 0})
        check("a word count is a sentence", s == "1 word, 5 characters.", s)
        check("a conversion keeps its unit", f({"ok": True, "result": 6.2137, "unit": "mi"}) == "That's 6.2137 mi.")
        check("a plain result is still just the result", f({"ok": True, "result": 84}) == "84")
        s = f({"ok": True, "operation": "search_files", "match_count": 2, "matches": [{"file": "C:/x/a.py"}, {"file": "C:/x/b.py"}]})
        check("a search says where", s == "2 matches, in a.py, b.py.", s)
    text = open(b.__file__, encoding="utf-8").read()
    check("a paused plan says 'paused', not the intent's id in quotes", "I've paused the plan before" in text
          and "plan before '{step.intent}'" not in text)
    check("...and a veto names 'the safety check', not 'safety_board'", "the safety check" in text)
    from src.brain.gist import readable
    check("an intent id reads as words", readable("get_weather") == "get weather")


if __name__ == "__main__":
    t1_mail()
    t2_plan_and_tools()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
