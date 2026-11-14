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
