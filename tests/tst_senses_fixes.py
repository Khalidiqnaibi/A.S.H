"""
tst_senses_fixes.py -- small things in src/senses/ that were wrong and are now pinned.

  S1  Windows idle time across the 24.8-day tick wrap (it was about -4 million seconds)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_senses_fixes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def s1_idle():
    print("\n=== S1. Idle time across the tick wrap ===")
    from src.senses.screen import idle_ms
    check("an ordinary gap", idle_ms(100_000, 90_000) == 10_000)
    check("GetTickCount read as a signed int (past 24.8 days of uptime) is still right",
          idle_ms(-2_000_000_000, 2_294_967_296 - 5_000) == 5_000, str(idle_ms(-2_000_000_000, 2_294_967_296 - 5_000)))
    check("the 49.7-day wrap", idle_ms(5, 4_294_967_290) == 11)
    check("never negative", idle_ms(0, 0) == 0 and idle_ms(3, 3) == 0)


def s2_name():
    print("\n=== S2. The name is stripped the way address.py reads it ===")
    from src.senses.mic import MicrophoneSensor as M
    for said, want in [("Hey, Ash, what time is it?", "what time is it?"), ("hi ash open Spotify", "open Spotify"),
                       ("Hello Ash, how are you", "how are you"), ("A.S.H., stop", "stop"), ("what time is it, ash", "what time is it"),
                       ("hey ash what's up", "what's up"), ("Ash.", "Ash."), ("the ash cloud is big", "the ash cloud is big"),
                       ("what do you think of Ashley", "what do you think of Ashley")]:
        got = M.strip_address(said)
        check(f"{said!r} -> {want!r}", got == want, repr(got))


def s3_third_person():
    print("\n=== S3. 'Ash' talked about, not to ===")
    from src.senses import address
    now = 1_800_000_000.0
    for said in ("I told Linda about ash.", "Ash is slow today.", "Ash never answers me."):
        v = address.score(said, now=now) if hasattr(address, "score") else None
        check(f"{said!r} is not certainly addressed (was 0.95)", v is not None and v.confidence < 0.6, str(getattr(v, "confidence", v)))
    for said in ("Ash, what time is it?", "what time is it, ash", "Hey Ash, open Chrome", "Ash is it raining?"):
        v = address.score(said, now=now)
        check(f"{said!r} still is", v.confidence >= 0.6, str(v.confidence))


def s4_titles():
    print("\n=== S4. A hidden window's title does not leak into the next event ===")
    from src.senses.screen import ActiveWindowSensor, set_private_content_protection
    ws = ActiveWindowSensor(enabled=False)
    seq = [("Bank of America - Opera", "opera.exe"), ("notes.txt - Notepad", "notepad.exe")]
    cur = [seq[0]]
    ws.current = lambda: cur[0]
    ws.blocklist = ["bank of america"]
    first = ws.read()
    cur[0] = seq[1]
    second = ws.read()
    txt = second[0].text if second else ""
    check("leaving a blocked window: the next event says '[redacted window]', not its title",
          "Bank of America" not in txt and "[redacted window]" in txt, txt)
    check("...and the first event never had it", first and "Bank of America" not in first[0].text and first[0].data["title"] is None)
    ws2 = ActiveWindowSensor(enabled=False)
    ws2.blocklist = []
    cur2 = [("Inbox (3) - Gmail", "chrome.exe")]
    ws2.current = lambda: cur2[0]
    set_private_content_protection(True)
    try:
        a = ws2.read()
        cur2[0] = ("todo.md - Notepad", "notepad.exe")
        b = ws2.read()
    finally:
        set_private_content_protection(False)
    check("with private-content protection on, a mailbox title is not in its own event", a and "Gmail" not in a[0].text and a[0].data["title"] is None,
          a[0].text if a else "")
    check("...nor in the one after it", b and "Gmail" not in b[0].text, b[0].text if b else "")
    ws3 = ActiveWindowSensor(enabled=False)
    ws3.blocklist = []
    cur3 = [("a.py - Code", "code.exe")]
    ws3.current = lambda: cur3[0]
    ws3.read()
    cur3[0] = ("b.py - Code", "code.exe")
    t = ws3.read()[0].text
    check("ordinary windows are still named", "a.py" in t and "b.py" in t, t)


if __name__ == "__main__":
    s1_idle()
    s4_titles()
    s2_name()
    s3_third_person()
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)
