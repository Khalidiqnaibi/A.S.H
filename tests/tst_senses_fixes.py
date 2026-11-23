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


if __name__ == "__main__":
    s1_idle()
    s2_name()
    s3_third_person()
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)
