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


if __name__ == "__main__":
    s1_idle()
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)
