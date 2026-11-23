"""
tst_retention.py -- retention of 0 days means "do not keep", not "keep forever".

Run:  PYTHONIOENCODING=utf-8 python tests/tst_retention.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.daemon.privacy import PrivacyPolicy  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def files(d, ages_h):
    now = time.time()
    for name, h in ages_h.items():
        p = os.path.join(d, name)
        open(p, "w").write("x")
        os.utime(p, (now - h * 3600, now - h * 3600))


def main():
    print("\n=== P1. Retention in days ===")
    j, f = tempfile.mkdtemp(), tempfile.mkdtemp()
    files(j, {"old.jsonl": 24 * 10, "week.jsonl": 24 * 3, "now.jsonl": 0})
    files(f, {"old.jpg": 24 * 3, "now.jpg": 0})
    r = PrivacyPolicy(raw_journal_days=7, frame_days=2).sweep(j, f)
    check("7 days: the 10-day-old journal and the 3-day-old frame go, the rest stay",
          sorted(os.listdir(j)) == ["now.jsonl", "week.jsonl"] and os.listdir(f) == ["now.jpg"], str(r))
    j, f = tempfile.mkdtemp(), tempfile.mkdtemp()
    files(j, {"day1.jsonl": 30, "day2.jsonl": 5, "now.jsonl": 0.1})
    files(f, {"a.jpg": 5, "b.jpg": 0.1})
    PrivacyPolicy(raw_journal_days=0, frame_days=0).sweep(j, f)
    check("0 days: everything older than an hour goes (it used to keep it all)",
          os.listdir(j) == ["now.jsonl"] and os.listdir(f) == ["b.jpg"], f"{os.listdir(j)} {os.listdir(f)}")
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
