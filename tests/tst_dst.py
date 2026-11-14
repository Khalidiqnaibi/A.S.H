"""
tst_dst.py -- "tomorrow at 7" is seven o'clock on the wall, across a clock change too.

An alarm said after its time had passed was moved on with `due += 86400.0` -- twenty-four hours of seconds, which is an hour
wrong on the night the clocks change: "wake me at 7" said on the evening before spring-forward woke the person at 8.

The Windows C library cannot be told a different time zone from inside a process, so the test swaps in a datetime that reads
and writes Europe/Berlin (the clocks went forward there on 2026-03-29).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_dst.py
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import reminders as rem  # noqa: E402

check = tw.check
TZ = ZoneInfo("Europe/Berlin")


class Berlin(datetime):
    """A datetime whose 'local time' is Berlin, whatever this computer's is."""

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        d = datetime.fromtimestamp(ts, TZ)
        return cls(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) if tz is None else d

    def timestamp(self):
        return datetime(self.year, self.month, self.day, self.hour, self.minute, self.second, self.microsecond,
                        tzinfo=TZ).timestamp() if self.tzinfo is None else datetime.timestamp(self)


def local(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


def t1_spring_forward():
    print("\n=== The night the clocks go forward ===")
    real = rem.datetime
    rem.datetime = Berlin
    try:
        now = local(2026, 3, 28, 20, 0)                          # Saturday evening; Sunday 29 March loses an hour at 02:00
        r = rem.Reminders(clock=lambda: now)
        say = r.hear("wake me up at 7", now)
        due = sorted(x.due for x in r.items.values())[0]
        want = local(2026, 3, 29, 7, 0)
        check("an alarm at 7 said in the evening is 07:00 the next morning ON THE WALL", due == want,
              f"{datetime.fromtimestamp(due, TZ):%a %d %H:%M} vs {datetime.fromtimestamp(want, TZ):%a %d %H:%M}")
        check("...which is 10 real hours away (an hour is skipped overnight), not 11", abs((due - now) - 10 * 3600) < 1, str((due - now) / 3600))
        check("...and it says tomorrow, with the right hour", "7:00" in (say or "") or "07:00" in (say or ""), str(say))
    finally:
        rem.datetime = real


def t2_autumn_back():
    print("\n=== The night the clocks go back ===")
    real = rem.datetime
    rem.datetime = Berlin
    try:
        now = local(2026, 10, 24, 20, 0)                         # the clocks go back on Sunday 25 October
        r = rem.Reminders(clock=lambda: now)
        r.hear("wake me up at 7", now)
        due = sorted(x.due for x in r.items.values())[0]
        check("an alarm at 7 is 07:00 on the wall the morning the clocks go back too", due == local(2026, 10, 25, 7, 0),
              f"{datetime.fromtimestamp(due, TZ):%a %d %H:%M}")
        check("...which is 12 real hours away", abs((due - now) - 12 * 3600) < 1, str((due - now) / 3600))
    finally:
        rem.datetime = real


def t3_ordinary_day():
    print("\n=== An ordinary day ===")
    real = rem.datetime
    rem.datetime = Berlin
    try:
        now = local(2026, 6, 10, 20, 0)
        r = rem.Reminders(clock=lambda: now)
        r.hear("wake me up at 7", now)
        due = sorted(x.due for x in r.items.values())[0]
        check("an alarm at 7 in the evening is 07:00 tomorrow", due == local(2026, 6, 11, 7, 0))
        now2 = local(2026, 6, 10, 6, 0)
        r2 = rem.Reminders(clock=lambda: now2)
        r2.hear("wake me up at 7", now2)
        check("...and at 6 in the morning it is 07:00 today", sorted(x.due for x in r2.items.values())[0] == local(2026, 6, 10, 7, 0))
    finally:
        rem.datetime = real


if __name__ == "__main__":
    t1_spring_forward()
    t2_autumn_back()
    t3_ordinary_day()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
