"""
tst_calendar.py -- the user's calendar, from .ics files and feeds.

  C1  Parsing: one-off, all-day, UTC and local times, folded lines, escapes,
      cancelled events, and nonsense
  C2  Repeats: daily, weekly on given days, monthly, yearly, with COUNT,
      UNTIL, INTERVAL and EXDATE
  C3  Sources: a file re-read when it changes, a feed fetched on schedule,
      a bad feed ignored
  C4  Uses: "what's on my calendar today/tomorrow", the briefing part
  C5  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tst_calendar.py
"""

import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.calendar_ics import Calendar, CalendarConfig, parse_ics  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
DAY0 = datetime(2026, 3, 10)                 # a Tuesday
T0 = time.mktime((2026, 3, 10, 8, 0, 0, 0, 0, -1))


def ics(*events):
    return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n" + "".join(
        "BEGIN:VEVENT\r\n" + e.strip().replace("\n", "\r\n") + "\r\nEND:VEVENT\r\n" for e in events) + "END:VCALENDAR\r\n"


def on(text, d, days=1):
    return parse_ics(text, d, d + timedelta(days=days))


def c1_parsing():
    print("\n=== C1. Parsing ===")
    t = ics("DTSTART:20260310T100000\nDTEND:20260310T103000\nSUMMARY:Standup\nLOCATION:Room 4")
    e = on(t, DAY0)
    check("a one-off event", len(e) == 1 and e[0].summary == "Standup" and e[0].start == datetime(2026, 3, 10, 10)
          and e[0].end == datetime(2026, 3, 10, 10, 30) and e[0].location == "Room 4")
    check("not on another day", on(t, DAY0 + timedelta(days=1)) == [])
    e = on(ics("DTSTART;VALUE=DATE:20260310\nSUMMARY:Holiday"), DAY0)
    check("all day", len(e) == 1 and e[0].allday)
    utc = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    e = on(ics("DTSTART:20260310T120000Z\nSUMMARY:Call"), DAY0)
    check("UTC is shown in local time", len(e) == 1 and e[0].start == utc, str(e))
    e = on(ics("DTSTART;TZID=Asia/Amman:20260310T150000\nSUMMARY:Review"), DAY0)
    check("a named zone is taken as local", e and e[0].start.hour == 15)
    e = on(ics("DTSTART:20260310T090000\nSUMMARY:A very long\n  title, folded\\, with escapes"), DAY0)
    check("folded lines and escapes", e and e[0].summary == "A very long title, folded, with escapes", e[0].summary)
    check("cancelled is skipped", on(ics("DTSTART:20260310T090000\nSUMMARY:X\nSTATUS:CANCELLED"), DAY0) == [])
    check("no start is skipped", on(ics("SUMMARY:X"), DAY0) == [])
    check("nonsense is nothing", on("hello", DAY0) == [] and on("", DAY0) == []
          and on(ics("DTSTART:notadate\nSUMMARY:X"), DAY0) == [])
    e = on(ics("DTSTART:20260310T090000"), DAY0)
    check("no title", e and e[0].summary == "(untitled)")


def c2_repeats():
    print("\n=== C2. Repeats ===")
    daily = ics("DTSTART:20260309T090000\nSUMMARY:Gym\nRRULE:FREQ=DAILY;COUNT=5")
    check("daily, counted", [e.start.day for e in on(daily, datetime(2026, 3, 1), 30)] == [9, 10, 11, 12, 13])
    weekly = ics("DTSTART:20260302T100000\nSUMMARY:Standup\nRRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR")
    days = [e.start.day for e in on(weekly, datetime(2026, 3, 9), 7)]
    check("weekly on given days", days == [9, 11, 13], str(days))
    check("at the right time", all(e.start.hour == 10 for e in on(weekly, datetime(2026, 3, 9), 7)))
    check("today (a Tuesday) has none", on(weekly, DAY0) == [])
    late = ics("DTSTART:20260311T100000\nSUMMARY:Sync\nRRULE:FREQ=WEEKLY;BYDAY=MO,WE")
    check("a weekday before the first one is not an occurrence",
          [e.start.day for e in on(late, datetime(2026, 3, 9), 7)] == [11])
    every2 = ics("DTSTART:20260309T090000\nSUMMARY:Water plants\nRRULE:FREQ=DAILY;INTERVAL=2;COUNT=3")
    check("every other day", [e.start.day for e in on(every2, datetime(2026, 3, 1), 30)] == [9, 11, 13])
    biweek = ics("DTSTART:20260303T140000\nSUMMARY:1:1\nRRULE:FREQ=WEEKLY;INTERVAL=2")
    check("every other week", [e.start.day for e in on(biweek, datetime(2026, 3, 1), 31)] == [3, 17, 31])
    until = ics("DTSTART:20260309T090000\nSUMMARY:Sprint\nRRULE:FREQ=DAILY;UNTIL=20260311T235959")
    check("until", [e.start.day for e in on(until, datetime(2026, 3, 1), 30)] == [9, 10, 11])
    ex = ics("DTSTART:20260309T090000\nSUMMARY:Gym\nRRULE:FREQ=DAILY;COUNT=4\nEXDATE:20260310T090000")
    check("an excluded date", [e.start.day for e in on(ex, datetime(2026, 3, 1), 30)] == [9, 11, 12])
    monthly = ics("DTSTART:20260115T090000\nSUMMARY:Rent\nRRULE:FREQ=MONTHLY")
    check("monthly", [e.start.month for e in on(monthly, datetime(2026, 1, 1), 120)] == [1, 2, 3, 4])
    yearly = ics("DTSTART;VALUE=DATE:20200310\nSUMMARY:Birthday\nRRULE:FREQ=YEARLY")
    check("yearly, from years ago", len(on(yearly, DAY0)) == 1)
    forever = ics("DTSTART:19500101T090000\nSUMMARY:Old\nRRULE:FREQ=DAILY")
    t1 = time.time()
    got = on(forever, datetime(2030, 1, 1))
    check("a repeat from decades ago still shows today, quickly",
          len(got) == 1 and got[0].start == datetime(2030, 1, 1, 9) and time.time() - t1 < 1.0, str(got))
    wk = ics("DTSTART:20000103T090000\nSUMMARY:Weekly\nRRULE:FREQ=WEEKLY;INTERVAL=2")
    for q in (datetime(2030, 1, 1), datetime(2030, 1, 8), datetime(2031, 6, 2)):
        got = on(wk, q, 14)
        check(f"a fortnightly one lands on the right week ({q.date()})", len(got) == 1
              and (got[0].start - datetime(2000, 1, 3)).days % 14 == 0, str(got))


def c3_sources():
    print("\n=== C3. Sources ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cal.ics")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(ics("DTSTART:20260310T100000\nSUMMARY:Standup"))
    c = Calendar(CalendarConfig(files=(p,)), clock=lambda: T0, background=False)
    c.refresh()
    check("a file is read", [e.summary for e in c.events_on(DAY0.date())] == ["Standup"])
    c.refresh()
    check("not re-read when unchanged", c.stats["read"] == 1)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(ics("DTSTART:20260310T100000\nSUMMARY:Standup moved"))
    os.utime(p, (time.time() + 5, time.time() + 5))
    c.refresh()
    check("re-read when it changes", [e.summary for e in c.events_on(DAY0.date())] == ["Standup moved"])
    calls = []
    now = [T0]

    def fetch(url, timeout):
        calls.append(url)
        return ics("DTSTART:20260310T150000\nSUMMARY:Review")
    c = Calendar(CalendarConfig(urls=("https://example.test/cal.ics",)), fetch=fetch, clock=lambda: now[0],
                 background=False)
    c.refresh()
    check("a feed is fetched", calls and [e.summary for e in c.events_on(DAY0.date())] == ["Review"])
    c.refresh()
    check("not again before the refresh time", len(calls) == 1)
    now[0] += 1801
    c.refresh()
    check("again after it", len(calls) == 2)
    c = Calendar(CalendarConfig(urls=("https://x",)), fetch=lambda u, t: "<html>login</html>", background=False)
    c.refresh()
    check("a page that is not a calendar is ignored", c.events_on(DAY0.date()) == [] and c.stats["errors"] == 1)
    c = Calendar(CalendarConfig(urls=("https://x",)), fetch=lambda u, t: 1 / 0, background=False)
    c.refresh()
    check("a failing feed is not fatal", c.stats["errors"] == 1)
    c = Calendar(CalendarConfig(files=("~/no-such.ics",)), background=False)
    c.refresh()
    check("a missing file is not fatal", c.events_on(DAY0.date()) == [])
    both = Calendar(CalendarConfig(files=(p,), urls=("https://y",)),
                    fetch=lambda u, t: ics("DTSTART:20260310T100000\nSUMMARY:Standup moved"), background=False)
    both.refresh()
    check("the same event from two sources is listed once", len(both.events_on(DAY0.date())) == 1)


def c4_uses():
    print("\n=== C4. Uses ===")
    t = ics("DTSTART:20260310T100000\nSUMMARY:Standup", "DTSTART:20260310T143000\nSUMMARY:Design review",
            "DTSTART;VALUE=DATE:20260310\nSUMMARY:Sara's birthday", "DTSTART:20260311T090000\nSUMMARY:Dentist")
    c = Calendar(CalendarConfig(urls=("https://x",)), fetch=lambda u, tt: t, clock=lambda: T0, background=False)
    s = c.hear("what's on my calendar today?")
    check("today, all-day first, then by time", s == "Today: Sara's birthday, 10:00 Standup, 14:30 Design review.", s)
    check("tomorrow", c.hear("what's on my calendar tomorrow") == "Tomorrow: 09:00 Dentist.")
    check("'do I have any meetings tomorrow'", c.hear("do I have any meetings tomorrow") == "Tomorrow: 09:00 Dentist.")
    check("not a calendar question", c.hear("what's the weather today") is None)
    check("the morning briefing part", c.briefing_part("morning", T0).startswith("On your calendar today: Sara's"))
    check("the evening one is tomorrow", c.briefing_part("evening", T0) == "On your calendar tomorrow: 09:00 Dentist.")
    empty = Calendar(CalendarConfig(urls=("https://x",)), fetch=lambda u, tt: ics(), clock=lambda: T0,
                     background=False)
    check("nothing today: said so", empty.hear("what's on my calendar today") == "Nothing on your calendar today.")
    check("and no briefing part", empty.briefing_part("morning", T0) is None)
    check("no sources: questions left to the brain",
          Calendar(CalendarConfig()).hear("what's on my calendar today") is None)
    many = ics(*[f"DTSTART:20260310T{9 + i:02d}0000\nSUMMARY:M{i}" for i in range(9)])
    c = Calendar(CalendarConfig(urls=("https://x",)), fetch=lambda u, tt: many, clock=lambda: T0, background=False)
    check("a long day is cut short", c.hear("what's on my calendar today").endswith("and 3 more."))


def c5_runtime():
    print("\n=== C5. The runtime ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cal.ics")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(ics("DTSTART:20260310T100000\nSUMMARY:Standup"))
    clock = [T0]
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, calendar=True,
                             calendar_cfg={"files": [p]}, briefings=True,
                             briefings_path=os.path.join(d, "b.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="what's on my calendar today",
                          addressed=True, salience=0.9, urgency=0.6))
    rt._tick()
    check("answered without the model", [o[0] for o in out] == ["Today: 10:00 Standup."] and ash.calls == [],
          str(out))
    check("and in the morning briefing", "On your calendar today: 10:00 Standup." in rt.briefings.compose("morning"))
    check("status carries calendar", rt.status()["calendar"]["read"] == 1)


if __name__ == "__main__":
    for fn in (c1_parsing, c2_repeats, c3_sources, c4_uses, c5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
