"""
tst_settings.py -- one place for settings and status; a visible "waiting for you".

  S1  Reading a value: on/off, links, folders, one-line text; unknown keys refused
  S2  Saving: only the one key changes in daemon.json, atomically, nested keys too
  S3  Applying: live where the part runs, "after a restart" where it does not;
      the token is never sent back
  S4  Waiting for you: "Close Word?", a correction, a game, an offer, a
      question, focus -- most pressing first, and each one expires
  S5  Up to: reminders, watches, questions, follow-ups, look-ups, routines,
      what it is holding off on, recent corrections; a broken part costs only
      its own lines
  S6  The console: the commands, the status it broadcasts, the page

Run:  PYTHONIOENCODING=utf-8 python tst_settings.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.ui import settings as st  # noqa: E402

check = tw.check
T0 = tw.T0


def s1_parse():
    print("\n=== S1. Reading a value ===")
    for raw, want in (("on", True), ("TRUE", True), ("1", True), ("yes", True), (True, True),
                      ("off", False), ("false", False), ("0", False), ("no", False), (False, False)):
        check(f"bool {raw!r}", st.parse("play.enabled", raw) is want)
    for bad in ("maybe", "", "2"):
        try:
            st.parse("play.enabled", bad)
            check(f"bool {bad!r} refused", False)
        except ValueError:
            check(f"bool {bad!r} refused", True)
    check("links split on commas", st.parse("calendar.urls", " https://a/x.ics , webcal://b/y.ics ,")
          == ["https://a/x.ics", "webcal://b/y.ics"])
    check("empty list clears", st.parse("calendar.urls", "") == [])
    try:
        st.parse("calendar.urls", "https://a/x.ics, file:///etc/passwd")
        check("a non-link refused", False)
    except ValueError:
        check("a non-link refused", True)
    check("folders: any path", st.parse("wondering.lookup.roots", "~/Code, D:\\Notes") == ["~/Code", "D:\\Notes"])
    check("text trimmed", st.parse("weather.city", "  Amman ") == "Amman")
    for key, bad in (("weather.city", "a\nb"), ("smarthome.token", "x" * 500), ("smarthome.url", "homeassistant:8123"),
                     ("wondering.lookup.roots", "a\nb")):
        try:
            st.parse(key, bad)
            check(f"{key} refuses {bad[:12]!r}", False)
        except ValueError:
            check(f"{key} refuses {bad[:12]!r}", True)
    check("an empty address is allowed (turns it off)", st.parse("smarthome.url", "") == "")
    check("unknown key refused by change()", st.change(None, None, "log.level", "DEBUG")["ok"] is False)
    check("a section name alone is not a key", st.change(None, None, "play", "on")["ok"] is False)


def s2_save():
    print("\n=== S2. Saving ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "daemon.json")
    orig = {"play": {"enabled": True, "other": 5}, "log": {"level": "INFO"},
            "wondering": {"enabled": True, "lookup": {"web": True, "roots": ["~/Documents"]}}}
    json.dump(orig, open(p, "w", encoding="utf-8"))
    st.save(p, "play.enabled", False)
    got = json.load(open(p, encoding="utf-8"))
    check("the key changed", got["play"]["enabled"] is False)
    check("its neighbours kept", got["play"]["other"] == 5 and got["log"] == {"level": "INFO"})
    st.save(p, "wondering.lookup.roots", ["~/Code"])
    got = json.load(open(p, encoding="utf-8"))
    check("nested key", got["wondering"]["lookup"]["roots"] == ["~/Code"] and got["wondering"]["lookup"]["web"] is True
          and got["wondering"]["enabled"] is True)
    st.save(p, "weather.city", "Amman")
    check("a missing section is made", json.load(open(p, encoding="utf-8"))["weather"] == {"city": "Amman"})
    check("no temp files left", sorted(os.listdir(d)) == ["daemon.json"], str(os.listdir(d)))
    q = os.path.join(d, "fresh.json")
    st.save(q, "focus.enabled", False)
    check("no file yet: one is made", json.load(open(q, encoding="utf-8")) == {"focus": {"enabled": False}})
    open(p, "w").write("{ not json")
    try:
        st.save(p, "play.enabled", True)
        check("a broken daemon.json is not overwritten", False)
    except Exception:
        check("a broken daemon.json is not overwritten", open(p).read() == "{ not json")


def runtime(**over):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import PeripheryRegistry
    d = tempfile.mkdtemp()
    clock = [T0]
    cfg = dict(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, wonder=True, wonder_cfg={"min_gap_s": 0},
               wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json"),
               play=True, focus=True, reminders=True, reminders_path=os.path.join(d, "r.json"),
               desktop=True, weather=True, weather_cfg={"city": "Amman"}, calendar=True,
               smarthome=True, research=True, research_path=os.path.join(d, "rs.json"),
               watch=True, watch_path=os.path.join(d, "wt.json"), followups=True,
               followups_path=os.path.join(d, "f.json"), routines=True, routines_path=os.path.join(d, "ro.json"),
               moments=True)
    cfg.update(over)
    out = []
    rt = AmbientRuntime(ash=tw.Ash(), periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
                        gate=AttentionGate(), privacy=PrivacyPolicy(), config=AmbientConfig(**cfg),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.t, rt.out, rt.dir = clock, out, d
    if rt.weather is not None:
        rt.weather.background = False
        rt.weather.fetch = lambda url, timeout: (_ for _ in ()).throw(TimeoutError("offline"))
    if rt.calendar is not None:
        rt.calendar.background = False
    return rt


def s3_apply():
    print("\n=== S3. Applying ===")
    rt = runtime()                           # no remarks_dir: the remark log is off
    p = os.path.join(rt.dir, "daemon.json")
    r = st.change(rt, p, "wondering.enabled", "off")
    check("wondering off, live", r["ok"] and r["live"] and not rt.wonder.cfg.enabled and not r["restart"])
    check("and saved", json.load(open(p))["wondering"]["enabled"] is False and r["saved"])
    rt.wonder.off = True                     # "stop wondering" said out loud earlier
    st.change(rt, p, "wondering.enabled", "on")
    check("on again lifts a spoken 'stop wondering' too", rt.wonder.cfg.enabled and not rt.wonder.off)
    st.change(rt, p, "play.enabled", "off")
    check("a part with its own switch (play)", rt.play.enabled is False)
    st.change(rt, p, "focus.enabled", "off")
    check("focus", rt.focus.enabled is False)
    rt.weather._coords = (1.0, 2.0)
    rt.weather._fetched_at = T0
    st.change(rt, p, "weather.city", "Paris")
    check("a new city is looked up afresh", rt.weather.cfg.city == "Paris" and rt.weather._coords is None
          and rt.weather._fetched_at < 0)
    rt.calendar._texts = {"https://old/x.ics": "OLD", "/home/me/cal.ics": "FILE"}
    rt.calendar._fetched = {"https://old/x.ics": T0}
    st.change(rt, p, "calendar.urls", "https://new/y.ics")
    check("calendar links replaced, the old feed dropped, files kept",
          rt.calendar.cfg.urls == ("https://new/y.ics",) and "https://old/x.ics" not in rt.calendar._texts
          and "/home/me/cal.ics" in rt.calendar._texts and rt.calendar._fetched == {})
    st.change(rt, p, "smarthome.token", "SECRET-123")
    check("token set live", rt.smarthome.cfg.token == "SECRET-123")
    rows = {r["key"]: r for r in st.listing(rt, p)}
    check("the token is never sent back", rows["smarthome.token"]["value"] is True
          and "SECRET-123" not in json.dumps(st.listing(rt, p)))
    check("listing shows live values", rows["weather.city"]["value"] == "Paris" and rows["play.enabled"]["value"] is False
          and rows["calendar.urls"]["value"] == ["https://new/y.ics"])
    st.change(rt, p, "wondering.lookup.roots", "~/Code")
    check("lookup folders live", rt.lookup_engine.cfg.roots == ("~/Code",))
    check("saved nested", json.load(open(p))["wondering"]["lookup"]["roots"] == ["~/Code"])
    r = st.change(rt, p, "remarks.enabled", "on")
    check("a part not running: saved, needs a restart", r["ok"] and r["restart"] and not r["live"]
          and json.load(open(p))["remarks"]["enabled"] is True)
    rows = {r["key"]: r for r in st.listing(rt, p)}
    check("and listed from daemon.json", rows["remarks.enabled"]["value"] is True and not rows["remarks.enabled"]["live"])
    rt2 = runtime(watch=False)
    r = st.change(rt2, None, "watch.enabled", "on")
    check("no daemon.json path and not running: said so, not a silent success", r["ok"] and not r["saved"] and r["restart"])
    r = st.change(rt2, os.path.join(rt2.dir, "nope", "x", "daemon.json"), "watch.enabled", "on")
    check("cannot save and not live: an error", r["ok"] is False and "save" in r["error"])
    r = st.change(rt2, os.path.join(rt2.dir, "nope", "x", "daemon.json"), "play.enabled", "off")
    check("cannot save but live: applied, saved false", r["ok"] and r["live"] and not r["saved"])
    r = st.change(rt, p, "play.enabled", "sometimes")
    check("a bad value changes nothing", not r["ok"] and json.load(open(p))["play"]["enabled"] is False)


def s4_waiting():
    print("\n=== S4. Waiting for you ===")
    rt = runtime()
    now = rt.t[0]
    check("nothing pending: nothing shown", st.waiting(rt) is None)
    rt.focus.hear("I'm going to focus for 25 minutes", now)
    w = st.waiting(rt)
    check("focus", w and w["kind"] == "focus" and "Done focusing" in w["replies"], str(w))
    rt.wonder.observe("I told Omar about it", now=now)
    q = rt.wonder.next_question(tw.Environment(idle_s=60.0, free_moment=True), now)
    rt.wonder.mark_asked(q.id, now)
    w = st.waiting(rt)
    check("an asked question comes before focus", w and w["kind"] == "question" and "Omar" in w["text"], str(w))
    rt.play.offer("game", now)
    w = st.waiting(rt)
    check("an offered game", w["kind"] == "offer" and w["replies"] == ["Yes", "No thanks"], str(w))
    rt.play.start("")
    w = st.waiting(rt)
    check("a game in play", w["kind"] == "game" and w["replies"] == ["Stop"], str(w))
    rt._awaiting_meaning = {"query": "play the thing", "wrong": "music", "t": now}
    w = st.waiting(rt)
    check("a correction", w["kind"] == "correction" and "play the thing" in w["text"], str(w))
    rt.desktop._pending = ("Word", "winword.exe", now)
    w = st.waiting(rt)
    check("'Close Word?' first of all", w == {"kind": "confirm", "text": "Close Word?", "replies": ["Yes", "No"]}, str(w))
    rt.t[0] = now + rt.desktop.cfg.confirm_window_s + 1
    w = st.waiting(rt)
    check("the confirmation expires", w["kind"] != "confirm", str(w))
    rt.t[0] = now + rt.cfg.correction_window_s + 1
    check("so does the correction", st.waiting(rt)["kind"] != "correction")
    rt.play.game = None
    rt.t[0] = now + 121
    w = st.waiting(rt)
    check("and the offer", w is None or w["kind"] not in ("offer", "game"), str(w))
    rt.t[0] = now + 10 ** 6
    check("in the end, nothing", st.waiting(rt) is None, str(st.waiting(rt)))


def s5_up_to():
    print("\n=== S5. Up to ===")
    rt = runtime()
    now = rt.t[0]
    check("nothing yet", [g for g in st.up_to(rt) if g["title"] != "Recent corrections"] == [], str(st.up_to(rt)))
    for i, m in enumerate((10, 20, 30, 40)):
        rt.reminders.hear(f"remind me to call person{i} in {m} minutes", now)
    from src.daemon.watch import Watch
    rt.watcher.watches["w1"] = Watch("w1", "download", "the download", now)
    rt.wonder.observe("I told Omar about it", now=now)
    rt.followups.observe("I have a dentist appointment tomorrow", now=now)
    from src.daemon.research import Job
    rt.research.jobs["j1"] = Job("j1", "solar panels", now, status="ready")
    rt.research.jobs["j2"] = Job("j2", "old thing", now, status="told")
    rt.routines.seen["a > b"] = {"steps": ["Outlook", "Slack"], "days": ["d1", "d2", "d3"], "hours": [9], "asked": True}
    rt.routines.seen["c"] = {"steps": ["Paint"], "days": ["d1"], "hours": [9], "asked": False}
    for _ in range(3):
        rt.moments.backoff.note("mood_check", "rejected", now)
    g = {x["title"]: x for x in st.up_to(rt)}
    check("reminders, soonest first, three and a count", g["Reminders"]["items"][0].endswith("call person0")
          and len(g["Reminders"]["items"]) == 3 and g["Reminders"]["more"] == 1, str(g.get("Reminders")))
    check("watches", g["Watching"]["items"] == ["the download"])
    check("questions", any("Omar" in x for x in g["Wondering"]["items"]), str(g.get("Wondering")))
    check("follow-ups", "Will ask how it went" in g, str(list(g)))
    check("look-ups: not the ones already told", g["Looking up"]["items"] == ["solar panels (ready)"])
    check("routines seen enough", g["Routines noticed"]["items"] == ["Outlook > Slack"])
    check("holding off", g["Holding off on"]["items"] == ["mood check"])
    from src.brain.corrections import default
    default().record("play the thing", "music_play", "", now)
    g = {x["title"]: x for x in st.up_to(rt)}
    check("recent corrections", any("play the thing" in x and "music play" in x
                                    for x in g["Recent corrections"]["items"]), str(g.get("Recent corrections")))
    rt.watcher.active = lambda: 1 / 0
    g = {x["title"]: x for x in st.up_to(rt)}
    check("a broken part costs only its own lines", "Watching" not in g and "Reminders" in g and "Wondering" in g)
    rt2 = runtime(reminders=False, watch=False, research=False, followups=False, routines=False, wonder=False,
                  moments=False, corrections=False)
    check("nothing enabled: nothing listed", st.up_to(rt2) == [], str(st.up_to(rt2)))


def s6_console():
    print("\n=== S6. The console ===")
    from src.ui import UIBus, attach_ui
    from src.ui import hooks
    rt = runtime()
    p = os.path.join(rt.dir, "daemon.json")
    bus = UIBus()
    _, ctl, state = attach_ui(rt, bus, opener=None, status_every_s=10 ** 9, config_path=p)
    r = ctl("settings", "")
    check("settings listed", r["ok"] and any(s["key"] == "weather.city" for s in r["settings"]))
    r = ctl("setting", "weather.city=Paris")
    check("one changed through the console", r["ok"] and rt.weather.cfg.city == "Paris"
          and json.load(open(p))["weather"]["city"] == "Paris")
    sysmsg = [m for m in bus.history() if m.get("kind") == "system"]
    check("said in the chat", sysmsg and "Paris" in sysmsg[-1]["text"], str(sysmsg[-1:]))
    ctl("setting", "smarthome.token=TOPSECRET")
    check("the token is never in the chat", all("TOPSECRET" not in json.dumps(m) for m in bus.history()))
    check("nor in the state", "TOPSECRET" not in json.dumps(state()))
    check("no '=': refused", ctl("setting", "weather.city")["ok"] is False)
    check("not whitelisted: refused", ctl("setting", "log.file=/etc/passwd")["ok"] is False
          and "log" not in json.load(open(p)))
    s = hooks._status(rt)
    check("status carries waiting and up_to", "waiting" in s and "up_to" in s and s["waiting"] is None)
    rt.desktop._pending = ("Word", "winword.exe", rt.t[0])
    check("and shows the confirmation", hooks._status(rt)["waiting"]["text"] == "Close Word?")
    check("state has settings for the page", "settings" in state() and "status" in state())
    rt.wonder.awaiting = lambda now=None: 1 / 0
    rt.desktop._pending = None
    s = hooks._status(rt)
    check("a broken part never breaks the status", s["waiting"] is None and "sensors" in s)
    html = open(os.path.join("src", "ui", "static", "console.html"), encoding="utf-8").read()
    for ident in ('id="waiting"', 'id="statusBtn"', 'id="drawer"', 'function applyWaiting', 'cmd: "setting"',
                  'type = s.type === "secret" ? "password"'):
        check(f"page has {ident}", ident in html)


if __name__ == "__main__":
    for fn in (s1_parse, s2_save, s3_apply, s4_waiting, s5_up_to, s6_console):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
