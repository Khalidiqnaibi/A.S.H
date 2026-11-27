"""tst_recall.py -- Windows notifications (off unless switched on; only the app and who from), "pull up everything on X", "find that page about X
from yesterday", limits the person sets, the offer to watch a download, and handing a slow question to Gemini. Fakes only: no notification
database, journal, browser or download folder of this machine is read.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_recall.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon import recall, toasts  # noqa: E402
from src.senses.browsertabs import BrowserTabSensor  # noqa: E402

check = tw.check
WA = "<toast><visual><binding template='ToastGeneric'><text>Omar</text><text>are you coming tonight? the secret is 1234</text></binding></visual></toast>"
DC = "<toast><visual><binding><text>Lieth</text><text>gg</text></binding></visual></toast>"


def main():
    print("\n=== N1. Notifications: the app and who from, nothing else ===")
    check("WhatsApp's id", toasts.app_name("5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App") == "WhatsApp")
    check("Windows itself is not news", toasts.app_name("Windows.SystemToast.SecurityAndMaintenance") == "")
    check("ASH's own toasts are not news", toasts.app_name("C:\\Python311\\python.exe") == "")
    check("only the first line is read", toasts.first_line(WA) == "Omar")
    now = [time.time()]
    rows = [(toasts.to_filetime(now[0] + 5), "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App", WA),
            (toasts.to_filetime(now[0] + 6), "com.squirrel.Discord.Discord", DC)]
    t = toasts.Toasts(enabled=False, reader=lambda after: [r for r in rows if r[0] > after], clock=lambda: now[0],
                      who=lambda n: "your cousin" if n == "Omar" else "")
    check("off: nothing read", t.poll() == [] and t.stats["read"] == 0)
    check("off, and asked: says how to turn it on", "off" in t.hear("any notifications?"))
    t.enabled = True
    new = t.poll()
    check("on: two, as app and sender", [(a, f) for _t, a, f in new] == [("WhatsApp", "Omar"), ("Discord", "Lieth")], str(new))
    line = t.line(new)
    check("said with who they are, never the message", line == "2 notifications: WhatsApp from Omar, your cousin and Discord from Lieth."
          and "secret" not in line and "1234" not in line, line)
    check("one alone", t.line(new[:1]) == "A message on WhatsApp from Omar, your cousin.", t.line(new[:1]))
    check("not twice", t.poll() == [])
    check("'any notifications?' says the last hour's", t.hear("any notifications?").startswith("In the last hour: 2 notifications"), t.hear("any notifications?"))
    rt = tm.rig()
    rt.toasts = toasts.Toasts(enabled=True, reader=lambda after: [r for r in rows if r[0] > after], clock=time.time,
                              who=rt._relation_of)
    rt.toasts.since = 0
    rt.toasts.poll()
    from src.daemon import away
    got = away.items(rt, time.time() - 600, time.time() + 600)
    check("'what did I miss?' lists them", any("whatsapp from omar" in g.lower() for g in got), str(got))
    from src.ui import settings as st
    rt2 = tm.rig()
    check("off by default in the runtime", rt2.toasts.enabled is False)
    st.apply(rt2, "notifications.enabled", True)
    check("the settings switch turns it on, live", rt2.toasts.enabled is True)

    print("\n=== N2. Pull up everything on X ===")
    for q, want in (("pull up everything on NoSQL", "NoSQL"), ("what do you have on Omar", "Omar"), ("Ash, everything you know about the Raft algorithm", "Raft algorithm"),
                    ("show me everything you have on me", None)):
        check(f"{q!r} -> {want!r}", recall.dossier_subject(q) == want, str(recall.dossier_subject(q)))
    rt = tm.rig()
    rt.wonder.kb.people["omar"] = {"name": "Omar", "relation": "cousin"}
    rt.say("remember that Omar owes me 20 dollars")
    used, said = rt.say("pull up everything on Omar")
    check("who, and the note -- one answer, no model", not used and said and said[-1].startswith("On Omar: Omar is your cousin;") and "owes you 20 dollars" in said[-1],
          f"{used} {said}")
    used, said = rt.say("pull up everything on Zanzibar")
    check("nothing: said so", said and said[-1].startswith("I don't have anything on Zanzibar"), str(said))

    print("\n=== N3. The page from yesterday ===")
    yday = time.time() - 86400
    day = time.strftime("%Y-%m-%d", time.localtime(yday))

    class J:
        def read_day(self, d):
            return [{"ts": yday, "source": "active_window", "meta": {"subject": "Liquid Biopsy Basics", "project": "Google Gemini", "process": "opera.exe"}},
                    {"ts": yday + 60, "source": "active_window", "meta": {"subject": "My bank", "private": True}}] if d == day else []

        def flush(self):
            pass

        def append(self, ev):
            pass
    check("parsed", recall.page_query("find that page about liquid biopsy from yesterday") == ("liquid biopsy", "yesterday"))
    hit = recall.find_page(J(), "liquid biopsy", "yesterday", time.time())
    check("found by its words, on the day asked", hit and hit["title"] == "Liquid Biopsy Basics", str(hit))
    check("a private window is never found", recall.find_page(J(), "bank", "", time.time()) is None)
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.journal = J()
    used, said = rt.say("find that page about liquid biopsy from yesterday")
    check("not open: offered as a search", said and said[-1].startswith("\u201cLiquid Biopsy Basics\u201d in Google Gemini, yesterday at") and "search for it" in said[-1], str(said))
    rt.say("yes")
    check("a yes searches its title", rt.desktop.ex_.opened and "Liquid+Biopsy+Basics" in rt.desktop.ex_.opened[-1], str(rt.desktop.ex_.opened))
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.add("opera.exe")
    rt.journal = J()
    bt = BrowserTabSensor(enabled=True)
    bt.submit({"focused": True, "tabs": [{"id": 4, "url": "https://gemini.google.com/app/1", "title": "Liquid Biopsy Basics - Google Gemini", "active": False,
                                          "audible": False, "incognito": False}]})
    bt.read()
    real = rt.periphery.sensors

    class S(dict):
        def get(self, k, d=None):
            return bt if k == "browser_tabs" else real.get(k, d)
    rt.periphery.sensors = S(real)
    used, said = rt.say("find that page about liquid biopsy from yesterday")
    check("still open: offered", said and "still open" in said[-1], str(said))
    rt.say("open it")
    check("'open it' brings the tab forward", bt.take_command() == {"activate": 4}, "")

    print("\n=== N4. Limits ===")
    check("parsed", recall.parse_limit("tell me after 30 minutes of YouTube") == ("YouTube", 1800.0) and recall.parse_limit("warn me if I spend more than an hour on reddit") == ("reddit", 3600.0))
    rt = tm.rig()
    spent = [100.0]
    real_on = recall.seconds_on
    recall.seconds_on = lambda r, what, now: spent[0]
    try:
        out = []
        sink = rt.response_sink
        rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
        used, said = rt.say("tell me after 30 minutes of YouTube")
        check("set, counting from now", said and said[-1] == "Okay -- I'll tell you after 30 minutes of YouTube, counting from now.", str(said))
        spent[0] = 100.0 + 1000
        rt._limits_at = 0
        rt._limit_step(rt.clock())
        check("not yet", not any("limit you asked" in o for o in out))
        spent[0] = 100.0 + 1801
        rt._limits_at = 0
        rt._limit_step(rt.clock())
        check("reached: said once", sum("That's 30 minutes of YouTube" in o for o in out) == 1, str(out[-2:]))
        rt._limits_at = 0
        rt._limit_step(rt.clock())
        check("...and not again", sum("That's 30 minutes of YouTube" in o for o in out) == 1)
    finally:
        recall.seconds_on = real_on

    print("\n=== N5. A download, and Gemini when it is slow ===")
    rt = tm.rig()

    from src.daemon.watch import Probe as _RealProbe

    class Probe(_RealProbe):
        def __init__(self):
            super().__init__()
            self.files = {"big.iso.crdownload": (100, 0.0)}

        def listdir(self, p):
            return dict(self.files)
    from src.daemon.watch import Watcher, WatchConfig
    rt.watcher = Watcher(WatchConfig.from_dict({}), probe=Probe(), clock=lambda: rt.clock()) if "probe" in Watcher.__init__.__code__.co_varnames else None
    if rt.watcher is not None:
        offers = []
        real_submit = rt.submit
        rt.submit = lambda ev: (offers.append(ev.text), real_submit(ev))[1]
        t0 = rt.clock()
        rt._download_offer_step(t0)
        rt._download_offer_step(t0 + 61)
        check("seen a minute: not yet", offers == [], str(offers))
        rt._download_offer_step(t0 + 200)
        check("going two minutes: offered once", offers == ["That download's still going -- want me to tell you when it's done?"], str(offers))
        rt._download_offer_step(t0 + 400)
        check("...not twice", len(offers) == 1)
        rt.submit = real_submit
        used, said = rt.say("yes")
        check("a yes sets the watch", any(x.kind == "download" for x in rt.watcher.active()), str(said))
    else:
        check("the watcher takes a probe", False)
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.add("opera.exe")
    rt._thinking, rt._think_query = 1, "can my pc run gta 6"
    clip = []
    real_a = rt.periphery.actuators

    class A(dict):
        def get(self, k, d=None):
            return types.SimpleNamespace(act=lambda x: (clip.append(x), {"ok": True})[1]) if k == "clipboard" else real_a.get(k, d)
    rt.periphery.actuators = A(real_a)
    used, said = rt.say("ask Gemini")
    check("a bare 'ask Gemini' hands over the question in progress", clip == ["Can my pc run gta 6?"] and rt._cancelled and said
          and said[-1].startswith("Opening Gemini"), f"{clip} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
