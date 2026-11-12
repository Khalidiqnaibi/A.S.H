"""
tst_followups.py -- the fixes and features that came out of reviewing the browsing and profile work.

Each section is one change and was added with it; the file grows one commit at a time. Fixes first (the bugs a probe
confirmed), then the features. Nothing here touches the real machine: no real media key, no app launch, no network.

Run:  PYTHONIOENCODING=utf-8 python tst_followups.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
SECTIONS = []


def section(fn):
    SECTIONS.append(fn)
    return fn


@section
def fix_small_model_outage():
    print("\n=== FIX: one Ollama hiccup does not switch the small model off for the session ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ash_llm_fu", os.path.join(HERE, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    calls = []
    state = {"down": True, "missing": False}

    class R:
        status_code = 200
        text = '{"message": {"content": "ok"}}'

    class R404:
        status_code = 404
        text = 'model "tiny:1b" not found'

    def post(url, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "tiny:1b":
            if state["missing"]:
                return R404()
            if state["down"]:
                raise ConnectionError("Ollama is restarting")
        return R()
    L.requests.post = post
    llm = L.LLM(mode="ollama", ollama_model="big:7b")
    llm.set_light_model("tiny:1b")
    msg = [L.HumanMessage(content="hi")]
    with llm.route("light"):
        said = llm._call(msg)
    check("a refused connection: the main model answers", said == "ok" and calls == ["tiny:1b", "big:7b"], f"{said!r} {calls}")
    check("...and the small model stays on", llm._light_model == "tiny:1b", repr(llm._light_model))
    state["down"] = False
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("...so it is used again as soon as Ollama is back", calls == ["tiny:1b"], str(calls))
    state["missing"] = True
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("a model that is not installed is still switched off (404)", llm._light_model == "", repr(llm._light_model))
    calls.clear()
    with llm.route("light") as on:
        llm._call(msg)
    check("...and not tried again", calls == ["big:7b"] and not on, str(calls))


@section
def fix_exploration_in_new_activities():
    print("\n=== FIX: a new activity context is not explored harder than the situation it came from ===")
    from src.daemon.social import SocialLearner
    now = 1_800_000_000.0
    lr = SocialLearner(clock=lambda: now)
    for _ in range(30):
        lr.update("moment", "idle/evening", 1.0, 1.0, now)
    coarse = lr.exploration("moment", "idle/evening", 0.0, now)
    fine = lr.exploration("moment", "idle/evening/coding", 0.0, now)
    check("with 30 reactions in 'idle/evening', 'idle/evening/coding' is no keener to speak (it was 0.10 vs 0.02)",
          fine <= coarse + 1e-9, f"coarse {coarse:.3f} fine {fine:.3f}")
    fresh = SocialLearner(clock=lambda: now)
    check("a kind it has never seen at all is still explored, as before",
          abs(fresh.exploration("joke", "idle/evening/coding", 0.0, now) - fresh.explore) < 1e-9)
    for _ in range(20):
        lr.update("joke", "away/night/gaming", 1.0, 1.0, now)          # lots known about jokes while gaming
    check("what is known about the kind during that activity counts too",
          lr.exploration("joke", "idle/evening/gaming", 0.0, now) < 0.03, str(lr.exploration("joke", "idle/evening/gaming", 0.0, now)))
    check("without an activity nothing changed", abs(lr.exploration("moment", "idle/evening", 0.0, now) - coarse) < 1e-12)
    check("urgency still scales it down", lr.exploration("joke", "idle/evening/coding", 1.0, now) == 0.0)


@section
def fix_this_is_not_the_background_song():
    print("\n=== FIX: 'I hate this' about a bug is not a verdict on the song playing behind it ===")
    from tst_taste import kb_with
    from src.daemon.taste import Taste, TasteConfig, BIND_RECENT_S
    tk = Taste(kb_with(), user="Khalid", cfg=TasteConfig(), clock=lambda: 1000.0)

    def say(line, started_ago):
        tk.kb.tastes.clear()
        playing = {"item": "Midnight City by M83", "kind": "music", "at": 1000.0 - started_ago}
        return tk.observe(line, "console", 1000.0, playing=playing)
    check("a track that has only just started: a bare 'I hate this' is about it", say("I hate this.", 10) == ["The user hates Midnight City by M83."])
    check("...five minutes in, a bare 'I hate this.' is not (could be anything)", say("I hate this.", 300) == [])
    check("...nor 'this is so annoying'", say("this is so annoying", 300) == [])
    check("'this song' always means the song, however long it has been on", say("I hate this song", 300) == ["The user hates Midnight City by M83."])
    check("...and 'this track'", say("I love this track so much", 300) == ["The user loves Midnight City by M83."])
    check("a taste with no 'this' in it is unaffected", say("I love Interstellar", 300) == ["The user loves Interstellar."])
    check("the window is short and named", 0 < BIND_RECENT_S <= 60, str(BIND_RECENT_S))
    out = say("I hate this, it's so slow", 10)
    check("'it's' is never kept as something liked or hated", not any("it's" in s.lower() for s in out), str(out))
    check("...whatever the verb", all("it's" not in s.lower() for s in say("I can't stand this, it's so slow", 300)))

    # the runtime records when the track started, which is what the rule above reads
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    rt = te.rig()
    rt.t[0] = 5000.0
    rt.submit(SensorEvent(source="now_playing", modality=Modality.TEXT, text="now playing", salience=0.1, ts=5000.0,
                          persistable=False, meta=dict(event="started", player="Spotify", kind="music", title="Flowers",
                                                       artist="Miley Cyrus", item="Flowers by Miley Cyrus")))
    rt._tick()
    check("the runtime keeps when the track started", rt._playing and rt._playing.get("at") == 5000.0, str(rt._playing))


@section
def fix_never_record_list():
    print("\n=== FIX: chat, mail, cloud files, adult and patient sites are never recorded ===")
    from src.daemon import pagekind as pk
    c = pk.Classifier()
    chat = ["app.slack.com", "mycompany.slack.com", "teams.microsoft.com", "web.skype.com", "chat.google.com", "messenger.com",
            "web.whatsapp.com", "discord.com", "app.element.io"]
    check("chat apps (the desktop ones were already protected; the web ones were not)", all(pk.is_sensitive(h) for h in chat),
          str([h for h in chat if not pk.is_sensitive(h)]))
    mail = ["gmail.com", "outlook.com", "hotmail.com", "mail.proton.me", "mail.google.com", "mail.yahoo.com", "icloud.com"]
    check("mail", all(pk.is_sensitive(h) for h in mail), str([h for h in mail if not pk.is_sensitive(h)]))
    files = ["drive.google.com", "photos.google.com", "contacts.google.com", "keep.google.com", "onedrive.live.com", "takeout.google.com"]
    check("your files, photos and notes (these were being called 'search')", all(pk.is_sensitive(h) for h in files),
          str([h for h in files if not pk.is_sensitive(h)]))
    adult = ["onlyfans.com", "fansly.com", "chaturbate.com", "xvideos.com", "somehentaisite.net", "nsfw-art.com", "escort-guide.com"]
    check("adult sites, by name and by word", all(pk.is_sensitive(h) for h in adult), str([h for h in adult if not pk.is_sensitive(h)]))
    health = ["mychart.org", "mychart.example.com", "mypatientportal.org"]
    check("patient portals", all(pk.is_sensitive(h) for h in health), str([h for h in health if not pk.is_sensitive(h)]))
    check("a dating site by word", pk.is_sensitive("mydatingsite.com"))
    check("they are classified 'sensitive', never guessed", c.classify("app.slack.com") == ("unknown", "sensitive")
          and c.classify("drive.google.com") == ("unknown", "sensitive"))
    fine = ["github.com", "docs.google.com", "calendar.google.com", "maps.google.com", "colab.research.google.com", "history.com",
            "stackoverflow.com", "notion.so", "youtube.com", "wikipedia.org", "localhost"]
    check("ordinary sites are not swept up (docs, calendar, maps, colab are not Drive)", not any(pk.is_sensitive(h) for h in fine),
          str([h for h in fine if pk.is_sensitive(h)]))
    check("slack is no longer in the 'work' table", "slack.com" not in pk.TABLE)
    from src.senses.browsertabs import BrowserTabSensor
    s = BrowserTabSensor(enabled=True)
    check("and the sensor drops them on arrival", all(s.sanitize({"id": i, "url": f"https://{h}/x", "title": "x", "active": True}) is None
                                                      for i, h in enumerate(chat + mail + files)))


@section
def fix_forget_a_site_really_forgets():
    print("\n=== FIX: forgetting a site removes its time from the totals too ===")
    import sqlite3
    from src.daemon.ledger import Ledger, DAY
    now = 100 * DAY
    led = Ledger(clock=lambda: now, retention_days=30)
    led.record(now - 40 * DAY, now - 40 * DAY + 3600, "secret-hobby.com", "social", "user")
    led.record(now - 41 * DAY, now - 41 * DAY + 1800, "github.com", "code", "table")
    led.rollup(now)
    check("rolled up, the hour is in the social total", led.by_category(60).get("social") == 3600.0, str(led.by_category(60)))
    led.forget(site="secret-hobby.com")
    check("after 'forget everything about it', nothing of it is left in the totals", "social" not in led.by_category(60), str(led.by_category(60)))
    check("...nor in the per-site table", all(r[0] != "secret-hobby.com" for r in led.top_sites(20)))
    check("...nor in the daily rows", led.db.execute("SELECT COUNT(*) FROM daily_site WHERE site = 'secret-hobby.com'").fetchone()[0] == 0)
    check("other sites are untouched", led.by_category(60).get("code") == 1800.0, str(led.by_category(60)))
    led.forget(category="code")
    check("forgetting a category clears the daily rows too", led.by_category(60) == {}, str(led.by_category(60)))
    led.record(now - 50 * DAY, now - 50 * DAY + 600, "a.com", "news", "table")
    led.rollup(now)
    led.churn(3, now)
    led.forget_all()
    check("forget all leaves nothing, churn included", led.counts() == {"episodes": 0, "daily": 0, "sites": 0}, str(led.counts()))

    # a ledger written before daily_site existed: its totals keep, as "(older)"
    p = os.path.join(tempfile.mkdtemp(), "old.sqlite3")
    old = sqlite3.connect(p)
    old.executescript("CREATE TABLE daily (day TEXT NOT NULL, category TEXT NOT NULL, seconds REAL NOT NULL DEFAULT 0,"
                      " episodes INTEGER NOT NULL DEFAULT 0, churn INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (day, category));"
                      "INSERT INTO daily VALUES ('2026-01-05', 'video', 7200, 3, 0);"
                      "INSERT INTO daily VALUES ('2026-01-05', '_churn', 0, 0, 9);")
    old.commit()
    old.close()
    led = Ledger(p, clock=lambda: 1_800_000_000.0)
    check("an older ledger is migrated: its totals are kept as '(older)'", led.by_category(10000).get("video") == 7200.0, str(led.by_category(10000)))
    check("...and its churn is kept", led.churn_between(0, 1_800_000_000.0) == 9)
    led.forget(category="video")
    check("...and can still be forgotten", "video" not in led.by_category(10000))


@section
def fix_churn_means_a_tab_that_came_and_went():
    print("\n=== FIX: churn is a tab opened and closed within seconds -- not a glance, not a switch ===")
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor

    def tab(i, url, title="", active=False):
        return dict(id=i, url=url, title=title, active=active, audible=False, incognito=False)
    t = [1000.0]
    led = Ledger(clock=lambda: t[0])
    s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: t[0])

    def feed(tabs):
        s.submit({"focused": True, "tabs": tabs})
        return s.read()
    feed([tab(1, "https://github.com", "a", True)])
    t[0] = 1100
    feed([tab(1, "https://github.com", "a", True)])
    t[0] = 1101
    feed([tab(1, "https://github.com", "a"), tab(2, "https://news.ycombinator.com", "hn", True)])      # open, glance...
    t[0] = 1103
    feed([tab(1, "https://github.com", "a", True)])                                                       # ...close
    rows = led.db.execute("SELECT SUM(churn) FROM daily WHERE category='_churn'").fetchone()[0]
    check("a tab opened, glanced at and closed within seconds is ONE (it was counted twice)", s.stats["churn"] == 1 and rows == 1,
          f"{s.stats['churn']} {rows}")
    check("...and the glance is not recorded as a visit", all(e["site"] != "ycombinator.com" for e in led.episodes()))

    s2 = BrowserTabSensor(enabled=True, ledger=Ledger(clock=lambda: t[0]), clock=lambda: t[0])
    t[0] = 2000
    s2.submit({"focused": True, "tabs": [tab(1, "https://a.com", "a", True), tab(2, "https://b.com", "b")]})
    s2.read()
    for active in (2, 1, 2, 1):
        t[0] += 1
        s2.submit({"focused": True, "tabs": [tab(1, "https://a.com", "a", active == 1), tab(2, "https://b.com", "b", active == 2)]})
        s2.read()
    check("four quick switches between tabs that stay open are not churn (they were counted as 4)", s2.stats["churn"] == 0, str(s2.stats))

    s3 = BrowserTabSensor(enabled=True, clock=lambda: t[0])
    t[0] = 3000
    s3.submit({"focused": True, "tabs": [tab(1, "https://a.com", "a", True)]})
    s3.read()
    t[0] = 3001
    s3.submit({"focused": True, "tabs": [tab(1, "https://a.com", "a", True), tab(2, "https://b.com", "b")]})
    s3.read()
    t[0] = 3030
    s3.submit({"focused": True, "tabs": [tab(1, "https://a.com", "a", True)]})
    s3.read()
    check("a tab that was open for 29 seconds is not churn either (the line is ten)", s3.stats["churn"] == 0, str(s3.stats))
    from src.daemon.browsing import Browsing
    src = open(os.path.join(HERE, "src", "daemon", "browsing.py"), encoding="utf-8").read()
    check("and the answers say what is counted, not 'dropped tabs' or 'hopping between tabs'",
          "dropped tabs" not in src and "hopping between" not in src)


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
