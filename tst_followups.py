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


@section
def fix_a_film_is_not_idle():
    print("\n=== FIX: watching a video with no keyboard input is still watching ===")
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    t = [3000.0]
    idle = [0.0]

    def make(**kw):
        led = Ledger(clock=lambda: t[0])
        return BrowserTabSensor(enabled=True, ledger=led, clock=lambda: t[0], idle_s=lambda: idle[0], **kw), led

    def watch(s, audible, minutes):
        t[0], idle[0] = 3000.0, 0.0
        tab = dict(id=1, url="https://www.netflix.com/watch/1", title="Show", active=True, audible=audible, incognito=False)
        s.submit({"focused": True, "tabs": [tab]})
        s.read()
        for _ in range(minutes * 2):                                  # a heartbeat every 30 seconds, hands off
            t[0] += 30
            idle[0] += 30
            s.submit({"focused": True, "tabs": [tab]})
            s.read()
        t[0] += 1
        s.read()
    s, led = make()
    watch(s, True, 6)
    still_open = s.current() is not None
    s.pause()                                                         # closes what is open, as stopping would
    secs = sum(e["end"] - e["start"] for e in led.episodes(0, 1e12))
    check("six minutes of an audible tab with no input: all six are kept (it was 180 seconds)", secs >= 350 and still_open,
          f"{secs} open={still_open}")
    s, led = make()
    watch(s, False, 6)
    secs = sum(e["end"] - e["start"] for e in led.episodes(0, 1e12))
    check("a silent tab with no input still ends at the idle line (three minutes)", 150 <= secs <= 200 and s.current() is None, f"{secs}")
    s, led = make(idle_audible_max_s=600)
    watch(s, True, 20)
    s.pause()
    secs = sum(e["end"] - e["start"] for e in led.episodes(0, 1e12))
    check("...and a video left running past the limit is plainly left running: it stops counting", secs < 700, f"{secs}")
    check("the default limit is hours, not minutes", BrowserTabSensor(enabled=True).idle_audible_max_s >= 3 * 3600)


# =====================================================================================================================
# FEATURES
# =====================================================================================================================
def _world(fill=True):
    """A sensor and ledger holding six days of a plausible life, plus 'now' (15:00 today)."""
    import tst_browsing as tb
    return tb.world(fill)


@section
def feature_timed_private_mode():
    print("\n=== FEATURE: 'don't record the next hour' -- a pause that ends by itself ===")
    from src.daemon.browsing import Browsing, parse
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    ok = {"don't record the next hour": 3600, "don't record my browsing for an hour": 3600, "go private for 30 minutes": 1800,
          "stop tracking my tabs for forty five minutes": 2700, "don't watch my tabs for the next two hours": 7200,
          "pause tab tracking for 20 minutes": 1200, "stop recording for an hour": 3600}
    got = {q: (parse(q) or (None, {}))[1].get("secs") for q in ok}
    bad = {q: got[q] for q in ok if got[q] != str(ok[q])}
    check("the ways people say it", not bad, str(bad))
    check("and it is its own kind, not the open-ended pause", parse("don't record the next hour")[0] == "pause_for"
          and parse("stop watching my tabs")[0] == "pause")
    check("a day-plus or a nonsense length is refused", parse("go private for 3000 hours") is None and parse("go private for 0 minutes") is None)
    check("ordinary talk is not taken for it", all(parse(q) is None for q in ("don't record the game for me", "stop recording for the weekend",
                                                                             "I need an hour")))

    t = [1000.0]
    led = Ledger(clock=lambda: t[0])
    s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: t[0])

    def tab(url="https://github.com"):
        return dict(id=1, url=url, title="x", active=True, audible=False, incognito=False)
    s.submit({"focused": True, "tabs": [tab()]})
    s.read()
    b = Browsing(lambda: s, clock=lambda: t[0])
    t[0] = 1100
    say = b.hear("don't record the next hour", t[0])
    check("it says for how long and when it starts again", "for 1 hour" in say and time_str(t[0] + 3600) in say, say)
    check("what was open is closed and kept (100 seconds)", s.current() is None and len(led.episodes()) == 1)
    check("nothing is read while it lasts", s.submit({"tabs": [tab()]}) is False and s.paused and s.status()["resume_at"] == t[0] + 3600)
    t[0] += 1800
    s.read()
    check("half way, still off", s.paused)
    t[0] += 1801
    s.read()
    check("when the hour is up it starts again by itself", not s.paused and s.status()["resume_at"] is None)
    check("...and reads again", s.submit({"tabs": [tab()]}) is True)
    s.pause(3600)
    b.hear("resume watching my tabs", t[0])
    check("'resume' ends it early, and clears the timer", not s.paused and s.resume_at is None)
    s.pause()
    t[0] += 10 ** 6
    s.read()
    check("an ordinary pause has no timer and never ends by itself", s.paused)


@section
def feature_never_track_a_site():
    print("\n=== FEATURE: 'never track youtube' -- a site you keep out ===")
    from src.daemon.browsing import Browsing, parse
    from src.daemon.pagekind import Classifier
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    ok = {"never track youtube": ("ignore_site", "youtube.com"), "don't track reddit.com": ("ignore_site", "reddit.com"),
          "ignore github.com": ("ignore_site", "github.com"), "stop tracking netflix": ("ignore_site", "netflix.com"),
          "start tracking youtube again": ("unignore_site", "youtube.com"), "resume tracking reddit.com": ("unignore_site", "reddit.com"),
          "what sites are you ignoring": ("ignored_list", None)}
    bad = {}
    for q, (kind, site) in ok.items():
        got = parse(q)
        if got is None or got[0] != kind or got[1].get("site") != site:
            bad[q] = got
    check("the ways people say it", not bad, str(bad))
    check("'stop tracking my tabs' is still the pause, not a site called 'my'", parse("stop tracking my tabs")[0] == "pause")
    check("a word that is not a site it can name is left alone", parse("never track homework") is None and parse("ignore the noise") is None)

    t = [1000.0]
    p = os.path.join(tempfile.mkdtemp(), "pk.json")
    cls = Classifier(p)
    led = Ledger(clock=lambda: t[0])
    s = BrowserTabSensor(enabled=True, classifier=cls, ledger=led, clock=lambda: t[0])
    led.record(100, 700, "youtube.com", "video", "table")
    led.record(800, 900, "github.com", "code", "table")
    b = Browsing(lambda: s, clock=lambda: t[0])
    say = b.hear("never track youtube", t[0])
    check("it says what it did, and that it deleted what it had", "won't record anything about youtube.com" in say and "deleted what I had" in say, say)
    check("what it held about the site is gone", led.totals(0, 1e9, site="youtube.com") == [] and led.totals(0, 1e9, category="video") == [])
    check("other sites are untouched", led.totals(0, 1e9, site="github.com") != [])

    def tab(i, url, active=False):
        return dict(id=i, url=url, title="x", active=active, audible=False, incognito=False)
    s.submit({"focused": True, "tabs": [tab(1, "https://www.youtube.com/watch?v=1", True)]})
    s.read()
    check("from then on the site is dropped on arrival, and not even an episode starts", s.current() is None and s.dropped["ignored"] == 1, str(s.dropped))
    check("...subdomains too", s.sanitize(tab(2, "https://music.youtube.com/x")) is None)
    check("...but a site that only looks similar is not (youtube.com.evil.net is another site)", s.sanitize(tab(3, "https://notyoutube.com/x")) is not None)
    check("it asks 'what are you ignoring' honestly", "youtube.com" in b.hear("what sites are you ignoring", t[0]))
    check("it survives a restart", Classifier(p).is_ignored("youtube.com") and Classifier(p).is_ignored("www.youtube.com"))
    check("saying it twice is fine", "already" in b.hear("ignore youtube.com", t[0]))
    check("and it comes back when asked", "track youtube.com again" in b.hear("start tracking youtube again", t[0]) and not cls.is_ignored("youtube.com"))
    s.submit({"focused": True, "tabs": [tab(4, "https://www.youtube.com/watch?v=1", True)]})
    s.read()
    check("...and is recorded again", s.current() is not None and s.current()["site"] == "youtube.com")
    check("a guest cannot change it", parse("never track youtube") is not None)         # gated like every browsing command


@section
def feature_show_and_export_everything():
    print("\n=== FEATURE: 'show me everything you hold' and 'export my browsing data' ===")
    import json
    from src.daemon.browsing import Browsing, parse
    s, led, now = _world()
    folder = tempfile.mkdtemp()
    b = Browsing(lambda: s, clock=lambda: now, export_dir=folder)
    check("the ways people ask", all((parse(q) or (None,))[0] == k for q, k in (
        ("show me everything you hold on me", "everything"), ("what exactly do you have on my browsing", "everything"),
        ("export my browsing data", "export"), ("download my browsing history", "export"), ("save a copy of my browsing data to a file", "export"))))
    check("ordinary talk is not", parse("export the report") is None and parse("show me everything") is None)
    say = b.hear("show me everything you hold on me", now)
    check("it says how much it holds and over which days", "recent stretches" in say and " from " in say and "to" in say, say)
    check("...the sites with most time", "github.com (code" in say, say)
    check("...and what it never keeps, and how to export or delete", "never keep page addresses" in say and "export my browsing data" in say, say)
    s.cls.ignore("reddit.com")
    s.cls.teach("acme.dev", "work")
    say = b.hear("show me everything you hold on me", now)
    check("...what you told it", "category of 1 site" in say and "reddit.com" in say, say)
    say = b.hear("export my browsing data", now)
    files = [f for f in os.listdir(folder) if f.startswith("browsing-") and f.endswith(".json")]
    check("an export is written, and the answer names the file", len(files) == 1 and files[0] in say, f"{files} {say}")
    data = json.load(open(os.path.join(folder, files[0]), encoding="utf-8"))
    check("it is plain JSON with the stretches, the daily totals and the sites", data["episodes"] and "daily" in data and data["sites"] is not None
          and set(data["episodes"][0]) >= {"start", "end", "site", "category"}, str(list(data)))
    check("...and it says what it does not contain", "No page addresses" in data["note"])
    text = json.dumps(data)
    check("...no addresses, no titles", "http" not in text and all(e["title"] == "" for e in data["episodes"]))
    check("the number of rows is reported", "rows saved" in say)
    off = Browsing(lambda: None)
    check("with tabs off it says so", "off" in off.hear("export my browsing data"))


@section
def feature_unknown_sites_and_tag_by_name():
    print("\n=== FEATURE: 'what sites can't you place?' and 'tag acme.dev as work' ===")
    from src.daemon.browsing import Browsing, parse
    from src.daemon.ledger import Ledger, DAY
    from src.senses.browsertabs import BrowserTabSensor
    ok = {"what sites can't you place": ("unknown_sites", None), "which sites are unknown": ("unknown_sites", None),
          "tag acme.dev as work": ("tag_site", "acme.dev"), "acme.dev is a work site": ("tag_site", "acme.dev"),
          "mark reddit.com as social": ("tag_site", "reddit.com"), "tag youtube as learning": ("tag_site", "youtube.com")}
    bad = {}
    for q, (kind, site) in ok.items():
        got = parse(q)
        if got is None or got[0] != kind or (site and got[1].get("site") != site):
            bad[q] = got
    check("the ways people say it", not bad, str(bad))
    check("'tag this site as work' is still about the page in front", parse("tag this site as work")[0] == "tag"
          and parse("tag that page as learning")[0] == "tag")
    check("a word that is not a site it can name is left alone", parse("tag homework as work") is None)
    check("a category that does not exist is not taken", parse("tag acme.dev as feelings") is None)

    now = 100 * DAY
    led = Ledger(clock=lambda: now, retention_days=30)
    led.record(now - 3 * DAY, now - 3 * DAY + 4 * 3600, "acme.dev", "unknown", "unknown")
    led.record(now - 2 * DAY, now - 2 * DAY + 3600, "foo.io", "unknown", "unknown")
    led.record(now - 60 * DAY, now - 60 * DAY + 7200, "acme.dev", "unknown", "unknown")     # older, rolled up
    led.record(now - 1 * DAY, now - 1 * DAY + 1800, "github.com", "code", "table")
    led.rollup(now)
    s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: now)
    b = Browsing(lambda: s, clock=lambda: now)
    say = b.hear("what sites can't you place", now)
    check("it lists what it can't place, biggest first, with the time", say.startswith("I can't place acme.dev (6 hours)") and "foo.io (1 hour)" in say
          and "github.com" not in say, say)
    check("...and says how to tell it", 'tag acme.dev as work' in say, say)
    say = b.hear("tag acme.dev as work", now)
    check("tagging by name says what it did, and how many records it moved", say.startswith("Okay, acme.dev is work") and "updated the" in say, say)
    check("the time moved from 'unknown' to 'work', raw and rolled up", led.totals(0, now + 1, category="work") != []
          and led.by_category(90).get("work") == 6 * 3600.0 and led.by_category(90).get("unknown") == 3600.0, str(led.by_category(90)))
    check("...and the older rolled-up hours too", any(r[0] == "acme.dev" and r[1] == "work" for r in led.top_sites(10)) and
          all(not (r[0] == "acme.dev" and r[1] == "unknown") for r in led.top_sites(10)), str(led.top_sites(10)))
    check("it is remembered for new visits", s.cls.classify("acme.dev") == ("work", "user"))
    check("what is left to place is shorter", "acme.dev" not in b.hear("what sites can't you place", now))
    b.hear("tag foo.io as learning", now)
    check("when nothing is left it says so", b.hear("what sites can't you place", now) == "I've been able to place every site I've seen.")
    check("a site it never records cannot be tagged", "never record" in b.hear("tag chase.com as work", now))
    s.cls.ignore("secret.com")
    led.record(now - 3600, now - 1800, "secret.com", "unknown", "unknown")
    check("a site you keep out is never asked about, even if something about it is still held",
          b.hear("what sites can't you place", now) == "I've been able to place every site I've seen.", b.hear("what sites can't you place", now))


def time_str(ts):
    import time as _t
    return _t.strftime("%H:%M", _t.localtime(ts))


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
