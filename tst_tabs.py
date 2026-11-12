"""
tst_tabs.py -- seeing what is open in the browser, with no language model.

  T1  pagekind: table, keywords, what the user teaches, honest "unknown", never-record sites
  T2  The sensor drops what it must (private windows, banks, mail, health) and keeps hosts only
  T3  Episodes: dwell, switch, churn, focus lost, extension silent, idle, paused
  T4  The ledger: record, totals, titles off, rollup, forget
  T5  The loopback listener: token, origin, size, 127.0.0.1 only
  T6  No LLM and no network anywhere in it -- enforced, not promised

Run:  PYTHONIOENCODING=utf-8 python tst_tabs.py
"""

import ast
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
HERE = os.path.dirname(os.path.abspath(__file__))


def tab(i, url, title="", active=False, audible=False, incognito=False):
    return dict(id=i, url=url, title=title, active=active, audible=audible, incognito=incognito)


def t1_pagekind():
    print("\n=== T1. What kind of page ===")
    from src.daemon import pagekind as pk
    c = pk.Classifier()
    want = {("github.com", ""): ("code", "table"), ("www.youtube.com", "x"): ("video", "table"),
            ("docs.google.com", ""): ("docs", "table"), ("google.com", ""): ("search", "table"),
            ("en.wikipedia.org", ""): ("reference", "table"), ("localhost", ""): ("code", "table"),
            ("unknownsite.xyz", "Python tutorial for beginners"): ("learning", "keyword"),
            ("unknownsite.xyz", "Weekly sync"): ("unknown", "unknown")}
    got = {k: c.classify(*k) for k in want}
    check("the table and the keywords", got == want, str({k: v for k, v in got.items() if v != want[k]}))
    check("the most specific host wins (docs.google.com is not search)", c.classify("docs.google.com")[0] == "docs"
          and c.classify("maps.google.com")[0] == "travel")
    check("host_of drops scheme, www, port, path, query, userinfo",
          pk.host_of("https://user:pw@www.Example.com:8080/a/b?q=1#x") == "example.com")
    check("registrable: a.b.example.co.uk -> example.co.uk", pk.registrable("a.b.example.co.uk") == "example.co.uk")

    sens = ["mail.google.com", "chase.com", "www.paypal.com", "webmd.com", "discord.com", "bitwarden.com",
            "192.168.1.1", "router", "nas.local", "myonlinebank.com", "somepornsite.com", "accounts.google.com",
            "web.whatsapp.com", "login.example.com"]
    check("never-record hosts", all(pk.is_sensitive(h) for h in sens), str([h for h in sens if not pk.is_sensitive(h)]))
    check("...as a classification they are 'sensitive', never guessed", c.classify("chase.com") == ("unknown", "sensitive"))
    fine = ["github.com", "history.com", "tortoise.org", "docs.google.com", "127.0.0.1", "localhost", "bbc.co.uk",
            "stackoverflow.com", "notion.so"]
    check("...and ordinary sites are not swept up (history.com is not 'tor')", not any(pk.is_sensitive(h) for h in fine),
          str([h for h in fine if pk.is_sensitive(h)]))
    check("a title can make a page private", pk.is_sensitive("example.com", "Symptoms of anxiety - Example")
          and pk.is_sensitive("example.com", "Your recovery key"))
    check("...an ordinary title does not", not pk.is_sensitive("example.com", "How to bake bread"))

    # teaching: the user's word, then the words of the title
    p = os.path.join(tempfile.mkdtemp(), "pk.json")
    c = pk.Classifier(p)
    check("unknown to begin with", c.classify("acme-internal.dev", "Sprint board - Acme")[1] == "keyword"
          or c.classify("acme-internal.dev", "Quarterly numbers")[0] == "unknown")
    check("'that's work' teaches the site", c.teach("https://acme-internal.dev/x", "work", "Quarterly numbers") is True
          and c.classify("acme-internal.dev", "anything") == ("work", "user"))
    check("...the user's word beats the table", (c.teach("github.com", "work") and c.classify("github.com")[0] == "work"))
    c.untag("github.com")
    check("...and can be undone", c.classify("github.com")[0] == "code")
    check("a sensitive site cannot be taught or stored", c.teach("chase.com", "work") is False)
    check("a made-up category is refused", c.teach("example.org", "feelings") is False)
    # three examples of a category teach it its words
    c2 = pk.Classifier()
    for i, t in enumerate(("Budget forecast variance", "Budget variance forecast notes", "Forecast budget deck")):
        c2.teach(f"corp{i}.example", "work", t)
    check("learned: a page with the same words, from a site it has never seen",
          c2.classify("brandnew.example", "Budget forecast for next year") == ("work", "learned"),
          str(c2.classify("brandnew.example", "Budget forecast for next year")))
    check("learned: one shared word is not evidence", c2.classify("brandnew.example", "Budget for a trip")[1] != "learned")
    c3 = pk.Classifier()
    c3.teach("a.example", "work", "Budget forecast")
    check("learned: too few examples says nothing", c3.classify("b.example", "Budget forecast review")[1] == "unknown")
    c.save()
    check("it is kept across a restart", pk.Classifier(p).classify("acme-internal.dev")[0] == "work")


def t2_sanitize():
    print("\n=== T2. What is dropped, what is kept ===")
    from src.senses.browsertabs import BrowserTabSensor
    s = BrowserTabSensor(enabled=True)
    ok = s.sanitize(tab(1, "https://github.com/me/repo/pull/12?token=abc#frag", "Fix bug - Pull Request #12", True))
    check("kept: the host, not the path or query", ok and ok["host"] == "github.com" and "pull" not in json.dumps(ok)
          and "token" not in json.dumps(ok), str(ok))
    check("...with a category", ok["category"] == "code")
    check("a private window is dropped", s.sanitize(tab(2, "https://github.com", "x", incognito=True)) is None)
    check("a bank is dropped", s.sanitize(tab(3, "https://www.chase.com/accounts", "Accounts")) is None)
    check("mail is dropped", s.sanitize(tab(4, "https://mail.google.com/mail/u/0/", "Inbox (3)")) is None)
    check("a health page is dropped by its title", s.sanitize(tab(5, "https://example.org/a", "Symptoms of the flu")) is None)
    check("a router page is dropped", s.sanitize(tab(6, "http://192.168.0.1/", "Router")) is None)
    check("a non-web page (chrome://, file://) is dropped", s.sanitize(tab(7, "chrome://settings", "Settings")) is None
          and s.sanitize(tab(8, "file:///C:/secret.txt", "secret")) is None)
    check("they are only counted", (s.dropped["private"], s.dropped["sensitive"], s.dropped["scheme"]) == (1, 4, 2), str(s.dropped))
    t = s.sanitize(tab(9, "https://example.com", "Invoice for jo@example.com order 123456789", True))
    check("an email and long numbers are taken out of the title", "@" not in t["title"] and "123456789" not in t["title"], t["title"])
    check("garbage is ignored, not a crash", s.sanitize("x") is None and s.sanitize({}) is None)


def sensor(**kw):
    from src.daemon.ledger import Ledger
    from src.senses.browsertabs import BrowserTabSensor
    t = [1000.0]
    led = Ledger(clock=lambda: t[0], **kw.pop("ledger_kw", {}))
    s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: t[0], **kw)
    return s, led, t


def feed(s, tabs, focused=True):
    s.submit({"focused": focused, "tabs": tabs})
    return s.read()


def t3_episodes():
    print("\n=== T3. Episodes ===")
    s, led, t = sensor()
    gh, yt = "https://github.com/x", "https://www.youtube.com/watch?v=1"
    ev = feed(s, [tab(1, gh, "repo", True), tab(2, yt, "video")])
    check("the first active tab starts an episode", len(ev) == 1 and ev[0].meta["event"] == "started"
          and ev[0].meta["site"] == "github.com" and ev[0].meta["category"] == "code", str([e.meta for e in ev]))
    check("its event is never journaled and carries no title", ev[0].persistable is False and "repo" not in json.dumps(ev[0].meta))
    t[0] = 1100
    ev = feed(s, [tab(1, gh, "repo"), tab(2, yt, "video", True)])
    check("switching tabs ends the first and starts the second", ev and ev[0].meta["event"] == "changed"
          and ev[0].meta["site"] == "youtube.com" and ev[0].meta["finished_site"] == "github.com"
          and ev[0].meta["finished_s"] == 100, str([e.meta for e in ev]))
    eps = led.episodes()
    check("the ledger has the github stretch", len(eps) == 1 and eps[0]["site"] == "github.com" and eps[0]["end"] - eps[0]["start"] == 100)
    t[0] = 1102
    feed(s, [tab(1, gh, "repo"), tab(2, yt, "video"), tab(3, "https://news.ycombinator.com", "hn", True)])
    t[0] = 1104
    feed(s, [tab(1, gh, "repo"), tab(2, yt, "video", True)])
    check("a tab opened and gone in seconds is churn, and not a visit", s.stats["churn"] >= 1
          and all(e["site"] != "ycombinator.com" for e in led.episodes()), str(s.stats))
    t[0] = 1200
    feed(s, [tab(1, gh, "repo"), tab(2, yt, "video", True)], focused=False)
    t[0] = 1210
    ev = s.read()
    check("losing focus is not the end of it yet", not ev)
    t[0] = 1216
    ev = s.read()
    check("...but past the grace it ends, at the moment focus was lost",
          ev and ev[0].meta["event"] == "stopped" and led.episodes()[-1]["end"] == 1200, str([e.meta for e in ev]))

    # the extension goes quiet (browser closed)
    t[0] = 1300
    feed(s, [tab(1, gh, "repo", True)])
    t[0] = 1330
    feed(s, [tab(1, gh, "repo", True)])
    t[0] = 1430
    ev = s.read()
    check("a silent extension ends it at the last report, not now", ev and led.episodes()[-1]["end"] == 1330, str(led.episodes()[-1]))

    # private and sensitive pages are not attributed to the page before
    t[0] = 1500
    feed(s, [tab(1, gh, "repo", True)])
    t[0] = 1520
    ev = feed(s, [tab(1, gh, "repo"), tab(4, "https://www.chase.com/", "Chase", True)])
    check("moving to a bank ends the episode and starts none", ev and ev[0].meta["event"] == "stopped" and s.current() is None, str(s.current()))
    t[0] = 1600
    feed(s, [tab(5, "https://reddit.com", "r/x", True, incognito=True)])
    check("a private window starts none", s.current() is None)
    check("and no bank or private site is anywhere in the ledger", all(e["site"] in ("github.com", "youtube.com") for e in led.episodes()),
          str([e["site"] for e in led.episodes()]))

    # idle: at the keyboard or not
    idle = [0.0]
    s2, led2, t2 = sensor(idle_s=lambda: idle[0])
    feed(s2, [tab(1, gh, "repo", True)])
    t2[0] = 1400
    idle[0] = 500
    feed(s2, [tab(1, gh, "repo", True)])
    t2[0] = 1420
    s2.read()
    check("an idle user is not 'reading github'", s2.current() is None and led2.episodes()[-1]["end"] == 1400, str(led2.episodes()))

    # pause
    s3, led3, t3 = sensor()
    feed(s3, [tab(1, gh, "repo", True)])
    t3[0] = 1050
    s3.pause()
    check("pause closes what is open", s3.current() is None and len(led3.episodes()) == 1)
    check("...and reads nothing while paused", s3.submit({"tabs": [tab(1, gh, "repo", True)]}) is False)
    s3.resume()
    check("...until resumed", s3.submit({"tabs": [tab(1, gh, "repo", True)]}) is True)

    # teaching the site in front
    s4, led4, t4 = sensor()
    feed(s4, [tab(1, "https://acme-internal.dev/board", "Quarterly numbers", True)])
    check("'that's work' sets the site in front", s4.teach_current("work") == "acme-internal.dev"
          and s4.current()["category"] == "work")
    t4[0] = 1100
    feed(s4, [tab(1, "https://github.com", "x", True)])
    check("...and it is what the ledger kept", led4.episodes()[0]["category"] == "work", str(led4.episodes()))


def t4_ledger():
    print("\n=== T4. The ledger ===")
    from src.daemon.ledger import Ledger, DAY
    t = [100 * DAY]
    led = Ledger(clock=lambda: t[0], retention_days=30)
    now = t[0]
    led.record(now - 3600, now - 3000, "github.com", "code", "table", "Secret title")
    led.record(now - 2000, now - 1900, "youtube.com", "video", "table")
    led.record(now - 40 * DAY, now - 40 * DAY + 600, "github.com", "code", "table")
    led.churn(3, now)
    check("titles are not stored by default", all(e["title"] == "" for e in led.episodes(0)))
    check("records backwards or empty are refused", led.record(5, 5, "x.com", "code") is False and led.record(9, 5, "x.com", "code") is False)
    by = led.by_category(7)
    check("seconds per category for the week", by == {"code": 600.0, "video": 100.0}, str(by))
    n = led.rollup(now)
    check("rollup folds the old episode into aggregates and deletes the raw row", n == 1 and len(led.episodes(0)) == 2, str(led.counts()))
    check("...and nothing is lost from the totals", led.by_category(60).get("code") == 1200.0, str(led.by_category(60)))
    top = led.top_sites(3)
    check("top sites add raw and aggregate", top[0][0] == "github.com" and top[0][2] == 1200.0 and top[0][3] == 2, str(top))
    check("churn is counted without a page", led.db.execute("SELECT churn FROM daily WHERE category='_churn'").fetchone()[0] == 3)
    kt = Ledger(clock=lambda: t[0], keep_titles=True)
    kt.record(now - 100, now, "a.com", "code", "table", "Kept title")
    check("titles are kept only when asked", kt.episodes(0)[0]["title"] == "Kept title")
    check("forget a site removes it everywhere", led.forget(site="github.com") >= 2 and all(r[0] != "github.com" for r in led.top_sites(9)))
    led.record(now - 50, now, "youtube.com", "video")
    check("forget a category", led.forget(category="video") >= 1 and led.by_category(7) == {})
    led.record(now - 50, now, "x.com", "social")
    led.forget_all()
    check("forget all", led.counts() == {"episodes": 0, "daily": 0, "sites": 0})
    p = os.path.join(tempfile.mkdtemp(), "l.sqlite3")
    a = Ledger(p)
    a.record(1, 100, "a.com", "code")
    a.close()
    check("it is a file that survives a restart", len(Ledger(p).episodes(0, 1e12)) == 1)


def t5_listener():
    print("\n=== T5. The loopback listener ===")
    from src.senses.browsertabs import BrowserTabSensor, load_or_create_token, write_extension_config
    d = tempfile.mkdtemp()
    tok = load_or_create_token(os.path.join(d, "tok"))
    check("a token is made once and kept", len(tok) >= 16 and load_or_create_token(os.path.join(d, "tok")) == tok)
    s = BrowserTabSensor(enabled=True, token=tok, port=0)
    check("it starts", s.start_listener() and s.port > 0)
    check("on 127.0.0.1 only", s._server.server_address[0] == "127.0.0.1", str(s._server.server_address))
    url = f"http://127.0.0.1:{s.port}/tabs"
    body = json.dumps({"focused": True, "tabs": [tab(1, "https://github.com", "x", True)]}).encode()

    def post(headers=None, data=body, path="/tabs"):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        r = urllib.request.Request(f"http://127.0.0.1:{s.port}{path}", data=data, headers=h, method="POST")
        try:
            return urllib.request.urlopen(r, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return "refused"            # the server hung up on an oversized body: also a refusal
    good = {"X-ASH-Token": tok, "Origin": "chrome-extension://abcdef"}
    check("the right token from an extension is accepted", post(good) == 204)
    ev = s.read()
    check("...and becomes an event", ev and ev[0].meta["site"] == "github.com", str(ev))
    check("no token: refused", post({"Origin": "chrome-extension://abcdef"}) == 403)
    check("a wrong token: refused", post({"X-ASH-Token": "nope", "Origin": "chrome-extension://abcdef"}) == 403)
    check("a web page's origin is refused even with the token", post({"X-ASH-Token": tok, "Origin": "https://evil.example"}) == 403)
    check("the wrong path is refused", post(good, path="/other") == 403)
    check("not JSON: a 400, not a crash", post(good, data=b"{nope") == 400)
    check("too large: refused", post(good, data=b"x" * 300000) in (400, 413, "refused"))
    check("the rejections are counted", s.stats["rejected"] >= 5, str(s.stats))
    s.stop_listener()
    no = BrowserTabSensor(enabled=True, token="", port=0)
    check("with no token it will not listen at all", no.start_listener() is False)
    ext = os.path.join(d, "ext")
    os.makedirs(ext)
    check("the extension is given its config", write_extension_config(ext, 4711, tok)
          and json.load(open(os.path.join(ext, "config.json"))) == {"port": 4711, "token": tok})
    check("no extension folder: nothing is written", write_extension_config(os.path.join(d, "nope"), 1, "t") is False)
    m = json.load(open(os.path.join(HERE, "extension", "ash-tabs", "manifest.json")))
    check("the extension asks for tabs and 127.0.0.1 only, and not for private windows",
          set(m["permissions"]) == {"tabs", "alarms"} and m["host_permissions"] == ["http://127.0.0.1/*"]
          and m["incognito"] == "not_allowed", str(m))
    js = open(os.path.join(HERE, "extension", "ash-tabs", "background.js"), encoding="utf-8").read()
    check("...and it reads no page, cookie or history", not any(w in js for w in ("chrome.history", "chrome.cookies", "chrome.scripting", "executeScript",
                                                                                   "document.", "innerText")))


def t6_no_llm():
    print("\n=== T6. No LLM, no network ===")
    banned = {"requests", "openai", "ollama", "langchain", "langchain_core", "transformers", "torch", "urllib3", "httpx",
              "aiohttp", "anthropic", "tools"}
    for rel in ("src/daemon/pagekind.py", "src/daemon/ledger.py", "src/senses/browsertabs.py", "src/daemon/browsing.py"):
        tree = ast.parse(open(os.path.join(HERE, rel), encoding="utf-8").read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods.add(n.module.split(".")[0])
        check(f"{rel} imports no model or HTTP client", not (mods & banned), str(mods & banned))

    # run the whole thing with the outside world cut off and a model that explodes if touched
    import socket
    real = socket.socket.connect

    def guarded(self, addr, *a, **k):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError(f"outbound connection to {addr}")
        return real(self, addr, *a, **k)

    class Boom:
        touched = []

        def __getattr__(self, name):
            Boom.touched.append(name)
            raise AssertionError(f"the model was touched: {name}")

    socket.socket.connect = guarded
    try:
        from src.daemon.ledger import Ledger
        from src.senses.browsertabs import BrowserTabSensor
        t = [1000.0]
        led = Ledger(clock=lambda: t[0])
        s = BrowserTabSensor(enabled=True, ledger=led, clock=lambda: t[0])
        rt = te.rig()
        rt.ash.llm = Boom()
        seen = []
        orig = rt.ash.think
        rt.ash.think = lambda *a, **k: (seen.append(a), orig(*a, **k))[1]
        for url, title, secs in (("https://github.com/x", "repo", 60), ("https://www.youtube.com/w", "video", 60),
                                 ("https://www.chase.com", "bank", 30), ("https://acme.example/x", "Quarterly roadmap", 60)):
            feed(s, [tab(1, url, title, True)])
            t[0] += secs
        for e in s.read() + feed(s, [tab(1, "https://github.com", "x", True)]):
            rt.submit(e)
            rt._tick()
        check("a day of browsing goes through the sensor, ledger and runtime", len(led.episodes()) >= 3, str(led.counts()))
        check("the model was never touched", Boom.touched == [], str(Boom.touched))
        check("the runtime did not ask it to think either", seen == [], str(seen))
    finally:
        socket.socket.connect = real


if __name__ == "__main__":
    t1_pagekind()
    t2_sanitize()
    t3_episodes()
    t4_ledger()
    t5_listener()
    t6_no_llm()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
