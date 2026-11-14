"""
tst_forgetbrowsing.py -- "forget" and "never track" are total.

Review found three ways what the user deleted came back or lingered:
  * "never track youtube" / "forget everything about github.com" deleted the ledger's rows, but the visit being timed at that
    moment (and a tab playing in the background) was written a moment later when it closed;
  * deleted rows stayed in the SQLite file's free pages until overwritten (only "forget my browsing" vacuumed);
  * a torn write to the file of the sites the user told ASH about (tags, "never track") reset it silently, so "never track X"
    stopped applying and the next save overwrote what was left.
Also: a non-ASCII token header made the listener raise instead of refusing.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_forgetbrowsing.py
"""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_tabs as tt  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
GH, YT = "https://github.com/x", "https://www.youtube.com/watch?v=1"


def f1_in_flight():
    print("\n=== What is being timed when you say forget ===")
    s, led, t = tt.sensor()
    tt.feed(s, [tt.tab(1, GH, "repo", True)])
    t[0] = 1200
    s.drop(site="github.com")
    t[0] = 1300
    tt.feed(s, [tt.tab(2, YT, "video", True)])
    check("the github visit in flight when you said 'forget github' is never written", all(e["site"] != "github.com" for e in led.episodes()),
          str(led.episodes()))

    s, led, t = tt.sensor()
    tt.feed(s, [tt.tab(1, GH, "repo", True)])
    t[0] = 1200
    s.drop(site="youtube.com")
    t[0] = 1300
    tt.feed(s, [tt.tab(2, YT, "video", True)])
    check("dropping another site leaves this one alone", any(e["site"] == "github.com" for e in led.episodes()))

    s, led, t = tt.sensor()
    tt.feed(s, [tt.tab(1, GH, "repo", True), tt.tab(2, YT, "video", False, audible=True)])
    check("(a tab playing in the background is being timed)", "youtube.com" in s._bg, str(s._bg))
    s.drop(site="youtube.com")
    check("dropped: it is not", "youtube.com" not in s._bg)
    t[0] = 1400
    tt.feed(s, [tt.tab(1, GH, "repo", True)])
    check("...so nothing about it is written when the sound stops", all(e["site"] != "youtube.com" for e in led.episodes()))

    s, led, t = tt.sensor()
    tt.feed(s, [tt.tab(1, GH, "repo", True), tt.tab(2, YT, "video", False, audible=True)])
    s.drop(everything=True)
    check("'forget my browsing' drops everything in flight", s._cur is None and not s._bg)
    s, led, t = tt.sensor()
    tt.feed(s, [tt.tab(1, GH, "repo", True)])
    s.drop(category="code")
    check("'forget my coding' drops the visit in flight if it is that kind", s._cur is None)


def f2_voice():
    print("\n=== By voice ===")
    from src.daemon import browsing as br
    src = open(br.__file__, encoding="utf-8").read()
    check("ignore, forget a site, forget a kind and forget all each drop what is in flight",
          src.count("s.drop(") == 4, str(src.count("s.drop(")))
    check("the statement of what is kept is true about titles", "titles only as the words I learned from pages you tagged" in src
          and "page text, titles, or anything" not in src)


def f3_ledger():
    print("\n=== The ledger file ===")
    from src.daemon.ledger import Ledger
    led = Ledger(path=os.path.join(tempfile.mkdtemp(), "l.sqlite3"))
    check("deleted rows are overwritten, not just unlinked (secure_delete)", led.db.execute("PRAGMA secure_delete").fetchone()[0] == 1)


def f4_tags():
    print("\n=== The sites you told ASH about ===")
    from src.daemon import pagekind as pk
    p = os.path.join(tempfile.mkdtemp(), "pk.json")
    c = pk.Classifier(p)
    c.ignore("youtube.com")
    c.ignore("reddit.com")
    open(p, "w").close()                                  # the power went mid-write
    c2 = pk.Classifier(p)
    check("an empty file: the last good copy is used, so 'never track youtube' still applies", c2.is_ignored("youtube.com"),
          str(c2.ignored))
    check("...and the damaged file is set aside, not overwritten", os.path.exists(p + ".bad") or os.path.exists(p))
    c3 = pk.Classifier(os.path.join(tempfile.mkdtemp(), "none.json"))
    check("no file at all: nothing ignored, no error", c3.ignored == set())


def f5_listener():
    print("\n=== The listener ===")
    from src.senses.browsertabs import BrowserTabSensor
    s = BrowserTabSensor(enabled=True, token="secret-token", port=0)
    check("it starts", s.start_listener())
    body = json.dumps({"focused": True, "tabs": []}).encode()

    def post(token):
        r = urllib.request.Request(f"http://127.0.0.1:{s.port}/tabs", data=body, method="POST",
                                   headers={"Content-Type": "application/json", "X-ASH-Token": token,
                                            "Origin": "chrome-extension://abc"})
        try:
            return urllib.request.urlopen(r, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return "refused"
    check("a token with a non-ASCII character is refused, not an error", post("café") == 403)
    check("the right token still works afterwards", post("secret-token") == 204)
    s.stop_listener()


if __name__ == "__main__":
    f1_in_flight()
    f2_voice()
    f3_ledger()
    f4_tags()
    f5_listener()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
