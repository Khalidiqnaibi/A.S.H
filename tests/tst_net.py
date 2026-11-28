"""
tst_net.py -- the network layer says what really happened, and keeps a record of what left.

28 Nov: DuckDuckGo's 202 bot check was read as 'no connection'. Run:  PYTHONIOENCODING=utf-8 python tests/tst_net.py
"""

import json
import os
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import net  # noqa: E402

check = tw.check


class Resp:
    def __init__(self, code, body):
        self.status, self._b = code, body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self._b


def opener(code=200, body="hello", exc=None):
    def go(req, timeout=0):
        if exc is not None:
            raise exc
        return Resp(code, body)
    return go


def n1_results():
    print("\n=== N1. Four ways a request ends ===")
    d = tempfile.mkdtemp()
    log = os.path.join(d, "network.jsonl")
    t = [1000.0]
    net.configure(log_path=log, probe=lambda: True, clock=lambda: t[0])
    r = net.fetch("https://example.org/search?q=cats", "a test", opener=opener(200, "ok body"))
    check("a body: ok", r.ok and r.text == "ok body" and r.status == "ok")
    check("a 202 (a search page's bot check): refused, with its code, not 'offline'", net.fetch("https://x.org/?q=a", "t", opener=opener(202, "<html>anomaly</html>")).status == "refused")
    check("a 429: refused", net.fetch("https://x.org/?q=a", "t", opener=opener(429, "slow down")).status == "refused")
    check("a 200 with nothing: empty", net.fetch("https://x.org/?q=a", "t", opener=opener(200, "  ")).status == "empty")
    err = urllib.error.HTTPError("https://x.org", 404, "nf", {}, None)
    check("an HTTP error is the site's: refused, code kept", (lambda r: r.status == "refused" and r.code == 404)(net.fetch("https://x.org", "t", opener=opener(exc=err))))
    net.configure(probe=lambda: False)
    t[0] += 100
    check("no route and no network: down", net.fetch("https://x.org", "t", opener=opener(exc=urllib.error.URLError("no route"))).status == "down")
    net.configure(probe=lambda: True)
    t[0] += 100
    check("a timeout with the network UP is the site's, not 'offline'", net.fetch("https://x.org", "t", opener=opener(exc=TimeoutError("slow"))).status == "refused")
    check("it never raises", True)
    res = net.fetch("https://x.org/", "t", opener=opener(429, "x"))
    check("the words follow the cause", "wouldn't give me" in res.words("the page") and "429" in res.words() and net.Result("down").words().startswith("I have no connection"))
    check("online() is cached", (lambda: (net.configure(probe=lambda: True), net.online())[1])() is True)
    return log


def n2_record(log):
    print("\n=== N2. What left this computer ===")
    rows = [json.loads(x) for x in open(log, encoding="utf-8")]
    check("every request is on record, with host, why and what was sent (the query, not a body)", len(rows) >= 6 and rows[0]["host"] == "example.org" and rows[0]["sent"] == "cats" and rows[0]["why"] == "a test", str(rows[0]))
    check("the status is kept", {r["status"] for r in rows} >= {"ok", "refused", "empty", "down"})
    check("no body is ever kept", all("body" not in r and "text" not in r for r in rows))
    now = rows[-1]["t"] + 10
    got = net.recent(log, now=now)
    say = net.sent_say(got)
    check("'what did you send out today' reads it", say.startswith("In the last day I sent") and "example.org 1 time (cats)" in say and "Only the words" in say, say)
    check("nothing sent: said", net.sent_say([]).startswith("Nothing"))
    check("old rows fall out of the day", net.recent(log, now=now + 3 * 86400) == [])
    for t in ("what did you send out today", "what has left this computer", "have you been online today", "did you go online", "what did you look up today"):
        check(f"asked: {t!r}", net.asks_sent(t))
    for t in ("what did you say", "what time is it", "what did you send me"):
        check(f"not: {t!r}", not net.asks_sent(t))


def n3_runtime(log):
    print("\n=== N3. On the runtime ===")
    import tst_effects as te
    h = te.rig(network_path=log)
    rt = getattr(h, "rt", h)
    n = len(rt.ash.calls)
    h.say("what did you send out today", source="console")
    check("answered from the record, no model", h.out[-1].startswith(("In the last day", "Nothing")) and len(rt.ash.calls) == n, h.out[-1])


if __name__ == "__main__":
    lg = n1_results()
    n2_record(lg)
    n3_runtime(lg)
    net.configure(log_path="", probe=None)
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
