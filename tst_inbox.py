"""
tst_inbox.py -- email, read-only.

  E1  Talking to the server: the folder opened read-only, headers peeked
      (never marked read), nothing stored, moved or deleted; the newest few
      kept; logout even on failure
  E2  Who sent it: display names, else the domain; encoded headers decoded
  E3  Checking: not configured, nothing; at most every ten minutes; on its
      own; a failure is kept as its kind, never the password
  E4  Saying it: the count and the senders; the subjects when asked; the
      briefing line; nothing when there is nothing
  E5  Hearing: the phrasings, and what is left alone
  E6  The runtime: answered without the model; in the briefing; status; the
      settings panel can set the account, and the password is never shown

Run:  PYTHONIOENCODING=utf-8 python tst_inbox.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import inbox as ib  # noqa: E402

check = tw.check
T0 = tw.T0
ROWS = [("Sara Ali", "Budget for March"), ("Omar", "Lunch?"), ("github", "[ash] CI failed")]


class FakeIMAP:
    log = []
    fail_login = False
    unseen = b"3 7 9 12"

    def __init__(self, host, port, timeout=None):
        FakeIMAP.log.append(("connect", host, port))

    def login(self, user, pw):
        FakeIMAP.log.append(("login", user))
        if FakeIMAP.fail_login:
            raise ib.imaplib.IMAP4.error("AUTHENTICATIONFAILED")

    def select(self, folder, readonly=False):
        FakeIMAP.log.append(("select", folder, readonly))
        return "OK", [b"4"]

    def search(self, charset, crit):
        FakeIMAP.log.append(("search", crit))
        return "OK", [FakeIMAP.unseen]

    def fetch(self, i, what):
        FakeIMAP.log.append(("fetch", i, what))
        hdr = {b"12": b'From: "Sara Ali" <sara@x.com>\r\nSubject: =?utf-8?q?Budget_for_March?=\r\n\r\n',
               b"9": b"From: Omar <omar@y.com>\r\nSubject: Lunch?\r\n\r\n",
               b"7": b"From: noreply@github.com\r\nSubject: [ash] CI failed\r\n\r\n"}.get(i, b"From: a@b.c\r\n\r\n")
        return "OK", [(b"x", hdr), b")"]

    def store(self, *a):
        FakeIMAP.log.append(("STORE",))

    def logout(self):
        FakeIMAP.log.append(("logout",))


def e1_server():
    print("\n=== E1. Talking to the server ===")
    real = ib.imaplib.IMAP4_SSL
    ib.imaplib.IMAP4_SSL = FakeIMAP
    try:
        FakeIMAP.log = []
        cfg = ib.InboxConfig(host="imap.x.com", user="me", password="pw", keep=3)
        n, rows = ib._imap_fetch(cfg)
        check("the unread count", n == 4)
        check("the newest few, newest first", rows == [("Sara Ali", "Budget for March"), ("Omar", "Lunch?"),
                                                       ("github", "[ash] CI failed")], str(rows))
        check("opened read-only", ("select", "INBOX", True) in FakeIMAP.log)
        fetches = [e for e in FakeIMAP.log if e[0] == "fetch"]
        check("headers only, peeked (never marked read)", fetches and all("BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)]" in e[2]
                                                                         for e in fetches))
        check("only the newest `keep` fetched", [e[1] for e in fetches] == [b"12", b"9", b"7"])
        check("nothing stored, moved or deleted", not any(e[0] == "STORE" for e in FakeIMAP.log))
        check("logged out", FakeIMAP.log[-1] == ("logout",))
        FakeIMAP.log, FakeIMAP.fail_login = [], True
        try:
            ib._imap_fetch(cfg)
            check("a bad login raises", False)
        except ib.imaplib.IMAP4.error:
            check("a bad login raises, and still logs out", FakeIMAP.log[-1] == ("logout",))
        FakeIMAP.fail_login, FakeIMAP.unseen = False, b""
        check("nothing unread", ib._imap_fetch(cfg) == (0, []))
    finally:
        ib.imaplib.IMAP4_SSL = real
        FakeIMAP.unseen = b"3 7 9 12"


def e2_senders():
    print("\n=== E2. Who sent it ===")
    check("a display name", ib.sender_name('"Sara Ali" <sara@x.com>') == "Sara Ali")
    check("no name: the domain", ib.sender_name("noreply@github.com") == "github")
    check("encoded", ib.sender_name("=?utf-8?b?2LPYp9ix2Kk=?= <s@x.com>") == "سارة")
    check("nothing", ib.sender_name("") == "someone")
    check("names joined", ib._names(["A", "B", "A", "C"]) == "A, B and C" and ib._names(["A"]) == "A"
          and ib._names(list("ABCDE")) == "A, B, C and others")


def rig(**cfg):
    t = [T0]
    calls = []

    def fetch(c):
        calls.append(c.user)
        if rig.fail:
            raise ib.imaplib.IMAP4.error("bad pw: hunter2")
        return 3, list(ROWS)
    rig.fail = False
    base = {"host": "imap.x.com", "user": "me", "password": "hunter2"}
    base.update(cfg)
    b = ib.Inbox(ib.InboxConfig.from_dict(base), fetch=fetch, clock=lambda: t[0], background=False)
    b.t, b.calls = t, calls
    return b


def e3_checking():
    print("\n=== E3. Checking ===")
    b = rig(host="")
    b.refresh()
    check("not configured: nothing fetched", b.calls == [] and not b.configured)
    check("disabled: not configured", not rig(enabled=False).configured)
    b = rig()
    b.refresh()
    check("checked", b.calls == ["me"] and b.unread == 3)
    b.t[0] += 300
    b.refresh()
    check("not again within ten minutes", len(b.calls) == 1)
    b.t[0] += 301
    b.refresh()
    check("again after", len(b.calls) == 2)
    b = rig()
    rig.fail = True
    b.refresh()
    check("a failure: kept as its kind", b.last_error == "login failed" and b.stats["errors"] == 1)
    check("never the password", "hunter2" not in str(b.status()) and "hunter2" not in b.hear("any new email?"))
    b.reset()
    check("reset clears it", b.unread is None and b.last_error == "" and b.checked_at < 0)
    check("from_dict ignores junk", ib.InboxConfig.from_dict({"port": "x", "keep": "4", "refresh_s": None}).keep == 4)


def e4_saying():
    print("\n=== E4. Saying it ===")
    b = rig()
    check("not checked yet: nothing", b.summary() is None and b.briefing_part("morning", T0) is None)
    b.refresh()
    check("count and senders", b.summary() == "3 unread: from Sara Ali, Omar and github.", b.summary())
    check("with subjects", b.summary(subjects=True) == "3 unread: Sara Ali -- Budget for March; Omar -- Lunch?; "
                                                       "github -- [ash] CI failed.", b.summary(True))
    check("the briefing line", b.briefing_part("morning", T0) == "Email: 3 unread, from Sara Ali, Omar and github.")
    check("not in other briefings", b.briefing_part("focus", T0) is None)
    b.unread, b.recent = 0, []
    check("nothing unread: said plainly", b.summary() == "No unread email.")
    check("and not in the briefing", b.briefing_part("morning", T0) is None)
    b.unread, b.recent = 1, [("Omar", "")]
    check("one", b.summary() == "1 unread email: from Omar." and
          b.summary(subjects=True) == "1 unread email: Omar -- (no subject).")


def e5_hearing():
    print("\n=== E5. Hearing ===")
    b = rig()
    b.refresh()
    for t in ("any new email?", "do I have any new emails", "did I get any mail", "check my email", "check my inbox",
              "how many unread emails do I have", "ash, any email?", "what's in my inbox", "got any new mail today?"):
        check(f"{t!r}", b.hear(t) == "3 unread: from Sara Ali, Omar and github.", str(b.hear(t)))
    for t in ("who emailed me?", "who are they from", "what are they about", "read me the subjects"):
        check(f"{t!r} -> subjects", (b.hear(t) or "").startswith("3 unread: Sara Ali -- Budget"), str(b.hear(t)))
    for t in ("send an email to Omar", "email Sara the report", "I hate email", "what is an email", "", None,
              "check my email and tell me a joke about the weather in Paris please now"):
        check(f"left alone: {str(t)[:30]!r}", b.hear(t) is None)
    n = rig(host="")
    check("not set up: says how", "not connected to your email" in n.hear("any new email?"))
    check("not set up: 'what are they about' is not ours", n.hear("what are they about") is None)
    check("disabled: not ours", rig(enabled=False).hear("any new email?") is None)
    c = rig()
    s = c.hear("any new email?")
    check("not checked yet: checks now", c.calls == ["me"] and s.startswith("I haven't checked"))
    rig.fail = True
    d = rig()
    rig.fail = True
    d.refresh()
    check("a failure: said", d.hear("any new email?") == "I couldn't check your email just now (login failed).")
    rig.fail = False


def e6_runtime():
    print("\n=== E6. The runtime ===")
    import json
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    from src.ui import settings as st
    clock = [T0]
    out = []
    d = tempfile.mkdtemp()
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, email=True, email_cfg={},
                             briefings=True, briefings_path=os.path.join(d, "b.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.inbox.background = False
    rt.inbox.fetch = lambda c: (2, [("Sara Ali", "Budget"), ("Omar", "Lunch?")])
    rt.inbox.clock = lambda: clock[0]

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("any new email?")
    check("not set up: said how, no model", "not connected" in out[-1] and ash.calls == [])
    p = os.path.join(d, "daemon.json")
    for k, v in (("email.host", "imap.x.com"), ("email.user", "me@x.com"), ("email.password", "hunter2")):
        r = st.change(rt, p, k, v)
        check(f"set {k} from the console, live", r["ok"] and r["live"], str(r))
    check("the password is never shown back", "hunter2" not in json.dumps(st.listing(rt, p)))
    check("saved for the restart", json.load(open(p))["email"]["password"] == "hunter2")
    rt._tick()
    say("any new email?")
    check("answered without the model", out[-1] == "2 unread: from Sara Ali and Omar." and ash.calls == [],
          str(out[-1:]))
    check("in the morning briefing", "Email: 2 unread, from Sara Ali and Omar." in rt.briefings.compose("morning", clock[0]))
    check("status", rt.status()["email"]["unread"] == 2 and "hunter2" not in json.dumps(rt.status()))
    st.change(rt, p, "email.user", "other@x.com")
    check("another account: the old count is forgotten", rt.inbox.unread is None and rt.inbox.checked_at < 0)
    say("send an email to Omar")
    check("sending is not done here: the brain gets it", ash.calls == ["send an email to Omar"])


if __name__ == "__main__":
    for fn in (e1_server, e2_senders, e3_checking, e4_saying, e5_hearing, e6_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
