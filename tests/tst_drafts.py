"""
tst_drafts.py -- email drafts, read back, sent only on "send it". No real mail server is ever contacted here: the
inbox fetch and the sender are fakes from the first line.

  D1  the ways people ask for a draft; a new email only to an address that was said
  D2  a reply: to the right person's newest email, threaded (In-Reply-To), "Re:" once; drafted by rule from your words
  D3  change it / make it more formal (the model, only when asked, and only if there is one)
  D4  sending: off until email.allow_send; then only "send it", only within ten minutes, exactly the text read back
  D5  a failed send keeps the draft and says why (never the password)
  D6  the inbox keeps addresses and Message-IDs from headers only; its own summaries are unchanged
  D7  in the runtime: no model for the draft; only for lines said to ASH; not for a guest; held for a stranger
  D8  the real sender builds a proper message (SMTP is faked)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_drafts.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.drafts import Draft, Drafts, compose, smtp_host_for, smtp_send  # noqa: E402
from src.daemon.inbox import Inbox, InboxConfig  # noqa: E402

check = tw.check
T = [1_800_000_000.0]
ACCOUNT = {"host": "imap.example.com", "user": "me@example.com", "password": "app-password"}
ROWS = [("Omar Haddad", "Friday plans", "omar@example.com", "<m1@example.com>"),
        ("GitHub", "PR merged", "noreply@github.com", "<m2@example.com>")]


def inbox(rows=ROWS):
    ib = Inbox(InboxConfig.from_dict(ACCOUNT), fetch=lambda c: (len(rows), list(rows)), clock=lambda: T[0], background=False)
    ib.refresh(force=True)
    return ib


def drafts(allow=True, sent=None, polish=None, ib=None):
    sent = sent if sent is not None else []
    d = Drafts(ib if ib is not None else inbox(), account=lambda: dict(ACCOUNT), allow_send=allow,
               send=lambda a, dr: sent.append(dr), polish=polish, signer=lambda: "Khalid", clock=lambda: T[0])
    return d, sent


def d1_asking():
    print("\n=== D1. Asking ===")
    d, _ = drafts()
    for said in ("reply to Omar saying I'll be late", "Ash, reply to Omar, saying I'll be late", "respond to Omar's email telling him I'll be late",
                 "write back to Omar that I'll be late", "can you reply to the last email saying thanks"):
        d.draft = None
        out = d.hear(said)
        check(f"{said!r}: drafted", out is not None and out.startswith("Here's the draft to"), str(out))
    d.draft = None
    out = d.hear("email sara@example.com saying the files are ready")
    check("a new email to an address that was said", out is not None and "sara@example.com" in out and "The files are ready." in out, str(out))
    for said in ("email Sara saying hi", "send an email", "what's in my inbox", "reply all", "tell Omar I'm late"):
        d.draft = None
        check(f"not a draft: {said!r}", d.hear(said) is None, str(d.hear(said)))


def d2_reply():
    print("\n=== D2. A reply ===")
    d, _ = drafts()
    out = d.hear("reply to Omar saying I'll be late")
    dr = d.draft
    check("to Omar's address, threaded under his email", dr.to_addr == "omar@example.com" and dr.in_reply_to == "<m1@example.com>", str(dr))
    check("'Re:' once", dr.subject == "Re: Friday plans")
    check("written from your words, signed, nothing made up", dr.body == "Hi Omar,\n\nI'll be late.\n\nKhalid", repr(dr.body))
    check("read back in full", "“Hi Omar, / I'll be late. / Khalid”" in out, out)
    ib = inbox([("Sara", "Re: the budget", "sara@example.com", "<m3@x>")])
    d2, _ = drafts(ib=ib)
    d2.hear("reply to Sara saying approved")
    check("an existing 'Re:' is not doubled", d2.draft.subject == "Re: the budget")
    d3, _ = drafts()
    out = d3.hear("reply to Zed saying hi")
    check("nobody by that name: says so", out.startswith("I don't see a recent email from Zed") and d3.draft is None, out)
    d4 = Drafts(Inbox(InboxConfig()), account=dict)
    check("no account: says what to do", "not connected to your email" in d4.hear("reply to Omar saying hi"))
    check("compose without a name", compose("", "the files are ready", "K") == "Hi,\n\nThe files are ready.\n\nK")


def d3_change():
    print("\n=== D3. Change it ===")
    d, _ = drafts(polish=lambda body, how: body.replace("Hi Omar,", "Dear Omar,") if "formal" in how else None)
    d.hear("reply to Omar saying I'll be late")
    out = d.hear("change it to I'll be there at 9")
    check("a new body, read back", d.draft.body == "Hi Omar,\n\nI'll be there at 9.\n\nKhalid" and "there at 9" in out, out)
    out = d.hear("make it more formal")
    check("reworded only when asked", d.draft.body.startswith("Dear Omar,") and "Dear Omar" in out, out)
    out = d.hear("make it friendlier")
    check("a rewording that gives nothing back leaves it unchanged", "unchanged" in out and d.draft.body.startswith("Dear Omar,"), out)
    d2, _ = drafts(polish=None)
    d2.hear("reply to Omar saying hi")
    check("no model: says so, tells you how to change it yourself", "change it to" in d2.hear("make it more formal"))


def d4_sending():
    print("\n=== D4. Sending ===")
    d, sent = drafts(allow=False)
    out = d.hear("reply to Omar saying I'll be late")
    check("off by default: the read-back says it stays a draft", "Sending is switched off" in out, out)
    check("'send it' does not send", "switched off" in d.hear("send it") and sent == [])
    d, sent = drafts(allow=True)
    d.hear("reply to Omar saying I'll be late")
    body = d.draft.body
    out = d.hear("send it")
    check("on: 'send it' sends exactly what was read back", out == "Sent to Omar Haddad." and len(sent) == 1 and sent[0].body == body, out)
    check("once", d.hear("send it") is None and len(sent) == 1)
    d.hear("reply to Omar saying I'll be late")
    T[0] += 700
    check("ten minutes later 'send it' sends nothing", d.hear("send it") is None and len(sent) == 1)
    d.hear("reply to Omar saying I'll be late")
    check("'don't send it' drops it", d.hear("don't send it") == "Okay, I won't send it." and d.draft is None and len(sent) == 1)
    check("with nothing waiting, 'send it' is left to others", d.hear("send it") is None)


def d5_failure():
    print("\n=== D5. A failed send ===")
    import smtplib

    def refuse(a, dr):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
    d = Drafts(inbox(), account=lambda: dict(ACCOUNT), allow_send=True, send=refuse, signer=lambda: "K", clock=lambda: T[0])
    d.hear("reply to Omar saying hi")
    out = d.hear("send it")
    check("says the login was refused, keeps the draft, no password in it", "login was refused" in out and d.draft is not None
          and "app-password" not in out, out)
    d.send = lambda a, dr: (_ for _ in ()).throw(TimeoutError())
    out2 = d.hear("send it")
    check("another failure is named in words, not by its class", "too long" in out2 and "TimeoutError" not in out2, out2)


def d6_inbox():
    print("\n=== D6. The inbox ===")
    ib = inbox()
    check("addresses and Message-IDs kept from headers", ib.headers[0] == ROWS[0])
    check("found by first name, full name, address, or 'the last email'",
          ib.last_from("omar") == ROWS[0] and ib.last_from("Omar Haddad") == ROWS[0] and ib.last_from("noreply@github.com") == ROWS[1]
          and ib.last_from("the last email") == ROWS[0])
    check("its summaries are as before", ib.summary(subjects=True) == "2 unread: Omar Haddad -- Friday plans; GitHub -- PR merged.",
          ib.summary(subjects=True))
    old = Inbox(InboxConfig.from_dict(ACCOUNT), fetch=lambda c: (1, [("Omar", "Hi")]), background=False)
    old.refresh(force=True)
    check("a fetch with only (sender, subject) still works; no reply target then", old.summary() == "1 unread email: from Omar."
          and old.last_from("omar") is None)
    check("smtp host from the imap host", smtp_host_for("imap.gmail.com") == "smtp.gmail.com"
          and smtp_host_for("outlook.office365.com") == "smtp.office365.com")


def d7_runtime():
    print("\n=== D7. In the runtime ===")
    import tst_effects as te
    sent = []
    rt = te.rig(email=True, email_cfg=dict(ACCOUNT, allow_send=True), email_send=lambda a, dr: sent.append((a["user"], dr)))
    rt.inbox.fetch = lambda c: (len(ROWS), list(ROWS))       # before any tick: no real server
    rt.inbox.background = False
    rt.inbox.refresh(force=True)
    rt.brain.user = "Khalid"
    n = len(rt.ash.calls)
    rt.say("Ash, reply to Omar saying I'll be late")
    check("drafted by rule, no model", rt.out[-1].startswith("Here's the draft to Omar Haddad") and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("send it", addressed=False)
    check("an overheard 'send it' sends nothing", sent == [])
    rt.say("Ash, send it")
    check("'send it' said to ASH sends it, with the account", len(sent) == 1 and sent[0][0] == "me@example.com"
          and rt.out[-1] == "Sent to Omar Haddad.", rt.out[-1])
    from src.senses.base import Modality, SensorEvent
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, reply to Omar saying cancel everything",
                          addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest cannot draft", rt.drafts.draft is None)
    rt._stranger_hold = lambda now: True
    rt.say("Ash, reply to Omar saying the code is 4321")
    check("a stranger in view: the read-back is held", rt.out[-1] == rt.HOLD_LINE and "4321" not in rt.out[-1], rt.out[-1])
    check("status", rt.status()["drafts"]["sent"] == 1 and rt.status()["drafts"]["allow_send"] is True)
    off = te.rig(email=True, email_cfg=dict(ACCOUNT))
    off.inbox.fetch = lambda c: (len(ROWS), list(ROWS))
    off.inbox.background = False
    check("allow_send is off unless set", off.drafts.allow_send is False)


def d8_real_sender():
    print("\n=== D8. The message the real sender builds (SMTP faked) ===")
    import smtplib
    seen = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            seen["host"], seen["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            seen["tls"] = True

        def login(self, u, p):
            seen["login"] = u

        def send_message(self, msg):
            seen["msg"] = msg
    real_ssl, real_plain = smtplib.SMTP_SSL, smtplib.SMTP
    smtplib.SMTP_SSL = smtplib.SMTP = FakeSMTP
    try:
        d = Draft("omar@example.com", "Omar", "Re: Friday plans", "Hi Omar,\n\nLate.\n\nK", in_reply_to="<m1@example.com>")
        smtp_send(dict(ACCOUNT), d)
        m = seen["msg"]
        check("over SSL to smtp.<domain>:465, logged in as you", (seen["host"], seen["port"], seen["login"]) == ("smtp.example.com", 465, "me@example.com"))
        check("threaded, addressed, the text as read back", m["To"] == "omar@example.com" and m["In-Reply-To"] == "<m1@example.com>"
              and m["Subject"] == "Re: Friday plans" and m.get_content().strip() == "Hi Omar,\n\nLate.\n\nK")
        seen.clear()
        smtp_send(dict(ACCOUNT, smtp_host="smtp.office365.com", smtp_port=587), d)
        check("587: STARTTLS", seen.get("tls") and seen["port"] == 587)
    finally:
        smtplib.SMTP_SSL, smtplib.SMTP = real_ssl, real_plain


if __name__ == "__main__":
    for fn in (d1_asking, d2_reply, d3_change, d4_sending, d5_failure, d6_inbox, d7_runtime, d8_real_sender):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
