"""
tst_daemon.py -- smoke test for the always-on layer.

Runs the ambient loop against fake sensors and a stub ASH, so the whole
observe/respond pipeline is exercised without a microphone, a model, or an
LLM. Verifies the behaviors that actually matter for a 24h process:

  * "was that meant for me" reads intent, not just the name
  * a probable question is answered; overheard speech is not
  * direct address always gets a response, even under a spent budget
  * routine ambient events are observed, not answered
  * an urgent event breaks through
  * the hourly unsolicited budget is a hard stop
  * refractory period suppresses back-to-back remarks
  * mute keeps observing but stops responding
  * privacy pause halts sensitive sensors within a tick
  * redaction happens before anything is journaled
  * journal rollup turns thousands of events into a handful of episodes
  * a throwing sensor is backed off and disabled, not fatal
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

from src.daemon import (  # noqa: E402
    AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy,
)
from src.daemon.attention import Disposition  # noqa: E402
from src.senses.base import (  # noqa: E402
    Capability, Modality, PeripheryRegistry, Sensor, SensorEvent, State,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class ScriptedSensor(Sensor):
    """Emits a pre-baked list of events, one batch per poll."""

    capability = Capability(name="scripted", description="test sensor")
    default_interval = 0.01

    def __init__(self, script, privacy_sensitive=False, name="scripted"):
        self.capability = Capability(name=name, description="test",
                                     privacy_sensitive=privacy_sensitive)
        self.script = list(script)
        super().__init__(enabled=True)

    def read(self):
        return [self.script.pop(0)] if self.script else []


class BrokenSensor(Sensor):
    capability = Capability(name="broken", description="always throws")
    default_interval = 0.01

    def read(self):
        raise RuntimeError("sensor on fire")


class StubDrives:
    urgency = 0.3
    caution = 0.3
    curiosity = 0.4
    effort_budget = 0.5
    social = 0.5
    fatigue = 0.0

    def as_dict(self):
        return {k: getattr(self, k) for k in
                ("urgency", "caution", "curiosity", "effort_budget", "social", "fatigue")}


class StubTrace:
    def __init__(self):
        self.percepts = {"novelty": 0.5}
        self.notes = []


class StubBrain:
    def __init__(self):
        self.homeostasis = type("H", (), {"drives": StubDrives()})()
        self.observed = []
        self.slept = 0

    def observe(self, text="", image=None, source="ambient", salience=0.3):
        self.observed.append((source, text))
        return StubTrace()

    def sleep(self, seconds_idle=0):
        self.slept += 1
        return {"ok": True}

    def status(self):
        return {"cycles": len(self.observed)}


class StubResponse:
    def __init__(self, text):
        self.text = text
        self.pathway = type("P", (), {"value": "slow"})()


class StubASH:
    def __init__(self):
        self.brain = StubBrain()
        self.thought = []
        self.episodic_memory = StubEpisodic()

    def think(self, query, image=None, context=""):
        self.thought.append(query)
        return StubResponse(f"noted: {query[:40]}")

    def explain_last(self):
        return {}


class StubEpisodic:
    def __init__(self):
        self.episodes = []

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.episodes.append(summary)


def ev(text, salience=0.3, urgency=0.0, addressed=False, source="scripted"):
    return SensorEvent(source=source, modality=Modality.TEXT, text=text,
                       salience=salience, urgency=urgency, addressed=addressed)


# ----------------------------------------------------------------------
def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    # ---------------------------------------------------------------
    print("\n=== 0b. What is open in the app, not just which app ===")
    from src.daemon.social import compose_check_in, where_phrase
    from src.senses.screen import window_fields, window_subject

    # The title carries the only field that says what the work was, and it
    # was being dropped for the process name everywhere downstream: ASH knew
    # you had been in Code for forty minutes and never which file.
    for title, proc, want in [
        ("brain.py - A.S.H - Visual Studio Code", "Code.exe", "brain.py"),
        ("● system2.py - A.S.H - Visual Studio Code", "Code.exe", "system2.py"),
        ("Trust-Weighted Retrieval — Google Docs - Google Chrome", "chrome.exe",
         "Trust-Weighted Retrieval"),
        ("todo.txt - Notepad", "notepad.exe", "todo.txt"),
        ("Downloads", "explorer.exe", "Downloads"),
        ("README.md • A.S.H", "Code.exe", "README.md"),
        ("Code.md - Visual Studio Code", "Code.exe", "Code.md"),
    ]:
        got = window_subject(title, proc)
        check(f"{title[:34]!r} -> the subject", got == want, repr(got))

    # A title that only names the app is not information.
    for title, proc in [("Discord", "Discord.exe"), ("Visual Studio Code", "Code.exe"),
                        ("", "Code.exe")]:
        check(f"{title!r} says nothing the process does not",
              window_subject(title, proc) == "", repr(window_subject(title, proc)))

    # The title is fielded, and the middle field is the project.
    check("the whole title is read, not just the first field",
          window_fields("brain.py - A.S.H - Visual Studio Code", "Code.exe")
          == ("brain.py", "A.S.H"))
    check("...and a title with nothing to say yields neither field",
          window_fields("Discord", "Discord.exe") == ("", ""))
    check("the project beats the app as the container",
          where_phrase("Code.exe", "brain.py", "A.S.H") == " on brain.py in A.S.H")
    check("the subject leads, because it is the part that says what the work was",
          where_phrase("Code.exe", "brain.py") == " on brain.py in VS Code")
    check("...and the app alone still reads properly",
          where_phrase("Code.exe", "") == " in VS Code")
    check("...and with neither, nothing is invented", where_phrase("", "") == "")
    check("a check-in names the file",
          "on brain.py" in compose_check_in(95, "Code.exe", 14, subject="brain.py"))
    check("...and still reads when there is no file",
          compose_check_in(40, "Code.exe", 14).startswith("You've been in VS Code"),
          compose_check_in(40, "Code.exe", 14))

    # The rollup is where the day's memory is formed; the subject has to
    # survive that or it may as well not have been captured.
    rolled = Journal.summarize_block([
        {"ts": 1000.0 + i * 60, "source": "active_window", "salience": 0.3,
         "text": f"switched to {t}", "meta": {"process": "Code.exe", "subject": s}}
        for i, (t, s) in enumerate([("brain.py - A.S.H - VSC", "brain.py"),
                                    ("system2.py - A.S.H - VSC", "system2.py"),
                                    ("brain.py - A.S.H - VSC", "brain.py")])])
    check("a rolled-up episode remembers the files, not just the editor",
          "on brain.py, system2.py" in rolled["summary"]
          and "mostly in Code.exe" in rolled["summary"], rolled["summary"])
    check("...and they are queryable, not only prose",
          rolled["subjects"] == ["brain.py", "system2.py"], str(rolled["subjects"]))

    blocked = Journal.summarize_block([
        {"ts": 1000.0, "source": "active_window", "salience": 0.3,
         "text": "switched to [redacted window]",
         "meta": {"process": "chrome.exe", "subject": "", "blocked": True}}])
    check("a blocklisted window contributes no subject",
          " on " not in blocked["summary"] and blocked["subjects"] == [],
          blocked["summary"])

    # ---------------------------------------------------------------
    print("\n=== 0c. Looking at a screen, and reading one ===")
    import numpy as np
    from PIL import Image, ImageDraw

    from src.daemon.service import DEFAULT_CONFIG
    from src.senses.screen import ActiveWindowSensor, ScreenshotSensor
    from src.senses.screenread import ScreenReader, _gist, describe, phrase

    # Two layers, because they cost very differently. Looking is eight numpy
    # statistics and tells text from video from an empty desktop. Reading is
    # OCR: ~6s of CPU per frame, and the contents of the screen turned into
    # journaled text.
    text_img = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(text_img)
    for y in range(16, 388, 12):
        draw.text((12, y), "def window_fields(title, process): return subject, project",
                  fill="black")
    xs = np.linspace(0, 255, 640)[None, :, None]
    ys = np.linspace(0, 255, 400)[:, None, None]
    media_img = Image.fromarray(np.clip(
        np.concatenate([np.tile(xs, (400, 1, 1)), np.tile(ys, (1, 640, 1)),
                        np.full((400, 640, 1), 180.0)], axis=2)
        + np.random.rand(400, 640, 3) * 18, 0, 255).astype("uint8"))
    blank_img = Image.new("RGB", (640, 400), (32, 32, 34))

    for name, im in (("text", text_img), ("media", media_img), ("blank", blank_img)):
        got = describe(im)
        check(f"a {name} screen is seen as {name}", got["kind"] == name,
              f"{got['kind']} (ratio {got.get('ratio')})")
    check("a screen it cannot parse says so rather than guessing",
          describe(None)["kind"] == "unknown")
    check("looking turns into a phrase", phrase({"kind": "media"}) == "video or images")
    check("...and carries the words when there are some",
          phrase({"kind": "text"}, "the gist") == "reading: the gist")

    # The gist is the few longest runs of real words -- a full desktop OCRs
    # into hundreds of fragments, most of them menu bars and clocks.
    gist = _gist(["File  Edit  View", "12:04", "x",
                  "def window_fields(title, process, limit=80):",
                  "    return subject[:limit], project[:limit]"])
    check("the gist keeps the sentences and drops the furniture",
          "window_fields" in gist and "12:04" not in gist and "File Edit View" not in gist,
          gist)

    check("a missing OCR engine reads nothing and does not raise",
          ScreenReader(backend="definitely-not-installed").read_text(text_img) == "")
    check("a screen with nothing to show still says something readable",
          phrase({"kind": "text"}) == "a screen of text")

    # The blocklist is matched against the title an application actually
    # writes, not its product name. Google Meet is "Meet - abc-defg-hij -
    # Opera": the first version of this list said "google meet" and missed a
    # call that was on screen at the time.
    blocker = ActiveWindowSensor(enabled=False)
    for title in ["Meet - abc-defg-hij - Opera", "Zoom Meeting",
                  "Chat | Microsoft Teams", "WhatsApp",
                  "Someone is sharing their screen", "1Password", "my bank of america"]:
        check(f"{title[:34]!r} is not captured", blocker.is_blocked(title))
    for title in ["brain.py - A.S.H - Visual Studio Code", "Downloads",
                  "Meet the team - Notion"]:
        check(f"{title[:34]!r} is ordinary work", not blocker.is_blocked(title))

    # The blocklist is a statement about the FOCUSED window, so capturing the
    # whole monitor quietly broke it: a call merely visible beside an allowed
    # window was read anyway, which is how a presenter's name and a meeting
    # URL reached the journal.
    import inspect
    src = inspect.getsource(ScreenshotSensor._capture)
    check("capture crops to the focused window where it can",
          "window_sensor.rect()" in src and "bbox=rect" in src)
    check("...and says which it did", "captures" in ScreenshotSensor(enabled=False).status())
    # Falling back to the whole monitor widens what gets captured past what
    # the blocklist can speak for, so it is not allowed to happen silently.
    nocrop = ActiveWindowSensor(enabled=False)
    nocrop._backend = "xdotool"          # a backend with no rectangle to give
    check("a backend that cannot crop says so rather than widening quietly",
          nocrop.rect() is None and nocrop.status().get("no_crop_because"),
          str(nocrop.status().get("no_crop_because")))

    # Both layers are switches, and the reader is not even built when off.
    off = ScreenshotSensor(enabled=False, read_text=False, describe=False)
    check("reading is off unless asked for", off.reader is None and not off.read_text_enabled)
    on = ScreenshotSensor(enabled=False, read_text=True)
    check("...and OCR is throttled hard when it is on",
          on.read_text_every_s >= 60, f"{on.read_text_every_s}s")
    check("the switches are in the config file, not just in the code",
          {"describe", "read_text", "read_text_every_s"}
          <= set(DEFAULT_CONFIG["sensors"]["screenshot"]))
    check("...and capture itself is still off by default",
          DEFAULT_CONFIG["sensors"]["screenshot"]["enabled"] is False
          and DEFAULT_CONFIG["sensors"]["screenshot"]["read_text"] is False)

    # ---------------------------------------------------------------
    print("\n=== 0d. A noise made to show someone is listening is not a request ===")
    # "Mm-hmm" arrived a second after ASH spoke, scored 0.34 as
    # "mid-conversation", cleared the 0.32 bar, and got a full model answer:
    # a recital of every window the user had switched between, to a sound.
    from src.senses import address
    convo = address.Conversation()
    t0 = time.time()
    convo.note_ash_spoke(t0, asked=False)
    for noise in ["Mm-hmm", "Uh-huh.", "Hmm", "yeah", "okay", "right", "cool", "I see", "got it"]:
        v = address.score(noise, now=t0 + 1, convo=convo)
        check(f"{noise!r} after a statement is acknowledged, not answered",
              v.confidence < address.LIKELY_AT, f"{v.confidence:.2f} {','.join(v.reasons)}")

    convo.note_ash_spoke(t0, asked=True)
    for answer in ["yeah", "yes", "sure", "Mm-hmm"]:
        v = address.score(answer, now=t0 + 1, convo=convo)
        check(f"{answer!r} after a question IS the answer", v.addressed,
              f"{v.confidence:.2f} {','.join(v.reasons)}")

    convo.note_ash_spoke(t0, asked=False)
    check("a real question after a statement is still addressed",
          address.score("what about tomorrow", now=t0 + 1, convo=convo).addressed)
    check("...and 'thanks' is still worth a reply",
          address.score("thanks", now=t0 + 1, convo=convo).likely)
    check("the name always wins: 'ash, yeah' is spoken to ASH",
          address.score("ash, yeah", now=t0 + 1, convo=convo).addressed)
    convo.close()
    check("closing the exchange also forgets that ASH had asked", not convo.expects_reply(t0 + 1))

    # ---------------------------------------------------------------
    print("\n=== 0e. Chat and mail are never read, and a title is said aloud cleanly ===")
    from src.senses.screen import (clean_title, is_private_content, is_self_window,
                                   window_fields)
    from src.senses.screenread import _gist, similar

    # The first day with OCR on, ASH read and journaled a private chat, a
    # mailbox and a company dashboard, and the blocklist said nothing.
    for title, proc in [("General - Some Server - Discord", "Discord.exe"),
                        ("Sent Mail - someone@example.com - Gmail - Opera", "opera.exe"),
                        ("Inbox (12) - Outlook", "olk.exe"), ("Slack | general", "slack.exe"),
                        ("Microsoft Teams", "ms-teams.exe"), ("Signal", "Signal.exe")]:
        check(f"{title[:36]!r} is withheld", is_private_content(title, proc))
    for title, proc in [("brain.py - A.S.H - Visual Studio Code", "Code.exe"),
                        ("Signal processing - Wikipedia - Opera", "opera.exe"),
                        ("Thermodynamics notes - Google Gemini - Opera", "opera.exe")]:
        check(f"{title[:36]!r} is ordinary", not is_private_content(title, proc))
    check("a private window yields no subject or project to be spoken or remembered",
          window_fields("Canceling subscription - a@b.co - Gmail - Opera", "opera.exe") == ("", ""))
    check("ASH's own windows are recognised, so it does not read its own answers",
          is_self_window("A.S.H") and is_self_window("C:\\Windows\\cmd.exe - python  ashd.py")
          and not is_self_window("brain.py - A.S.H - Visual Studio Code"))

    # The protections are general, and a switch for the one person who does
    # not want them: everything ASH does is local.
    from src.senses import build_periphery
    from src.senses.screen import (DEFAULT_BLOCKLIST, looks_private,
                                   set_private_content_protection)
    from src.senses.base import PeripheryRegistry

    sensor = ActiveWindowSensor(enabled=False)
    check("by default the built-in blocklist is on", sensor.is_blocked("1Password - Vault"))
    check("...and a user's own entry is added to it, not instead of it",
          ActiveWindowSensor(enabled=False, blocklist=["my crm"]).is_blocked("1Password")
          and ActiveWindowSensor(enabled=False, blocklist=["my crm"]).is_blocked("My CRM - Acme"))
    off = ActiveWindowSensor(enabled=False, use_default_blocklist=False)
    check("it can be turned off, and then nothing is blocked",
          not any(off.is_blocked(b) for b in DEFAULT_BLOCKLIST))
    check("...while the user's own entries still apply",
          ActiveWindowSensor(enabled=False, blocklist=["my crm"],
                             use_default_blocklist=False).is_blocked("My CRM"))

    check("the chat/mail protection is on by default", is_private_content("x - Discord", "Discord.exe"))
    set_private_content_protection(False)
    check("...and off when asked: contents may be read",
          not is_private_content("x - Discord", "Discord.exe")
          and window_fields("brain notes - Gmail - Opera", "opera.exe")[0] == "brain notes")
    check("...though a chat is still not worth a 'nice focus' remark",
          looks_private("x - Discord", "Discord.exe"))
    set_private_content_protection(True)

    cfg = DEFAULT_CONFIG["privacy"]
    check("both protections default to on in the config",
          cfg["use_default_blocklist"] is True and cfg["protect_private_content"] is True)
    reg = build_periphery({"privacy": {"use_default_blocklist": False,
                                       "protect_private_content": False}},
                          registry=PeripheryRegistry())
    check("a config that turns them off is honoured end to end",
          not reg.sensors["active_window"].is_blocked("1Password")
          and not is_private_content("x - Discord", "Discord.exe"))
    set_private_content_protection(True)

    # A title with decoration spoken aloud read "speaker-high-volume, script
    # capital G...". NFKC folds the fonts back; symbols are dropped.
    spoken = clean_title("\U0001F50A \U0001D4A2\U0001D438\U0001D4C3 \u2461 | Server - Discord")
    check("emoji and font decoration are not spoken", "\U0001F50A" not in spoken
          and "\U0001D4A2" not in spoken and "Server" in spoken, spoken)

    check("two reads of an unchanged page are one reading, spacing jitter and all",
          similar("To teach the network to do something useful, the energy function is designed to accept an",
                  "To teachthe networktodo something useful,the energyfunctionis designed to accept an"))
    check("a different page is news",
          not similar("To teach the network to do something useful, the energy function",
                      "Your Starter plan is active and set to renew automatically on October"))
    check("an address bar is not a sentence worth keeping",
          "compose=" not in _gist(["mail.google.com/mail/u/2/#sent?compose=DmwnWrRvwLqDcSPRrTCb",
                                   "To teach the network to do something useful the energy"]))

    # ---------------------------------------------------------------
    print("\n=== 0f. 'What do you see' is answered by looking ===")
    from tools import screen_tool

    # ASH told its user "I can't capture your screen because I don't have a
    # graphical user interface" while running a screenshot sensor and an OCR
    # engine. Nothing could answer the question, so the model invented a
    # reason it couldn't.
    screen_tool.set_looker(None)
    out = screen_tool.screen_view_tool("what do you see")
    check("with no sensor it says so, truthfully", out["ok"] is False
          and "screen sensor" in out["error"], str(out))

    screen_tool.set_looker(lambda: {
        "ok": True, "title": "brain.py - A.S.H - Visual Studio Code", "process": "Code.exe",
        "subject": "brain.py", "project": "A.S.H", "scene": "text", "fresh": True,
        "age_s": 0.0, "withheld": None, "reader": True,
        "text": "def volunteer(self, situation, context): / return text"})
    out = screen_tool.screen_view_tool("what do you see")
    check("a look reports the window, the project and the words",
          out["ok"] and "VS Code" in out["result"] and "brain.py" in out["result"]
          and "A.S.H" in out["result"] and "volunteer" in out["result"], out.get("result", "")[:200])
    check("the tool marks its answer as already said, so the brain does not retell it",
          out.get("verbatim") is True)
    check("...as sentences, not a data dump", out["result"].count(". ") >= 2
          and "{" not in out["result"], out["result"][:160])

    for why, needle in [("blocked", "blocked list"), ("private", "chat or a mailbox"),
                        ("paused", "privacy pause"), ("off", "switched off"),
                        ("self", "my own window")]:
        screen_tool.set_looker(lambda w=why: {"ok": True, "withheld": w})
        got = screen_tool.screen_view_tool("what do you see")
        check(f"a {why} window is refused, with the reason in words",
              got["ok"] and needle in got["result"], got.get("result", ""))

    screen_tool.set_looker(lambda: {
        "ok": True, "title": "Notes", "process": "notepad.exe", "subject": "todo.txt",
        "project": "", "scene": "text", "fresh": False, "age_s": 70.0, "withheld": None,
        "reader": True, "text": "buy milk"})
    got = screen_tool.screen_view_tool("what do you see")
    check("a stale look says how stale", "ago" in got["result"], got["result"][:120])

    screen_tool.set_looker(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    got = screen_tool.screen_view_tool("what do you see")
    check("a failing sensor cannot take the brain down", got["ok"] is False, str(got))
    screen_tool.set_looker(None)

    # ---------------------------------------------------------------
    print("\n=== 0a. ashctl waits as long as the command needs ===")
    from src.daemon.service import CTL_SLOW_TIMEOUT, CTL_TIMEOUT, ctl_timeout

    # mistral's first `say` here took 122s and ashctl reported a timeout at
    # 30s for an answer the daemon produced perfectly well.
    check("a question waits for the model", ctl_timeout("say what's your status")
          == CTL_SLOW_TIMEOUT, f"{ctl_timeout('say hello'):.0f}s")
    check("...and that is longer than the model's own budget",
          CTL_SLOW_TIMEOUT > float(os.environ.get("ASH_LLM_TIMEOUT", "300")))
    for quick in ("status", "why", "sensors", "history 50", "pause"):
        check(f"{quick.split()[0]!r} still gives up quickly",
              ctl_timeout(quick) == CTL_TIMEOUT)
    check("consolidation is a slow one too", ctl_timeout("sleep") == CTL_SLOW_TIMEOUT)
    check("an empty command does not crash the choice", ctl_timeout("") == CTL_TIMEOUT)

    # ---------------------------------------------------------------
    print("\n=== 0. Addressing: was that meant for me? ===")
    from src.senses import address
    from src.senses.mic import build_event

    # The wake word answered "no" to every one of these, which is the whole
    # reason this module exists: you do not say someone's name to ask them
    # how they are.
    certain = ["hey ash", "ash, stop", "what time is it, ash", "what's your status",
               "how are you", "are you there", "what do you think",
               "can you check the weather", "please open notepad",
               "do you remember what I said", "remind me to call mom at six",
               "tell me a joke", "i need you to summarise that"]
    for t in certain:
        v = address.score(t)
        check(f"addressed: {t!r}", v.addressed, f"{v.confidence:.2f} {','.join(v.reasons)}")

    # Probably, not certainly: answered by route 1b, which the attention gate
    # section below exercises.
    probable = ["what time is it", "what's the weather like", "open notepad",
                "turn the volume down"]
    for t in probable:
        v = address.score(t)
        check(f"probably addressed: {t!r}", v.likely,
              f"{v.confidence:.2f} {','.join(v.reasons)}")

    # Heard, not spoken to. A false positive here is ASH talking over a phone
    # call, which is worse than a missed question.
    overheard = ["I told ash about it", "clean the ash tray", "the ash is everywhere",
                 "volcanic ash covered the road", "hey sam, can you grab that",
                 "i'm so tired", "my back hurts", "he said he would come by later",
                 "the meeting is at three", "switched to Chrome"]
    for t in overheard:
        v = address.score(t)
        check(f"not addressed: {t!r}", v.confidence < address.LIKELY_AT,
              f"{v.confidence:.2f} {','.join(v.reasons)}")

    # The follow-up window: nobody says the name twice in a row.
    convo = address.Conversation()
    t0 = time.time()
    convo.note_ash_spoke(t0)
    check("mid-conversation, a bare question is addressed",
          address.score("what about tomorrow", now=t0 + 3, convo=convo).addressed)
    check("mid-conversation, 'no thanks' is at least probable",
          address.score("no thanks", now=t0 + 3, convo=convo).likely)
    check("ten minutes later the window has shut",
          address.score("no thanks", now=t0 + 600, convo=convo).confidence == 0.0)
    check("a sentence about someone else stays out of it",
          address.score("sam is coming over", now=t0 + 3, convo=convo).confidence
          < address.LIKELY_AT)
    convo.note_ash_spoke(t0)
    convo.close()
    check("closing the conversation shuts the window immediately",
          convo.openness(t0 + 1) == 0.0)

    # ---------------------------------------------------------------
    print("\n=== 1. Attention gate: routing ===")
    gate = AttentionGate(base_threshold=0.62, max_unsolicited_per_hour=3,
                         refractory_seconds=120)
    d = StubDrives()

    r = gate.decide(ev("hey ash what time is it", addressed=True), d)
    check("direct address -> RESPOND", r.disposition == Disposition.RESPOND, r.reasons[0])

    r = gate.decide(ev("switched to Chrome", salience=0.25), d)
    check("routine event -> OBSERVE", r.disposition == Disposition.OBSERVE,
          f"score={r.score:.2f} thr={r.threshold:.2f}")

    r = gate.decide(ev("battery critical at 4%", salience=0.9, urgency=0.95), d)
    check("urgent event -> RESPOND", r.disposition == Disposition.RESPOND,
          f"score={r.score:.2f}")

    r = gate.decide(ev("something mildly interesting", salience=0.8, urgency=0.7), d)
    check("refractory suppresses the next remark", r.disposition == Disposition.OBSERVE,
          "; ".join(r.reasons))

    print("\n--- route 1b: probably addressed ---")
    g1b = AttentionGate(base_threshold=0.62, max_unsolicited_per_hour=0,
                        refractory_seconds=600)
    g1b._last_response = time.time()        # just spoke: refractory fully armed
    question = build_event("mic", "what time is it", address.score("what time is it"))
    r = g1b.decide(question, d)
    check("a probable question is answered", r.disposition == Disposition.RESPOND,
          "; ".join(r.reasons))
    check("...despite a spent budget and an armed refractory timer",
          g1b.status()["unsolicited_last_hour"] == 0)
    r = g1b.decide(question, d, user_busy=1.0)
    check("...and typing does not silence it", r.disposition == Disposition.RESPOND)
    r = g1b.decide(build_event("mic", "the meeting is at three",
                               address.score("the meeting is at three")), d)
    check("overheard speech is still only observed", r.disposition == Disposition.OBSERVE,
          "; ".join(r.reasons))

    print("\n=== 2. Hourly budget is a hard stop ===")
    g2 = AttentionGate(base_threshold=0.1, max_unsolicited_per_hour=2,
                       refractory_seconds=0.001)
    fired = 0
    for i in range(8):
        r = g2.decide(ev(f"interesting thing {i}", salience=0.9, urgency=0.6), d,
                      now=time.time() + i)
        if r.disposition == Disposition.RESPOND:
            fired += 1
    check("budget caps unsolicited remarks", fired <= 2, f"{fired} fired, cap 2")
    r = g2.decide(ev("hey ash", addressed=True), d)
    check("address bypasses a spent budget", r.disposition == Disposition.RESPOND)

    print("\n=== 3. Mute observes but does not respond ===")
    g3 = AttentionGate(base_threshold=0.1)
    g3.mute(True)
    r = g3.decide(ev("very interesting", salience=0.95, urgency=0.9), d)
    check("muted -> OBSERVE", r.disposition == Disposition.OBSERVE)
    r = g3.decide(ev("hey ash", addressed=True), d)
    check("muted still answers direct address", r.disposition == Disposition.RESPOND)

    print("\n=== 4. Privacy: redaction before persistence ===")
    priv = PrivacyPolicy()
    e = ev("my api_key=sk-abcdef0123456789abcdef and mail me at bob@example.com")
    priv.scrub_event(e)
    check("secrets redacted", "sk-abcdef" not in e.text and "bob@example.com" not in e.text,
          e.text[:70])

    print("\n=== 5. Privacy pause halts sensitive sensors ===")
    sensitive = ScriptedSensor([ev("saw something")], privacy_sensitive=True, name="cam")
    benign = ScriptedSensor([ev("cpu fine")], privacy_sensitive=False, name="sys")
    check("pause blocks sensitive", (priv.pause(), not priv.allows(sensitive))[1])
    check("pause allows benign", priv.allows(benign))
    priv.resume()
    check("resume restores sensitive", priv.allows(sensitive))

    # ---------------------------------------------------------------
    print("\n=== 6. Ambient loop end to end ===")
    tmp = tempfile.mkdtemp()
    ash = StubASH()
    periph = PeripheryRegistry()

    script = [ev(f"switched to app {i}", salience=0.2) for i in range(6)]
    script.append(ev("hey ash, what am I working on", salience=0.8, addressed=True))
    script.append(ev("disk 95% full", salience=0.7, urgency=0.9))
    periph.add_sensor(ScriptedSensor(script, name="scripted"))
    periph.add_sensor(BrokenSensor(enabled=True))

    delivered = []
    rt = AmbientRuntime(
        ash=ash, periphery=periph, journal=Journal(tmp),
        gate=AttentionGate(base_threshold=0.62, max_unsolicited_per_hour=5),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=999999),
        response_sink=lambda t, m: delivered.append((t, m)),
    )
    rt.start()
    time.sleep(1.2)
    rt.stop()

    check("all events consumed", rt.stats["events"] >= 8, str(rt.stats))
    check("most events observed silently", rt.stats["observed"] >= 5,
          f"observed={rt.stats['observed']} responded={rt.stats['responded']}")
    check("responded to address + urgent only", rt.stats["responded"] == 2,
          f"responded={rt.stats['responded']}, delivered={len(delivered)}")
    check("addressed prompt reached the brain verbatim",
          any("what am I working on" in q for q in ash.thought))
    check("unprompted remark was framed as unprompted",
          any("UNPROMPTED OBSERVATION" in q for q in ash.thought))

    broken = periph.sensors["broken"]
    check("broken sensor backed off, not fatal",
          broken.error_count > 0 and broken.interval > 0.01,
          f"errors={broken.error_count} interval={broken.interval:.2f}s")

    print("\n=== 7. Journal rollup ===")
    j = Journal(tempfile.mkdtemp())
    base = time.time() - 3 * 3600
    for i in range(400):
        e = ev(f"switched to editor {i % 5}", salience=0.2)
        e.ts = base + i * 8            # one dense block
        e.meta = {"process": "code.exe"}
        j.append(e)
    for i in range(30):
        e = ev("discussed the brain package", salience=0.6, source="microphone")
        e.source = "microphone"
        e.ts = base + 5000 + i * 20    # second block after a gap
        j.append(e)
    j.flush()

    from datetime import datetime
    day = datetime.fromtimestamp(base).strftime("%Y-%m-%d")
    epi = StubEpisodic()
    rep = j.rollup(day=day, episodic=epi)
    check("430 raw events -> few episodes",
          0 < rep.get("blocks", 0) <= 4 and len(epi.episodes) == rep["blocks"],
          f"{rep.get('events')} events -> {rep.get('blocks')} blocks")
    check("rollup is idempotent", j.rollup(day=day, episodic=epi).get("skipped") is not None)
    if epi.episodes:
        print(f"     e.g. {epi.episodes[0][:110]}")

    print("\n=== 8. Context block for prompts ===")
    ctx = j.context_block(seconds=99999, limit=5)
    check("context block renders", bool(ctx) and "\n" in ctx, f"{len(ctx)} chars")

    print("\n=== 9. Speech is the daemon's, not a tool ===")
    from src.senses.actuators import NotifyActuator, SpeakActuator
    from tools.registry import ToolEntry, ToolRegistry
    preg = PeripheryRegistry()
    preg.add_actuator(SpeakActuator(tts_engine=None, enabled=True, dry_run=False))
    preg.add_actuator(NotifyActuator(enabled=True, dry_run=True))
    treg = ToolRegistry()
    preg.bind_to_registry(treg, ToolEntry)
    check("chit-chat can never route to speak", all(treg.match(i) is None for i in ("speak", "say", "talk")),
          str(treg.all_intents()))
    check("other actuators are still routable", treg.match("notify") is not None)

    print("\n=== Attention counters ===")
    print("    ", rt.gate.status()["counts"])

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
