"""
tst_context.py -- "it", "that" and "who was that?".

  C1  Apps: "open it" / "close that" / "switch to it" after ASH opened one,
      or one came to the front; ASH's own windows do not count; stale after
      ten minutes
  C2  Reminders: "remind me about that later" from what you said last, as a
      to-do when it was one; a time kept when given
  C3  Look-ups: the subject of a recent question, a short phrase; never a
      whole sentence
  C4  People: the last person named; who they are, when ASH knows; a capital
      at the start of a sentence is not a name
  C5  "Say that again": ASH's last line
  C6  Not ours: ordinary sentences with "it" or "that" are left alone
  C7  The runtime: each one end to end through the real handlers

Run:  PYTHONIOENCODING=utf-8 python tests/tst_context.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.context import Context  # noqa: E402

check = tw.check
T0 = tw.T0


def ctx(kb=None):
    t = [T0]
    c = Context(clock=lambda: t[0], knowledge=kb, app_names={"winword.exe": "Word"})
    c.t = t
    return c


def c1_apps():
    print("\n=== C1. Apps ===")
    c = ctx()
    check("nothing yet: left alone", c.resolve("open it") is None)
    c.said("Opening Chrome.")
    for text, want in (("open it", "open Chrome"), ("close that", "close Chrome"), ("switch to it", "switch to Chrome"),
                       ("ash, please close it", "close Chrome"), ("Close it.", "close Chrome"),
                       ("open it again", "open Chrome"), ("close that app", "close Chrome")):
        r = c.resolve(text)
        check(f"{text!r} -> {want!r}", r == {"text": want}, str(r))
    c.window("winword.exe")
    check("the app in front, by its name", c.resolve("close it") == {"text": "close Word"})
    c.window("pythonw.exe")
    c.window("msedge.exe")
    check("ASH's own windows do not count", c.resolve("close it") == {"text": "close Word"})
    c.window("notepad.exe")
    check("an unknown process, by its file name", c.resolve("close it") == {"text": "close notepad"})
    c.t[0] += 601
    check("stale after ten minutes", c.resolve("close it") is None)


def c2_remind():
    print("\n=== C2. Reminders ===")
    c = ctx()
    check("nothing said yet: left alone", c.resolve("remind me about that later") is None)
    c.heard("I need to call the bank")
    check("a to-do, in an hour", c.resolve("remind me about that later") == {"text": "remind me to call the bank in 1 hour"})
    check("no 'later': also an hour", c.resolve("remind me about it") == {"text": "remind me to call the bank in 1 hour"})
    check("a time given is kept", c.resolve("remind me about that tomorrow at 9")
          == {"text": "remind me to call the bank tomorrow at 9"})
    check("in 20 minutes", c.resolve("remind me of that in 20 minutes")
          == {"text": "remind me to call the bank in 20 minutes"})
    c.heard("don't let me forget to water the plants.")
    check("another to-do form", c.resolve("remind me about that") == {"text": "remind me to water the plants in 1 hour"})
    c.heard("the dentist on Thursday")
    check("not a to-do: 'about' it", c.resolve("remind me about that later")
          == {"text": "remind me about the dentist on Thursday in 1 hour"})
    c.heard("remind me about that later")
    check("the request itself is not the topic", c.user[-1][1] == "the dentist on Thursday")
    c.heard("ok")
    check("a one-word reply is not the topic", c.resolve("remind me about that")["text"].startswith(
        "remind me about the dentist"))
    c.t[0] += 601
    check("stale after ten minutes", c.resolve("remind me about that") is None)


def c3_lookup():
    print("\n=== C3. Look-ups ===")
    c = ctx()
    c.heard("I had lunch with Omar today")
    check("a whole sentence is not a subject", c.resolve("look it up") is None)
    c.heard("the meeting ran long and everyone was tired")
    check("not even one without an 'I'", c.resolve("look it up") is None)
    c.heard("what's a quasar?")
    check("the subject of a question", c.resolve("look it up") == {"text": "look up a quasar"})
    c.heard("tell me about the Kyoto protocol")
    check("'tell me about'", c.resolve("look that up") == {"text": "look up the Kyoto protocol"})
    c.heard("black holes")
    check("a short phrase", c.resolve("research that") == {"text": "look up black holes"})
    c.heard("I think so")
    check("'I ...' is not a subject; the one before is", c.resolve("find out more about that")
          == {"text": "look up black holes"})


def c3b_lookup_26nov():
    print("\n=== C3b. 'Look that up' on 26 Nov ===")
    c = ctx()
    c.heard("What are the biomarkers that tend to...")
    c.heard("Everyone has cancer in their blood.")
    c.heard("what bio markers in the blood that are different with someone who has cancer that can help with early stage cancer detection")
    c.heard("Local, look that up.")
    r = c.resolve("Look that up.")
    check("the question, not the misheard command before it", r == {"text": "look up what bio markers in the blood that are different with someone who has cancer "
                                                                          "that can help with early stage cancer detection"}, str(r))
    for t in ("Local, look that up.", "Look up the previous stuff.", "search for that", "google that for me", "look up what I just asked"):
        check(f"{t!r} is a reference", Context.is_referring(t), t)
    check("'the previous stuff' resolves the same way", (c.resolve("Look up the previous stuff.") or {}).get("text", "").startswith("look up what bio markers"))
    c = ctx()
    c.heard("Can fix that.")
    c.heard("Nevermind.")
    c.heard("That's funny.")
    check("reactions and fillers are not subjects", c.resolve("look that up") is None, str(c.resolve("look that up")))
    from src.daemon import websearch
    for t in ("Look up the previous stuff.", "search for that stuff", "look up what I just asked"):
        check(f"{t!r} is left to context, not searched literally", websearch.parse(t) is None, str(websearch.parse(t)))
    check("a real look-up still is one", websearch.parse("look up the previous prime minister of Jordan") is not None)


def c3c_meaning():
    print("\n=== C3c. 'What does that mean?' ===")
    c = ctx()
    check("nothing said yet: left alone", c.resolve("What does that mean?") is None)
    c.said("Biomarkers like ctDNA can show a tumour early.")
    r = c.resolve("What does that mean?")
    check("asked with ASH's last line quoted", r == {"text": "What does this mean -- what you just said: \u201cBiomarkers like ctDNA can show a tumour early.\u201d? Explain it simply."}, str(r))
    for t in ("what do you mean by that", "explain that to me", "I don't get it", "what's that mean?"):
        check(f"{t!r} too", (c.resolve(t) or {}).get("text", "").startswith("What does this mean"), str(c.resolve(t)))
    check("'what does sourdough mean' is not about the last line", c.resolve("what does sourdough mean") is None)
    c.t[0] += 3600
    check("an hour later: left alone", c.resolve("What does that mean?") is None)


class KB:
    def __init__(self):
        self.people = {"omar": {"name": "Omar", "relation": "colleague", "desc": ""},
                       "lina": {"name": "Lina", "relation": "", "desc": "runs the bakery"}}

    def person_seen(self, name):
        return name.lower() in self.people

    def relation_of(self, name):
        return (self.people.get(name.lower()) or {}).get("relation", "")


def c4_people():
    print("\n=== C4. People ===")
    c = ctx(KB())
    check("no one named: left alone", c.resolve("who was that?") is None)
    c.heard("I had lunch with Omar today")
    check("a known relation", c.resolve("who was that?") == {"say": "That was Omar -- your colleague."})
    c.said("Lina called while you were out.")
    check("ASH naming someone counts; a description", c.resolve("who's that") == {"say": "That was Lina -- runs the bakery."})
    c.heard("Yesterday I saw Karim at the gym")
    check("a new name: said plainly", c.resolve("who is he?")
          == {"say": "That was Karim. I don't know who they are to you yet."})
    check("'Yesterday' at the start is not a name", "Yesterday" not in [n for _, n in c.people])
    c2 = ctx()
    c2.heard("I met Sara Ali at work on Monday")
    check("a full name; a weekday is not one", [n for _, n in c2.people] == ["Sara Ali"], str(list(c2.people)))
    c.t[0] += 601
    check("stale after ten minutes", c.resolve("who was that") is None)


def c5_again():
    print("\n=== C5. Say that again ===")
    c = ctx()
    check("nothing said: left alone", c.resolve("say that again") is None)
    c.said("Your meeting with Omar is at three.")
    for text in ("say that again", "what did you say?", "repeat that", "come again?", "sorry, what was that",
                 "I didn't catch that", "pardon?"):
        check(f"{text!r}", c.resolve(text) == {"say": "Your meeting with Omar is at three.", "repeat": True})
    c.t[0] += 601
    check("stale after ten minutes", c.resolve("say that again") is None)


def c5b_history():
    print("\n=== C5b. What was said before ===")
    c = ctx()
    c.heard("what time is it")
    c.said("It is ten past two.")
    c.said("Let me see.")
    c.heard("who is Linda")
    c.said("Linda is your cousin.")
    r = c.resolve("what did you say before that?")
    check("before that: the answer before the last, not the filler", r == {"say": "Before that I said: It is ten past two."}, str(r))
    check("'the one before it' is the same", c.resolve("the one before it") == r)
    check("'what was your last answer'", c.resolve("what was your last answer") == {"say": "My last answer was: Linda is your cousin."})
    r2 = c.resolve("what was your last answer")
    c.said(r2["say"])
    check("what was read back is not itself history", c.resolve("what was your last answer") == r2)
    check("what did I ask before that", c.resolve("what did I ask before that?") == {"say": "Before that you said: what time is it"}, str(c.resolve("what did I ask before that?")))
    check("what was my last question", c.resolve("what was my last question") == {"say": "You said: who is Linda"})
    check("these lines are not themselves kept as things the user said", [x[1] for x in c.user] == ["what time is it", "who is Linda"])
    e = ctx()
    check("nothing before: said plainly", e.resolve("what did you say before that") == {"say": "I haven't said anything before that."})
    check("an ordinary question is not a history question", c.resolve("what did you say to Omar") is None)


def c6_not_ours():
    print("\n=== C6. Not ours ===")
    c = ctx(KB())
    c.said("Opening Chrome.")
    c.heard("I need to call the bank")
    c.heard("I had lunch with Omar today")
    for text in ("open it in chrome and search for cats", "who was that guy in the movie Heat", "what is it",
                 "I love that", "close it all down for tonight and tell me a story", "is that right?",
                 "look it up on wikipedia and summarize the history of rome for me in detail", "", None, "x" * 200,
                 "close it" + ", please" * 20):
        check(f"left alone: {str(text)[:40]!r}", c.resolve(text) is None, str(c.resolve(text)))


def c7_runtime():
    print("\n=== C7. The runtime ===")
    import tst_desktop as td
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    d = tempfile.mkdtemp()
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, desktop=True, reminders=True,
                             reminders_path=os.path.join(d, "r.json"), wonder=True,
                             wonder_cfg={"min_gap_s": 10 ** 9}, wonder_path=os.path.join(d, "w.json"),
                             knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    ex = td.FakeEx()
    rt.desktop.ex = ex
    rt.desktop.fresh.window_s = -1                # exact words (see tst_phrasing)

    def say(text):
        clock[0] += 3
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("open chrome")
    say("close it")
    check("'close it' after opening chrome", out[-1].startswith("Close chrome?"), str(out[-1:]))
    rt.desktop._pending = None
    rt.submit(SensorEvent(source="active_window", modality=Modality.TEXT, text="Word: report.docx",
                          meta={"process": "winword.exe", "subject": "report.docx"}))
    rt._tick()
    rt.submit(SensorEvent(source="active_window", modality=Modality.TEXT, text="(private)",
                          meta={"process": "keepass.exe", "blocked": True}))
    rt._tick()
    say("close it")
    check("'close it' after Word came to the front", out[-1].startswith("Close Word?"), str(out[-1:]))
    say("no")
    say("I need to call the bank")
    say("remind me about that later")
    rs = [r for r in rt.reminders.items.values() if r.kind == "reminder"]
    check("'remind me about that later' set the reminder", rs and rs[0].text == "call the bank"
          and abs(rs[0].due - (clock[0] + 3600)) < 5, str(out[-1:]))
    rt.wonder.kb.store_relation("Omar", role="colleague")
    say("I had lunch with Omar today")
    say("who was that?")
    check("'who was that'", out[-1] == "That was Omar -- your colleague.", str(out[-1:]))
    say("say that again")
    check("'say that again'", out[-1] == "That was Omar -- your colleague." and out[-2] == out[-1])
    n_open = len(ex.opened)
    clock[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="open it", addressed=False,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("an overheard 'open it' opens nothing", len(ex.opened) == n_open, str(ex.opened))
    say("open it")
    check("'open it': the app last in front", out[-1] == "Opening Word.", str(out[-1:]))
    clock[0] += 700
    n = len(ash.calls)
    say("close it")
    check("ten minutes on: left to the brain", len(ash.calls) == n + 1 and ash.calls[-1] == "close it",
          str(ash.calls[-1:]))
    check("status", rt.status()["context"]["repeated"] == 1)


if __name__ == "__main__":
    for fn in (c1_apps, c2_remind, c3_lookup, c3b_lookup_26nov, c3c_meaning, c4_people, c5_again, c5b_history, c6_not_ours, c7_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
