"""
tst_arbiter.py -- a question that can be answered from several places is decided, not hard-wired to one.

"Who is Khalid?" is the person talking, somebody they told ASH about, something in the notes, something said before, an idea the graph
formed, or a public figure. Today "who is khalid" went to Wikipedia (a name-origin blurb) and "who is immortal" got "which one do you
mean?", both about the person asking; then "i mean from your memory" went to the model, which had nothing to go on.

  A1  the decision: what fits best is asked first; the personal beats the public; what is named outright wins; the online one is asked
      last and only if reached; a floor keeps weak guesses from being asked; a source with nothing says nothing
  A2  corrections ("i mean from your memory"): found, only just after a decision, and they move the standing preference (kept)
  A3  which questions are "who is X"
  A4  in the runtime: the user, a person, a note, Wikipedia, memory; the redirect; nothing fits -> the model, as before
  A5  by meaning (the choosers): a correction that is not worded like the rules, a who-question that is not, what the rules catch
      teaches them; and without an encoder the rules stand

Run:  PYTHONIOENCODING=utf-8 python tests/tst_arbiter.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.arbiter import Arbiter, Option, subject_of  # noqa: E402

check = tw.check


def opts(calls, **fits):
    out = []
    for name, fit in fits.items():
        out.append(Option(name, fit, f"{name} fits {fit}", lambda n=name: (calls.append(n), f"{n} says so")[1]))
    return out


def a1_decision():
    print("\n=== A1. The decision ===")
    a = Arbiter(clock=lambda: 1000.0)
    calls = []
    d = a.decide("who is khalid", "khalid", opts(calls, me=1.0, wikipedia=0.6))
    check("the person asking beats the public", d.source == "me" and calls == ["me"], f"{d.source} {calls}")
    check("...and Wikipedia is never asked (it is the one that goes online)", "wikipedia" not in calls)
    calls.clear()
    d = a.decide("who is linda", "linda", opts(calls, people=0.9, wikipedia=0.6, memory=0.24))
    check("somebody you told me about beats a public name", d.source == "people", str(d.ranking))
    calls.clear()
    d = a.decide("who is ada lovelace", "ada lovelace", opts(calls, wikipedia=0.6, memory=0.24))
    check("with no personal evidence the public one wins", d.source == "wikipedia", str(d.ranking))
    calls.clear()
    d = a.decide("who is linda on wikipedia", "linda", opts(calls, people=0.9, wikipedia=0.6))
    check("named outright, it wins over a better fit", d.source == "wikipedia", str(d.ranking))
    calls.clear()
    d = a.decide("who is x", "x", opts(calls, memory=0.05, concepts=0.1))
    check("below the floor nothing is asked", d.source is None and calls == [], f"{d.source} {calls}")
    calls.clear()
    nothing = [Option("people", 0.9, "p", lambda: None), Option("notes", 0.7, "n", lambda: "the note says so")]
    d = a.decide("who is x", "x", nothing)
    check("a source with nothing to say passes it on to the next", d.source == "notes", str(d.ranking))
    check("the ranking and the reason are kept", a.why() and "notes" in a.why(), str(a.why()))
    b = Arbiter(clock=lambda: 1000.0)
    d = b.decide("who is mercury", "mercury", opts([], wikipedia=0.4, concepts=0.3))
    check("a single plain word with a little evidence for each: the stronger fit", d.source == "wikipedia")
    d = b.decide("who is mercury in your concepts", "mercury", opts([], wikipedia=0.4, concepts=0.3))
    check("'in your concepts' names the concepts", d.source == "concepts")
    d = b.decide("who is mercury", "mercury", [], now=1.0)
    check("no options at all: nothing, no crash", d.source is None and d.answer is None)
    boom = [Option("people", 0.9, "p", lambda: 1 / 0), Option("notes", 0.5, "n", lambda: "fine")]
    check("a source that raises is passed over", b.decide("who is x", "x", boom).source == "notes")


def a2_corrections():
    print("\n=== A2. Corrections ===")
    t = [1000.0]
    path = os.path.join(tempfile.mkdtemp(), "arbiter.json")
    a = Arbiter(path, clock=lambda: t[0])
    check("with nothing asked before, there is nothing to correct", a.correction("i mean from your memory") is None)
    a.decide("who is immortal", "immortal", opts([], wikipedia=0.6, memory=0.24))
    for said, want in [("i mean from your memory", "memory"), ("no, from my notes", "notes"), ("I meant wikipedia", "wikipedia"),
                       ("from the web", "wikipedia"), ("nope in your memory", "memory"), ("ask your concepts", "concepts"),
                       ("try from your memory", "memory")]:
        got = a.correction(said)
        check(f"{said!r}", got == want, str(got))
    check("an ordinary question is not a correction", a.correction("who is linda") is None and a.correction("what time is it") is None)
    t[0] += 400
    check("long after, it is not a correction", a.correction("i mean from your memory") is None)
    a.learn("memory", "wikipedia")
    b = Arbiter(path)
    check("the preference moves a little, and is kept", abs(b.prefs["memory"] - 0.05) < 1e-9 and abs(b.prefs["wikipedia"] + 0.05) < 1e-9, str(b.prefs))
    for _ in range(20):
        b.learn("memory", "wikipedia")
    check("but never past a limit", b.prefs["memory"] <= 0.3 and b.prefs["wikipedia"] >= -0.3)


def a3_subjects():
    print("\n=== A3. Which questions ===")
    for said, want in [("who is khalid", "khalid"), ("who is Ada Lovelace?", "Ada Lovelace"), ("who's omar", "omar"), ("who was Alan Turing",
                       "Alan Turing"), ("tell me about lieth", "lieth"), ("tell me more about the roman empire", "the roman empire"),
                       ("what do you know about Omar", "Omar"), ("do you know Omar?", "Omar"), ("who is khalid in your memory", "khalid"),
                       ("Ash, who is Linda", "Linda"), ("what do you know about me", "me")]:
        check(f"{said!r} -> {want!r}", subject_of(said) == want, str(subject_of(said)))
    for said in ("who is this", "who was that?", "who is my friend", "what do you know about my dog", "who are you", "what is a transformer",
                 "who is 42", "who is", "do you know", "what time is it", "who is the one who called me yesterday at the office"):
        check(f"{said!r} is not for the arbiter", subject_of(said) is None, str(subject_of(said)))


class Fetch:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        key = url.rsplit("/", 1)[-1]
        if key not in self.pages:
            import urllib.error
            raise urllib.error.HTTPError(url, 404, "no", {}, None)
        return self.pages[key]


def runtime():
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(notes=True, notes_path=os.path.join(d, "notes.json"), wonder=True, wonder_path=os.path.join(d, "w.json"),
                knowledge_path=os.path.join(d, "k.json"), facts=True, facts_cfg={"enabled": True}, arbiter=True,
                arbiter_path=os.path.join(d, "arbiter.json"))
    rt.brain.user = "Immortal"
    rt.ash.user_aliases = {"Khalid"}
    rt.facts.fetch = Fetch({"Ada_Lovelace": {"type": "standard", "extract": "Ada Lovelace was an English mathematician."},
                            "Mercury": {"type": "standard", "extract": "Mercury is the smallest planet."}})
    return rt


def say(rt, text):
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say(text)
    return (rt.out[-1] if len(rt.out) > k else None), len(rt.ash.calls) > n


def a4_runtime():
    print("\n=== A4. In the runtime ===")
    rt = runtime()
    for s in ("Ash, remember that lieth is a friend", "Ash, remember abd allah sonokut is my best friend"):
        say(rt, s)
    want = rt.notes.everything()
    out, model = say(rt, "who is khalid")
    check("'who is khalid' (the user's other name) is the user, from what was kept; Wikipedia never asked",
          out == want and not model and rt.facts.fetch.calls == [], f"{out!r} {rt.facts.fetch.calls}")
    out, model = say(rt, "who is immortal")
    check("'who is immortal' (the name ASH uses) is the same", out == want and rt.facts.fetch.calls == [], f"{out!r}")
    out, model = say(rt, "who is lieth")
    check("'who is lieth' is what was kept about lieth", out == "You told me: lieth is a friend.", f"{out!r}")
    out, model = say(rt, "who was Ada Lovelace?")
    check("a public name nobody told it about is Wikipedia", out and out.endswith("(Wikipedia)") and not model, f"{out!r}")
    rt.say("remember Linda is my sister")
    n = len(rt.facts.fetch.calls)
    out, model = say(rt, "who is Linda")
    check("a person it was told about is answered from what it was told, not looked up", out and "sister" in out and len(rt.facts.fetch.calls) == n, f"{out!r}")

    rt._user_sentences = lambda: ["I saw mercury at dawn", "mercury is close to the sun in my book"]
    out, model = say(rt, "who is Mercury")
    check("a public name with only a little personal trace is Wikipedia", out and out.endswith("(Wikipedia)"), f"{out!r}")
    out, model = say(rt, "i mean from your memory")
    check("'i mean from your memory' re-asks from memory, with no model", out and out.startswith("From what you've said before") and "mercury" in out.lower()
          and not model, f"{out!r}")
    check("and ASH leans toward memory a little", rt.arbiter.prefs.get("memory", 0) > 0)
    check("how it chose is kept", rt._last_why.get("kind") == "arbiter" and "memory" in (rt._last_why.get("why") or ""), str(rt._last_why))
    out, model = say(rt, "no, from my notes")
    check("a redirect to a place with nothing says so, honestly", out and "your notes" in out and "nothing about" in out and not model, f"{out!r}")

    rt2 = runtime()
    rt2.facts.cfg.enabled = False
    out, model = say(rt2, "who is Zyxwvut")
    check("nothing fits: it goes on to the model as before", model, f"{out!r}")
    out, model = say(rt2, "i mean from your memory")
    check("after a question nothing answered, a correction is honest about having looked", out and "nothing about" in out and not model, f"{out!r}")
    rt3 = runtime()
    out, model = say(rt3, "i mean from your memory")
    check("a correction with no question before it is just a sentence for the model", model, f"{out!r}")
    off = runtime()
    off.arbiter = None
    out, model = say(off, "who is khalid")
    check("switched off: the old order (here Wikipedia has no page for it, so it says so)", out is not None and "Wikipedia" in out, f"{out!r}")


def a5_by_meaning():
    print("\n=== A5. By meaning ===")
    import semantic_fake as sf
    from src.daemon.arbiter import SOURCE_SEEDS, WHO_SEEDS, loose_subject
    from src.daemon.chooser import Chooser
    src = Chooser("s", sf.embed, SOURCE_SEEDS, accept=0.55, margin=0.1)
    a = Arbiter(clock=lambda: 1000.0, sources=src)
    a.decide("who is mercury", "mercury", opts([], wikipedia=0.6))
    check("a correction worded unlike the rules is understood by meaning", a.correction("check your memory now") == "memory", str(src.classify("check your memory now").scores))
    check("an ordinary sentence after a decision is not a correction", a.correction("what time is it") is None and a.correction("thanks") is None)
    check("with no chooser the rules alone decide", Arbiter(clock=lambda: 1000.0).correction("check your memory now") is None)
    before = len(src.learned.get("notes", []))
    a.correction("no, from my notes")
    check("what the rules catch teaches the chooser", len(src.learned.get("notes", [])) == before + 1, str(src.learned))
    for said, want in [("any idea who Linda is", "Linda"), ("what's the deal with Khalid", "Khalid"), ("do you have anything on lieth", "lieth"),
                       ("tell me what you've got on Sara", "Sara"), ("have we talked about Omar", "Omar"), ("I want to know about Lieth", "Lieth"),
                       ("who exactly is Linda", "Linda"), ("so who's this Omar guy", "Omar"), ("is Omar someone you know", "Omar"),
                       ("what can you say about Ada Lovelace", "Ada Lovelace")]:
        check(f"the subject of {said!r}", loose_subject(said) == want, str(loose_subject(said)))
    for said in ("what time is it", "how are you today", "thanks a lot"):
        check(f"no subject in {said!r}", loose_subject(said) is None, str(loose_subject(said)))
    # in the runtime
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(notes=True, notes_path=os.path.join(d, "n.json"), wonder=True, wonder_path=os.path.join(d, "w.json"),
                knowledge_path=os.path.join(d, "k.json"), facts=True, facts_cfg={"enabled": True}, arbiter=True,
                arbiter_path=os.path.join(d, "a.json"), chooser_embed=sf.embed, chooser_dir=os.path.join(d, "c"))
    rt.brain.user = "Immortal"
    rt.ash.user_aliases = set()
    rt.facts.fetch = Fetch({"Mercury": {"type": "standard", "extract": "Mercury is the smallest planet."}})
    rt.say("remember Omar is my cousin")
    out, model = say(rt, "do you have anything on Omar")
    check("a who-question the rules do not list is understood by meaning, and answered from who it is", out and "cousin" in out and not model, f"{out!r}")
    check("what it is called depends on the arbiter, not Wikipedia", rt.facts.fetch.calls == [], str(rt.facts.fetch.calls))
    rt._user_sentences = lambda: ["I saw mercury at dawn", "mercury is close to the sun in my book"]
    out, model = say(rt, "who is Mercury")
    check("a public name with a little personal trace: Wikipedia", out and out.endswith("(Wikipedia)"), f"{out!r}")
    out, model = say(rt, "check your memory now")
    check("and 'check your memory now' (not a listed wording) re-asks from memory", out and out.startswith("From what you've said before") and not model, f"{out!r}")
    out, model = say(rt, "what is the weather tomorrow")
    check("an unrelated question still goes on to the model", model)
    bare = te.rig(notes=True, notes_path=os.path.join(d, "n2.json"), wonder=True, wonder_path=os.path.join(d, "w2.json"),
                  knowledge_path=os.path.join(d, "k2.json"), arbiter=True)
    bare.say("remember Omar is my cousin")
    out, model = say(bare, "do you have anything on Omar")
    check("without an encoder the unlisted phrasing is left to the model, as before", model, f"{out!r}")


if __name__ == "__main__":
    for fn in (a1_decision, a2_corrections, a3_subjects, a4_runtime, a5_by_meaning):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
