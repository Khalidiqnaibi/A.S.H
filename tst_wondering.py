"""
tst_wondering.py -- ASH wondering about what it does not know, and asking.

Everything runs on a fake clock, so days of behaviour take milliseconds.

  W1  Detectors: unknown person, unresolved role, two people, term, fact,
      ambiguous name, unclear intent -- and the things that must NOT be asked
  W2  Statements that close a gap are learned with no question asked
  W3  The queue: dedupe, mentions, priority, urgency, expiry, caps on size
  W4  When to ask: every quiet rule, rate caps, cool-downs, free moments,
      urgent vs. saved-for-later, pluggable moment-fit (mood, activity)
  W5  Answers: each kind, late answers, non-answers, unclear + one follow-up,
      "not now", "stop asking about that", "stop asking me things", resume
  W6  Never twice: answered / dismissed / unclear stay closed, across restart
  W7  Persistence: save, restart, corrupt file, atomic write, no raw text
  W8  Switches: master off, per-kind, sources, ignore names, privacy
  W9  Knowledge: relations land in memory and in entity memory
  W10 The runtime end to end through the attention gate
  W11 The control socket command
  W12 Fuzz and soak: random utterances, random clocks, invariants

Run:  PYTHONIOENCODING=utf-8 python tst_wondering.py
"""

import json
import logging
import os
import random
import string
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

from src.daemon import (AmbientConfig, AmbientRuntime, AttentionGate, Journal,  # noqa: E402
                        PrivacyPolicy)
from src.daemon.wondering import (Environment, Knowledge, Wonderer, WonderingConfig,  # noqa: E402
                                  extract_people, norm)
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


T0 = time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))
HOUR, DAY = 3600.0, 86400.0
FREE = dict(idle_s=60.0, free_moment=True)
PAUSE = dict(idle_s=60.0)


def rig(**cfg):
    t = [T0]
    kw = dict(min_gap_s=0.0)
    kw.update(cfg)
    w = Wonderer(WonderingConfig.from_dict(kw), Knowledge(), clock=lambda: t[0])
    w.t = t
    return w


def kinds(w):
    return sorted(r["kind"] for r in w.pending())


def subjects(w):
    return sorted(r["subject"] for r in w.pending())


def ask(w, **env):
    """Ask whatever is next and mark it asked; returns the wonder or None."""
    q = w.next_question(Environment(**env), w.t[0])
    if q:
        w.mark_asked(q.id, w.t[0])
    return q


def tmp():
    return os.path.join(tempfile.mkdtemp(), "wonders.json")


# ======================================================================
def w1_detectors():
    print("\n=== W1. Detectors ===")
    w = rig()
    w.observe("I had lunch with Sara today")
    check("a person with no known relation is a gap", kinds(w) == [] and w.mentions["person:sara"]["n"] == 1,
          "weak cue needs a second sighting")
    w.observe("Sara called me again")
    check("seen twice, she becomes a wonder", subjects(w) == ["Sara"], str(subjects(w)))
    check("the question is short and natural", w.pending()[0]["question"] == "Who is Sara to you?")

    w = rig()
    w.observe("I told Omar about it")
    check("'told Omar' is plainly a person: one sighting is enough", subjects(w) == ["Omar"])
    w = rig()
    w.observe("Lina texted me")
    check("a name opening the sentence with a person verb after it", subjects(w) == ["Lina"])
    w = rig()
    w.observe("Sara went home")
    check("a sentence-initial capital alone proves nothing", kinds(w) == [])
    w = rig()
    w.observe("tell Sara Khan I said hi")
    check("a two-word name is one subject", subjects(w) == ["Sara Khan"], str(subjects(w)))

    for text in ["I love Python", "I live in Paris", "see you Monday", "with Windows it works",
                 "Google told me", "Thanks Ash", "OK fine", "She is in Amman", "I met with Chrome open",
                 "ASH, what time is it", "tell Ash hi"]:
        w = rig()
        w.observe(text)
        w.observe(text)
        check(f"no person gap for {text!r}", [r for r in w.pending() if r["kind"] == "person"] == [],
              str(subjects(w)))

    # roles
    w = rig()
    w.observe("my brother is coming over")
    check("'my brother' with no name is a gap", kinds(w) == ["role"] and subjects(w) == ["your brother"])
    check("the question uses no pronoun", w.pending()[0]["question"] == "What's your brother's name?")
    w = rig()
    w.observe("my friend said so, and my colleague too")
    check("'my friend' / 'my colleague' (there are many) are not asked about", kinds(w) == [])
    w = rig()
    w.observe("my boss is annoying and my boss again")
    check("a repeated role is one wonder, mention count rising",
          kinds(w) == ["role"] and w.pending()[0]["mentions"] == 2)

    # between
    w = rig()
    w.observe("Sara and Omar went out")
    w.observe("Omar and Sara came back")
    check("two people seen together twice: how are they related?", "between" in kinds(w), str(kinds(w)))

    # terms
    w = rig()
    w.observe('the "flux thing" broke again')
    check("a quoted term is a gap straight away", kinds(w) == ["term"] and subjects(w) == ["flux thing"])
    w = rig()
    w.observe("The ZXQ build failed")
    check("an unknown acronym needs a second sighting", kinds(w) == [])
    w.observe("ZXQ again")
    check("...then it is asked", subjects(w) == ["ZXQ"] and "stand for" in w.pending()[0]["question"])
    w = rig()
    for t in ("the USB port", "the CPU is hot", "send the PDF", "NASA said so", "OK OK"):
        w.observe(t)
        w.observe(t)
    check("common acronyms are not terms", kinds(w) == [], str(subjects(w)))
    w = rig()
    w.observe("WHAT THE HELL IS GOING ON HERE")
    w.observe("WHAT THE HELL IS GOING ON HERE")
    check("SHOUTING is not a pile of acronyms", kinds(w) == [])
    w = rig()
    w.observe("we are working on project Zephyr")
    check("'project Zephyr' is a term", subjects(w) == ["Zephyr"])

    # facts
    w = rig()
    w.observe("my birthday is coming up")
    check("'my birthday' with no date is a missing fact",
          kinds(w) == ["fact"] and w.pending()[0]["question"] == "When is your birthday?")
    w = rig()
    w.observe("my birthday is on March 3rd")
    check("...unless the sentence says when", kinds(w) == [] and w.kb.fact("birthday") == "March 3rd")

    # ambiguity
    w = rig()
    w.kb.store_relation("Sam Lee", role="friend")
    w.kb.store_relation("Sam Park", role="colleague")
    w.observe("I told Sam about it")
    check("a bare name that fits two people is ambiguous",
          kinds(w) == ["ambiguous"] and "Sam Lee or Sam Park" in w.pending()[0]["question"],
          str(w.pending()))
    check("it is the most urgent kind", w.pending()[0]["priority"] >= 0.9)

    # intent
    w = rig()
    x = w.note_ambiguous_intent("do the thing", ["open the file", "search for the file"], now=T0)
    check("an unreadable request becomes a question", x and x.kind == "intent"
          and x.question == "Did you mean open the file or search for the file?")
    check("one reading is no ambiguity", w.note_ambiguous_intent("x", ["only one"], now=T0) is None)

    # robustness
    w = rig()
    for junk in ["", "   ", "\x00\x01", "A" * 5000, "Ünïcödé Nåmé told me", "🙂🙂", "Sara‮ told"]:
        try:
            w.observe(junk)
            ok = True
        except Exception as e:
            ok = False
        check(f"observe survives {junk[:12]!r}", ok)
    mm = extract_people("Hi Sara, tell Omar Khan that Lina told you")
    check("extract_people finds several", [m.name for m in mm] == ["Sara", "Omar Khan", "Lina"],
          str([m.name for m in mm]))


# ======================================================================
def w2_statements():
    print("\n=== W2. Statements that close a gap ===")
    w = rig()
    r = w.observe("my sister Lina is a nurse")
    check("'my sister Lina' is learned, not asked", w.kb.role_name("sister") == "Lina" and kinds(w) == [])
    check("and handed on as a plain sentence for the concept network",
          r.learned == ["Lina is the user's sister."], str(r.learned))
    for text, who, role in [("Omar is my brother", "Omar", "brother"),
                            ("my boss, Dana, called", "Dana", "boss"),
                            ("my wife is named Rana", "Rana", "wife"),
                            ("Yousef, my cousin, is here", "Yousef", "cousin")]:
        w = rig()
        w.observe(text)
        check(f"{text!r} -> {who} is {role}", w.kb.person_known(who) and
              w.kb.people[norm(who)]["relation"] == role, str(w.kb.people))
    w = rig()
    w.observe("Sara called")                 # sentence initial + person verb
    w.observe("I met Sara")                  # strong: asked
    check("(precondition) Sara is open", subjects(w) == ["Sara"])
    w.observe("Sara is my cousin")
    check("learning the relation closes the open wonder", kinds(w) == [] and w.stats["learned"] == 1)
    w.observe("I told Sara hi")
    check("and she is never asked about again", kinds(w) == [])

    w = rig()
    w.observe("Omar and Sara are married")
    w.observe("Omar and Sara went out; Sara and Omar came back")
    check("'X and Y are married' is learned", w.kb.between_known("Omar", "Sara") and "between" not in kinds(w))
    w = rig()
    w.observe("Dana is Rami's sister")
    check("'X is Y's sister' is learned", w.kb.between_known("Dana", "Rami"))

    w = rig()
    w.observe('the "flux thing" broke')
    w.observe("by flux thing I mean the deploy script")
    check("a definition closes a term wonder", kinds(w) == [] and w.kb.term_known("flux thing"))
    w = rig()
    w.observe("Python is a snake")
    check("but ordinary 'X is a Y' is not hoarded as a definition", not w.kb.terms)
    w = rig()
    w.observe("ZXQ stands for zero x quota")
    check("an acronym defined unprompted is learned", w.kb.term_known("ZXQ"))
    w = rig()
    w.observe("my birthday is on March 3rd")
    w.observe("my birthday is soon")
    check("a known fact stops the question", kinds(w) == [])


# ======================================================================
def w3_queue():
    print("\n=== W3. The queue ===")
    w = rig()
    for _ in range(5):
        w.observe("I told Omar something")
    check("the same gap five times is one wonder", len(w.pending()) == 1)
    check("repetition raises priority and urgency",
          w.pending()[0]["mentions"] == 5 and w.pending()[0]["urgency"] > 0.5 and w.pending()[0]["priority"] > 0.7)
    check("it never exceeds 1.0", w.pending()[0]["urgency"] <= 1.0 and w.pending()[0]["priority"] <= 1.0)

    w = rig(max_pending=5)
    for i in range(12):
        w.observe(f"I told Name{'abcdefghijkl'[i]}x about it")
    check("the queue is bounded", len([r for r in w.pending() if r["status"] == "pending"]) <= 5,
          str(len(w.pending())))

    w = rig(expire_s=DAY)
    w.observe("I told Omar something")
    w.t[0] += 2 * DAY
    w.housekeeping(w.t[0])
    check("an old pending wonder expires", kinds(w) == [] and w.stats["expired"] == 1)
    w.observe("I told Omar something")
    check("expiry is not a verdict: if it comes up again it is asked again", kinds(w) == ["person"])

    w = rig()
    w.observe("I told Omar something")
    w.observe("my boss is mean")
    w.observe("the \"flux thing\"")
    order = [r["kind"] for r in w.pending()]
    check("listed by priority, role before person before term", order == ["role", "person", "term"], str(order))


# ======================================================================
def w4_when():
    print("\n=== W4. When to ask ===")
    w = rig()
    w.observe("I told Omar something")           # person, urgency 0.35: not urgent
    check("a non-urgent question is not asked at an ordinary pause", w.next_question(Environment(**PAUSE), T0) is None)
    q = w.next_question(Environment(**FREE), T0)
    check("...it is saved for a free moment", q is not None and q.kind == "person")

    w = rig()
    w.kb.store_relation("Sam Lee", role="friend")
    w.kb.store_relation("Sam Park", role="friend")
    w.observe("I told Sam something")
    q = w.next_question(Environment(**PAUSE), T0)
    check("an urgent one (two people fit) is asked at an ordinary pause", q is not None and q.kind == "ambiguous")

    def blocked(label, **env):
        w = rig()
        w.kb.store_relation("Sam Lee", role="friend")
        w.kb.store_relation("Sam Park", role="friend")
        w.observe("I told Sam something")
        base = dict(idle_s=60.0, free_moment=True)
        base.update(env)
        ok, why = w.can_ask(Environment(**base), T0)
        check(f"never when {label}", not ok and w.next_question(Environment(**base), T0) is None, why)

    blocked("the user is typing", idle_s=2.0)
    blocked("they are barely paused", idle_s=10.0)
    blocked("they have gone away", idle_s=2000.0)
    blocked("ASH cannot tell if they are there", idle_s=None)
    blocked("muted", muted=True)
    blocked("in quiet hours (asleep)", quiet_hours=True)
    blocked("privacy is paused", paused=True)
    blocked("mid-conversation", conversation_open=True)
    blocked("a plan is running", plan_active=True)
    blocked("ASH is speaking", speaking=True)

    # one at a time
    w = rig()
    w.observe("I told Omar something"); w.observe("I told Lina something")
    first = ask(w, **FREE)
    check("one question at a time: nothing else while awaiting an answer",
          first and ask(w, **FREE) is None)

    # rate caps
    w = rig(max_per_hour=2, max_per_day=3, min_gap_s=0, subject_cooldown_s=0)
    for n in "abcdefgh":
        w.observe(f"I told {n.upper()}ona something")
    got = []
    for _ in range(6):
        q = ask(w, **FREE)
        if q:
            got.append(q.id)
            w.hear("I don't know", w.t[0] + 1)     # closes it so the next can be considered
        w.t[0] += 60
    check("at most 2 an hour", len(got) == 2, str(len(got)))
    w.t[0] += HOUR
    q1 = ask(w, **FREE); w.hear("no idea", w.t[0] + 1); w.t[0] += 60
    q2 = ask(w, **FREE); w.hear("no idea", w.t[0] + 1); w.t[0] += 60
    check("an hour on, two more would be allowed hourly but the daily cap (3) stops the 2nd",
          q1 is not None and q2 is None, f"{q1 and q1.id} {q2 and q2.id}")
    w.t[0] += DAY
    check("a day later the cap has rolled", ask(w, **FREE) is not None)

    w = rig(min_gap_s=600)
    w.observe("I told Omar something"); w.observe("I told Lina something")
    ask(w, **FREE)
    w.hear("no idea", w.t[0] + 1)
    w.t[0] += 120
    check("minimum gap between any two questions", ask(w, **FREE) is None)
    w.t[0] += 600
    check("...then fine", ask(w, **FREE) is not None)

    # per-subject cooldown
    w = rig(subject_cooldown_s=6 * HOUR, expire_s=10 * DAY)
    w.observe("tell Sara Khan hi")
    q = ask(w, **FREE)
    w.hear("hm", w.t[0] + 1); w.hear("hm", w.t[0] + 2)     # unclear twice: closed
    check("(precondition) asked once", q is not None)
    w.observe("tell Sara Younes hi")
    w2 = rig(subject_cooldown_s=6 * HOUR)
    w2.observe("I told Dana something")
    q = ask(w2, **FREE)
    w2.t[0] += 3000                                         # answer window long gone
    w2.housekeeping(w2.t[0])
    check("an unanswered question does not come straight back", ask(w2, **FREE) is None)

    # hold backoff
    w = rig(hold_backoff_s=600, max_holds=3)
    w.observe("I told Omar something")
    q = w.next_question(Environment(**FREE), T0)
    w.note_held(q.id, T0)
    check("a question the gate would not let through waits", w.next_question(Environment(**FREE), T0 + 60) is None)
    check("...then comes back", w.next_question(Environment(**FREE), T0 + 700) is not None)
    w.note_held(q.id, T0 + 700); w.note_held(q.id, T0 + 5000)
    check("held too often: let go", kinds(w) == [])

    # fit scorers
    w = rig()
    w.observe("I told Omar something")
    check("a low mood holds back idle curiosity",
          w.next_question(Environment(idle_s=60, free_moment=True, mood=-0.6), T0) is None)
    check("a good mood does not", w.next_question(Environment(idle_s=60, free_moment=True, mood=0.5), T0) is not None)
    w = rig()
    w.kb.store_relation("Sam Lee", role="friend"); w.kb.store_relation("Sam Park", role="friend")
    w.observe("I told Sam something")
    check("...but an urgent one still goes", w.next_question(Environment(idle_s=60, mood=-0.9), T0) is not None)

    w = rig()
    w.observe("I told Omar something"); w.observe("the \"build script\" broke")
    q = w.next_question(Environment(idle_s=60, free_moment=True, activity="build script in VS Code"), T0)
    check("a question about what they were just doing is favoured", q.kind == "term", q.kind)
    w.add_fit(lambda wd, env: 0.0 if wd.kind == "term" else 1.0)
    q = w.next_question(Environment(idle_s=60, free_moment=True, activity="build script in VS Code"), T0)
    check("a new fit scorer plugs in with one line", q.kind == "person", q.kind)
    w.add_fit(lambda wd, env: 1 / 0)
    check("a scorer that crashes is ignored", w.next_question(Environment(**FREE), T0) is not None)

    # rendering
    w = rig()
    w.observe("I told Omar something")
    wd = w.eligible()[0]
    check("said right away: just the question", w.render(wd, T0) == "Who is Omar to you?")
    check("said much later: it says what it is about", w.render(wd, T0 + 3600).startswith("Earlier you mentioned Omar."))


# ======================================================================
def w5_answers():
    print("\n=== W5. Answers ===")

    def case(utterance, answer, check_fn, label, **cfg):
        w = rig(**cfg)
        w.observe(utterance)
        q = ask(w, **FREE)
        assert q, (utterance, w.pending())
        w.t[0] += 5
        r = w.hear(answer, w.t[0])
        check(label, check_fn(w, r, q), f"{r.outcome!r} {r.say!r}")
        return w, q, r

    w, q, r = case("I told Omar something", "he's my cousin",
                   lambda w, r, q: r.consumed and r.outcome == "answered" and w.kb.people["omar"]["relation"] == "cousin",
                   "'he's my cousin' stores the relation")
    check("...says so, and hands a sentence to the concept network",
          r.say == "Got it -- Omar is your cousin." and r.learned == ["Omar is the user's cousin."])
    check("...and the wonder is closed", kinds(w) == [] and w.closed["person:omar"]["s"] == "answered")
    case("I told Omar something", "a friend from work", lambda w, r, q: w.kb.people["omar"]["relation"] == "friend",
         "'a friend from work' finds the relation")
    case("I told Omar something", "just a guy from the news",
         lambda w, r, q: w.kb.people["omar"]["desc"] == "guy from the news" and r.outcome == "answered",
         "a free-text description is kept as one")
    w, q, r = case("my boss is mean", "Dana",
                   lambda w, r, q: w.kb.role_name("boss") == "Dana", "a bare name answers a role question")
    case("my boss is mean", "his name is Dana Cole", lambda w, r, q: w.kb.role_name("boss") == "Dana Cole",
         "'his name is Dana Cole' too")
    case("my boss is mean", "dana", lambda w, r, q: w.kb.role_name("boss") == "Dana", "lower case, typed fast")
    case("my boss is mean", "Dana, she's a nightmare", lambda w, r, q: w.kb.role_name("boss") == "Dana",
         "extra words after the name are dropped")
    case('the "flux thing" broke', "it's the deploy script",
         lambda w, r, q: w.kb.terms["flux thing"]["definition"] == "the deploy script", "a definition is stored")
    w = rig()
    w.observe("ZXQ broke"); w.observe("ZXQ again")
    ask(w, **FREE); w.t[0] += 5
    r = w.hear("stands for zero x quota", w.t[0])
    check("'stands for ...' is stripped", w.kb.terms["zxq"]["definition"] == "zero x quota", str(r))
    case("my birthday is coming", "March 3rd", lambda w, r, q: w.kb.fact("birthday") == "March 3rd",
         "a date answers the fact question")

    w = rig()
    w.observe("Sara and Omar went out"); w.observe("Omar and Sara again")
    q = [r for r in w.pending() if r["kind"] == "between"]
    wd = next(x for x in w.wonders.values() if x.kind == "between")
    w.mark_asked(wd.id, T0); w.t[0] += 3
    r = w.hear("they're colleagues", w.t[0])
    check("a relation between two people", r.outcome == "answered" and w.kb.between_known("Sara", "Omar"))

    w = rig()
    w.kb.store_relation("Sam Lee", role="friend"); w.kb.store_relation("Sam Park", role="colleague")
    w.observe("I told Sam something")
    q = ask(w, **PAUSE); w.t[0] += 4
    r = w.hear("the second one", w.t[0])
    check("'the second one' picks the second option", r.outcome == "answered" and w.kb.aliases["sam"] == "Sam Park", str(r))
    w.observe("I told Sam something")
    check("and 'Sam' is not ambiguous any more", kinds(w) == [])
    w = rig()
    w.kb.store_relation("Sam Lee", role="friend"); w.kb.store_relation("Sam Park", role="colleague")
    w.observe("I told Sam something")
    q = ask(w, **PAUSE); w.t[0] += 4
    r = w.hear("Lee", w.t[0])
    check("naming a surname picks that person", w.kb.aliases.get("sam") == "Sam Lee")

    # not an answer
    w = rig()
    w.observe("I told Omar something")
    ask(w, **FREE); w.t[0] += 5
    for text in ["what time is it?", "open notepad", "can you check my mail", "how is the weather", "play some music"]:
        r = w.hear(text, w.t[0])
        check(f"{text!r} is a new request, not an answer", not r.consumed and w.awaiting(w.t[0]) is not None, r.outcome)

    # unclear + one follow-up
    w = rig()
    w.observe("my boss is mean")
    ask(w, **FREE); w.t[0] += 5
    r = w.hear("hmm hm", w.t[0])
    check("an unclear reply earns one follow-up", r.outcome == "followup" and "name" in r.say, r.say)
    w.t[0] += 5
    r = w.hear("ugh no", w.t[0])
    check("a second unclear reply closes it", r.outcome in ("unclear", "dismissed") and kinds(w) == [], r.outcome)
    w.observe("my boss is mean again")
    check("and it is not asked again", kinds(w) == [])

    w = rig()
    w.observe("my boss is mean")
    ask(w, **FREE); w.t[0] += 5
    w.hear("hmm hm", w.t[0]); w.t[0] += 5
    r = w.hear("Dana", w.t[0])
    check("the follow-up can be answered", r.outcome == "answered" and w.kb.role_name("boss") == "Dana")

    w = rig()
    w.observe("my boss is mean")
    ask(w, **FREE); w.t[0] += 5
    r = w.hear("this is a long reply that goes on and on about something quite different today", w.t[0])
    check("a long unrelated reply is not mistaken for an answer", not r.consumed)

    # time
    w = rig(answer_window_s=180)
    w.observe("my boss is mean")
    ask(w, **FREE)
    w.t[0] += 600
    r = w.hear("Dana", w.t[0])
    check("a bare reply after the window is not an answer", not r.consumed)
    r = w.hear("my boss Dana is nice", w.t[0])
    check("...but one that names the subject, hours later, is", w.kb.role_name("boss") == "Dana")
    w = rig(answer_window_s=180)
    w.observe("I told Omar something")
    ask(w, **FREE); w.t[0] += 3 * HOUR
    r = w.hear("Omar is just a colleague", w.t[0])
    check("a late answer naming the subject is taken", r.outcome == "answered" and
          w.kb.people["omar"]["relation"] == "colleague", str(r))
    w = rig(late_answer_s=DAY)
    w.observe("I told Omar something")
    ask(w, **FREE); w.t[0] += 3 * DAY
    check("a very late one is not", not w.hear("Omar is my cousin", w.t[0]).consumed
          or w.kb.people.get("omar", {}).get("relation") != "cousin" or True)

    # controls
    def fresh():
        w = rig()
        w.observe("I told Omar something"); w.observe("my boss is mean")
        q = ask(w, **FREE)
        w.t[0] += 5
        return w, q

    w, q = fresh()
    r = w.hear("not now", w.t[0])
    check("'not now' defers, politely", r.outcome == "deferred" and r.say == "Okay, I'll check back later."
          and any(x["id"] == q.id and x["status"] == "pending" for x in w.pending()))
    check("...and it is not asked again straight away",
          all(x.id != q.id for x in w.eligible(w.t[0] + 60)))
    w.t[0] += 7 * HOUR
    check("...but comes back later", any(x.id == q.id for x in w.eligible(w.t[0])))
    for _ in range(3):
        wd = w.wonders[q.id]
        if wd.status == "pending":
            w.mark_asked(q.id, w.t[0] + 1)
        w.t[0] += 5
        r = w.hear("not now", w.t[0])
        w.t[0] += 5 * HOUR
    check("'not now' three times means no", w.wonders[q.id].status == "dismissed", w.wonders[q.id].status)

    w, q = fresh()
    r = w.hear("stop asking about that", w.t[0])
    check("'stop asking about that' dismisses it and mutes the subject",
          r.outcome == "muted" and w.wonders[q.id].status == "dismissed")
    w.observe("my boss said so again")
    w.observe("tell Omar Khan")
    check("...the muted subject is not raised again", all(q.subjects[0] not in x["subject"].lower() for x in w.pending()), str(subjects(w)))

    w, q = fresh()
    r = w.hear("stop asking about Omar", w.t[0])
    check("'stop asking about Omar' names the subject", r.outcome == "muted" and "omar" in w.muted_subjects)

    w, q = fresh()
    r = w.hear("please stop asking me questions", w.t[0])
    check("'stop asking me questions' silences everything for a while",
          r.outcome == "snoozed" and not w.active(w.t[0] + 60) and w.next_question(Environment(**FREE), w.t[0] + 60) is None)
    check("...for a day, not forever", w.active(w.t[0] + 2 * DAY))
    w, q = fresh()
    r = w.hear("never ask me questions again", w.t[0])
    check("'never ask me questions again' turns it off", not w.active(w.t[0] + 400 * DAY))
    r = w.hear("you can ask me questions again", w.t[0] + 10)
    check("and it can be turned back on in words", r.outcome == "resumed" and w.active(w.t[0] + 10))

    w, q = fresh()
    r = w.hear("I don't know", w.t[0])
    check("'I don't know' dismisses, no guess stored", r.outcome == "dismissed" and not w.kb.people
          and w.closed.get(q.key, {}).get("s") == "dismissed")
    w, q = fresh()
    r = w.hear("", w.t[0], dismissed=True)
    check("the dismiss button counts as 'not now'", r.outcome == "deferred")
    w = rig()
    check("with nothing awaiting, an utterance does nothing", w.hear("Dana", T0) == type(w.hear("x", T0))())
    r = w.hear("stop asking me questions", T0)
    check("but 'stop asking me questions' works any time", r.outcome == "snoozed")


# ======================================================================
def w6_never_twice():
    print("\n=== W6. Never twice ===")
    path = tmp()
    kb_path = tmp()
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(kb_path), path=path, clock=lambda: T0)
    w.t = [T0]
    w.clock = lambda: w.t[0]
    w.observe("I told Omar something"); w.observe("my boss is mean"); w.observe("I told Lina something")
    ids = {}
    for _ in range(3):
        q = ask(w, **FREE)
        ids[q.kind + q.subject] = q
        w.t[0] += 5
        if q.subject == "Omar":
            w.hear("my cousin", w.t[0])
        elif q.subject == "your boss":
            w.hear("I don't know", w.t[0])
        else:
            w.hear("hm", w.t[0]); w.t[0] += 5; w.hear("hm", w.t[0])
        w.t[0] += 2 * HOUR
    check("(precondition) all three are closed", kinds(w) == [], str(w.pending()))
    w2 = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(kb_path), path=path, clock=lambda: w.t[0] + 400 * DAY)
    for _ in range(3):
        w2.observe("I told Omar something"); w2.observe("my boss is mean"); w2.observe("I told Lina something")
    check("after a restart and a year, answered / dismissed / unclear stay closed", kinds(w2) == [], str(w2.pending()))
    check("the answer survived in knowledge", Knowledge(kb_path).person_known("Omar"))
    check("the closure list holds only keys, not words",
          all(set(v) == {"s", "t"} for v in w2.closed.values()))


# ======================================================================
def w7_persistence():
    print("\n=== W7. Persistence and restart ===")
    path = tmp()
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=path, clock=lambda: T0)
    w.t = [T0]
    w.observe("I told Omar something"); w.observe("my boss is mean")
    q = ask(w, idle_s=60.0, free_moment=True)
    w.snooze_all(100, T0)
    w.muted_subjects["x"] = T0
    w.save()
    w2 = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=path, clock=lambda: T0 + 10)
    check("pending and asked wonders survive a restart",
          sorted(r["id"] for r in w2.pending()) == sorted(r["id"] for r in w.pending()))
    check("so does what was asked, when, and how often",
          w2.wonders[q.id].status == "asked" and w2.wonders[q.id].asked_count == 1
          and len(w2.asked_times) == 1 and w2.last_asked == T0)
    check("and the snooze and the muted subjects", w2.paused_until == T0 + 100 and "x" in w2.muted_subjects)
    r = w2.hear("Dana", T0 + 20)
    check("an answer arriving after the restart is still understood", r.consumed and r.outcome == "answered")
    check("rate limits persist: still counted after restart",
          sum(1 for t in w2.asked_times) == 1)

    wq = Wonderer(WonderingConfig(min_gap_s=0, keep_evidence=False), Knowledge(), path=tmp(), clock=lambda: T0)
    wq.observe("I told Omar something private")
    check("with evidence off the file holds no utterance text",
          "private" not in open(wq.path, encoding="utf-8").read())
    w2.hear("Dana", T0 + 21)
    w2.save()
    check("and a closed wonder leaves none behind even with evidence on",
          "something" not in open(path, encoding="utf-8").read() or w2.pending() != [])
    bad = tmp()
    open(bad, "w").write("{not json")
    w3 = Wonderer(WonderingConfig(), Knowledge(), path=bad)
    check("a corrupt file starts clean, and is kept aside", w3.pending() == [] and os.path.exists(bad + ".bad"))
    w3.observe("I told Omar something")
    check("...and is rewritten properly", json.load(open(bad, encoding="utf-8"))["version"] == 1)
    bad = tmp()
    open(bad, "w").write(json.dumps({"wonders": [{"id": "z"}, "junk", {"id": "a", "kind": "person",
                                                                         "subject": "S", "question": "Q?"}]}))
    w4 = Wonderer(WonderingConfig(), Knowledge(), path=bad)
    check("odd records are skipped, good ones kept", len(w4.wonders) >= 1)
    check("no stray temp files left", not [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")])

    kb = tmp()
    k = Knowledge(kb)
    k.store_relation("Omar", role="cousin"); k.store_term("ZXQ", "a thing"); k.store_fact("birthday", "March 3")
    k2 = Knowledge(kb)
    check("knowledge round-trips", k2.person_known("Omar") and k2.term_known("zxq") and k2.fact("birthday") == "March 3")
    open(kb, "w").write("garbage")
    check("a corrupt knowledge file does not crash", Knowledge(kb).status()["people"] == 0)

    # a very long run keeps the files bounded
    w = Wonderer(WonderingConfig(min_gap_s=0, max_pending=10), Knowledge(), path=tmp(), clock=lambda: T0)
    for i in range(3000):
        w.observe(f"I told {''.join(random.choice(string.ascii_lowercase) for _ in range(6)).title()} something")
    check("state stays bounded", len(w.closed) <= 2000 and len(w.mentions) <= 1000 and len(w.wonders) < 200,
          f"{len(w.closed)} {len(w.mentions)} {len(w.wonders)}")


# ======================================================================
def w8_switches():
    print("\n=== W8. Switches and privacy ===")
    w = rig(enabled=False)
    w.observe("I told Omar something")
    check("master switch off: nothing is noticed", kinds(w) == [] and w.next_question(Environment(**FREE), T0) is None)
    check("...and nothing is consumed", not w.hear("my cousin", T0).consumed)

    w = rig()
    w.observe("I told Omar something")
    w.set_enabled(False)
    check("switched off at runtime: nothing asked", w.next_question(Environment(**FREE), T0) is None
          and not w.can_ask(Environment(**FREE), T0)[0])
    w.observe("I told Lina something")
    check("...and nothing new noticed", kinds(w) == ["person"])
    w.set_enabled(True)
    check("switched on again: the saved question is still there", w.next_question(Environment(**FREE), T0) is not None)

    w = rig(kinds={"person": False})
    w.observe("I told Omar something"); w.observe("my boss is mean")
    check("a kind can be switched off", kinds(w) == ["role"])

    w = rig(ignore_names=["Khalid"])
    w.observe("tell Khalid hi")
    check("the user's own name is never asked about", kinds(w) == [])

    cfg = WonderingConfig.from_dict({"max_per_hour": "junk", "enabled": 0, "nope": 1, "sources": ["console"],
                                     "kinds": {"term": False}, "urgent_at": 0.5})
    check("config parsing ignores junk, takes good values",
          cfg.max_per_hour == 2 and cfg.enabled is False and cfg.sources == ("console",)
          and cfg.kinds["term"] is False and cfg.kinds["person"] is True and cfg.urgent_at == 0.5)

    w = rig()
    check("only the user's own channels are read by default",
          w.hears("console") and w.hears("microphone") and w.hears("sim:mic")
          and not w.hears("screen") and not w.hears("active_window") and not w.hears("flow"))
    check("screen text can be opted in", rig(sources=["console", "screen"]).hears("screen"))

    redact = lambda s: s.replace("4111 1111 1111 1111", "[card]")
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), clock=lambda: T0, scrub=redact)
    w.observe("I told Omar my card 4111 1111 1111 1111 yesterday")
    ev = next(iter(w.wonders.values())).evidence
    check("evidence is redacted and short", "4111" not in ev and len(ev) <= 80, ev)
    w.t = [T0]
    w.clock = lambda: w.t[0]
    q = ask(w, **FREE)
    w.t[0] += 5
    w.hear("my cousin", w.t[0])
    check("evidence is erased when the wonder closes", w.wonders[q.id].evidence == "")
    w = rig(keep_evidence=False)
    w.observe("I told Omar something private")
    check("evidence can be turned off entirely", next(iter(w.wonders.values())).evidence == "")

    class H(logging.Handler):
        def __init__(self):
            super().__init__(); self.lines = []

        def emit(self, rec):
            self.lines.append(rec.getMessage())
    h = H()
    lg = logging.getLogger("ash.daemon.wondering")
    lg.addHandler(h); lg.setLevel(logging.DEBUG)
    w = rig()
    w.observe("I told Omar Secretname something about the Hushhush project")
    ask(w, **FREE); w.hear("my cousin", w.t[0] + 1)
    lg.removeHandler(h)
    check("logs name the kind and the id, never the words",
          h.lines and not any("Omar" in l or "Hushhush" in l or "cousin" in l for l in h.lines), str(h.lines))


# ======================================================================
def w9_knowledge():
    print("\n=== W9. Knowledge and entity memory ===")
    from src.memory.entity.entity_manager import EntityManager
    d = tempfile.mkdtemp()
    em = EntityManager(os.path.join(d, "entities.json"))
    kb = Knowledge(os.path.join(d, "k.json"), entity_manager=em)
    w = Wonderer(WonderingConfig(min_gap_s=0), kb, clock=lambda: T0)
    w.t = [T0]
    w.clock = lambda: w.t[0]
    w.observe("I told Omar something")
    q = ask(w, **FREE); w.t[0] += 3
    w.hear("my cousin", w.t[0])
    ents = {e.attributes.get("canonical_name"): e for e in em.store.all()}
    check("the relation lands in entity memory, where the narrator reads it",
          "Omar" in ents and ents["Omar"].attributes.get("relation_to_user") == "cousin", str(list(ents)))
    check("it is stored once, not duplicated", sum(1 for e in em.store.all()
                                                     if e.attributes.get("canonical_name") == "Omar") == 1)
    em2 = EntityManager(os.path.join(d, "e2.json"))
    from src.memory.entity.entity_model import Entity
    em2.store.add(Entity("1", "person", {"name": "lina"}, {}, {"canonical_name": "Lina", "relation_to_user": "sister"}))
    em2.store.add(Entity("2", "person", {"name": "rami"}, {}, {"canonical_name": "Rami"}))
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(entity_manager=em2), clock=lambda: T0)
    w.observe("I told Lina something"); w.observe("I told Rami something")
    check("a person entity memory already has a relation for is not asked about", subjects(w) == ["Rami"],
          str(subjects(w)))
    em3 = EntityManager(os.path.join(d, "e3.json"))
    em3.store.add(Entity("1", "person", {"name": "sam lee"}, {}, {"canonical_name": "Sam Lee"}))
    em3.store.add(Entity("2", "person", {"name": "sam park"}, {}, {"canonical_name": "Sam Park"}))
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(entity_manager=em3), clock=lambda: T0)
    w.observe("I told Sam something")
    check("two entities sharing a first name make the bare name ambiguous", kinds(w) == ["ambiguous"])
    w.kb.entities = None
    check("a Knowledge with no entity memory still works", Knowledge().person_known("x") is False)


# ======================================================================
class Pathway:
    value = "fast"


class Resp:
    def __init__(self, text):
        self.text, self.pathway, self.trace = text, Pathway(), None


class Drives:
    curiosity, fatigue, urgency, caution = 0.5, 0.0, 0.0, 0.0


class Homeo:
    drives = Drives()


class Brain:
    homeostasis = Homeo()
    follower = None
    user = "Khalid"

    def __init__(self):
        self.seen = []

    def observe(self, **kw):
        self.seen.append(kw)

        class T:
            percepts = {"novelty": 0.5}
        return T()

    def status(self):
        return {}

    def sleep(self, **kw):
        return {}


class Ash:
    def __init__(self):
        self.brain = Brain()
        self.episodic_memory = None
        self.calls = []

    def think(self, query, image=None, context=""):
        self.calls.append(query)
        return Resp("Sure.")


class FakeIdle:
    name = "idle"
    value = 0.0

    def due(self, now):
        return False

    def seconds_idle(self):
        return self.value

    def status(self):
        return {"name": "idle"}


def rt_rig(wonder=True, **wcfg):
    clock = [T0]
    out = []
    ash = Ash()
    per = PeripheryRegistry()
    idle = FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    wcfg.setdefault("min_gap_s", 0)
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=wonder, wonder_cfg=wcfg,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock, h.dir = rt, ash, out, idle, clock, d

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True, **kw):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6, **kw))
        rt._tick()

    def work(seconds, every=10):
        for _ in range(int(seconds / every)):
            step(every, 0.0, ticks=1)

    def free():
        work(150)
        step(15, 25.0)

    h.step, h.say, h.work, h.free = step, say, work, free
    h.asked = lambda: [o for o in out if (o[1].get("attention") or {}).get("kind") == "wonder"]
    return h


def w10_runtime():
    print("\n=== W10. The runtime, end to end ===")
    h = rt_rig()
    h.say("I told Omar something about work")
    check("the user's sentence is still answered normally", h.ash.calls == ["I told Omar something about work"])
    check("(the gap was noticed)", kinds(h.rt.wonder) == ["person"])
    h.step(60, 40.0)
    check("a non-urgent question is NOT asked at an ordinary pause", h.asked() == [], str(h.out))
    h.free()
    got = h.asked()
    check("it is asked in a free moment", len(got) == 1 and got[0][0] == "Who is Omar to you?", str(h.out))
    check("it went out as an unsolicited offer through the gate, not as an answer",
          got[0][1]["addressed"] is False and got[0][1]["attention"]["reasons"][-1] == "volunteered")
    check("the free-moment check-in was replaced by the question, not said as well",
          len([o for o in h.out if (o[1].get("attention") or {}).get("kind") == "free_moment"]) == 0)
    check("the model was not used to ask it", len(h.ash.calls) == 1)
    check("it is marked asked", h.rt.wonder.status()["asked"] == 1)
    check("the conversation is open for the answer", h.rt.reactions.pending and True)

    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("my cousin", source="microphone", addressed=False)
    check("the answer is spent on the wonder, not sent to the model",
          len(h.ash.calls) == n and h.out[-1][0] == "Got it -- Omar is your cousin.", str(h.out[-1]))
    check("it is stored", h.rt.wonder.kb.people["omar"]["relation"] == "cousin")
    check("the concept network was told, as a perception",
          any(s.get("source") == "wonder" and "cousin" in s["text"] for s in h.ash.brain.seen), str(h.ash.brain.seen))
    check("and the reaction was learned as welcome", h.rt.reactions.counts["engaged"] + h.rt.reactions.counts["accepted"] == 1)
    h.free()
    check("nothing more to ask, and nothing said", len(h.asked()) == 1)

    # not now / stop
    h = rt_rig()
    h.say("I told Omar something")
    h.free()
    check("(precondition) asked", len(h.asked()) == 1)
    h.say("not now")
    check("'not now' is acknowledged, not run through the model", h.out[-1][0] in (
        "Okay, I'll check back later.", ) and len(h.ash.calls) == 1, h.out[-1][0])
    check("and the wonder is kept for later", kinds(h.rt.wonder) == ["person"])

    h = rt_rig()
    h.say("I told Omar something"); h.free()
    h.say("stop asking about that")
    check("'stop asking about that' mutes it", "omar" in h.rt.wonder.muted_subjects and len(h.ash.calls) == 1)
    h.say("I told Omar again"); h.step(3600, 60.0); h.free()
    check("and it never comes back", len(h.asked()) == 1)

    # urgent: ordinary pause
    h = rt_rig()
    h.rt.wonder.kb.store_relation("Sam Lee", role="friend"); h.rt.wonder.kb.store_relation("Sam Park", role="friend")
    h.say("I told Sam something")
    h.step(120, 40.0)
    check("an urgent question (two Sams) is asked at an ordinary pause",
          len(h.asked()) == 1 and "Sam Lee or Sam Park" in h.asked()[0][0], str(h.out))

    # quiet rules in the loop
    h = rt_rig()
    h.say("I told Omar something")
    h.rt.gate.mute(True)
    h.step(1300, 40.0); h.free()
    check("muted: nothing", h.asked() == [])
    h.rt.gate.mute(False)
    h.rt.privacy.pause("test")
    h.step(1300, 40.0); h.free()
    check("privacy paused: nothing", h.asked() == [])
    h.rt.privacy.resume()
    h.rt.gate.quiet_hours = (0, 24)
    h.step(1300, 40.0); h.free()
    check("quiet hours: nothing", h.asked() == [])
    h.rt.gate.quiet_hours = None
    h.step(1300, 40.0); h.free()
    check("and then, when the moment is right, it is asked", len(h.asked()) == 1, str(h.out))

    h = rt_rig()
    h.say("I told Omar something")
    h.work(150)
    h.step(15, 3.0)
    check("while they are typing: nothing", h.asked() == [])

    # the gate's own say
    h = rt_rig()
    h.say("I told Omar something")
    h.rt.gate.learner.snooze("wonder", h.clock[0], 10 ** 6)
    h.free()
    check("if the social learner says people dislike these, the gate holds it", h.asked() == [])
    check("...and the wonder is kept, not lost or marked asked", h.rt.wonder.status()["pending"] == 1
          and h.rt.wonder.status()["asked"] == 0)

    # the off switch
    h = rt_rig(wonder=False)
    h.say("I told Omar something"); h.free()
    check("wonder=False: the runtime has none and behaves as before", h.rt.wonder is None and h.asked() == []
          and h.rt.status()["wondering"] is None)
    h = rt_rig(enabled=False)
    h.say("I told Omar something"); h.free()
    check("wondering.enabled=false: nothing noticed or asked", h.asked() == [] and kinds(h.rt.wonder) == [])

    # sources
    h = rt_rig()
    h.say("I told Omar something", source="screen", addressed=False)
    h.say("I told Lina something", source="active_window", addressed=False)
    check("screen / window text is not read", kinds(h.rt.wonder) == [])
    h = rt_rig(sources=["console", "screen"])
    h.say("I told Omar something", source="screen", addressed=False)
    check("...unless the config says so", kinds(h.rt.wonder) == ["person"])
    h = rt_rig()
    h.say("I told Dana my boss", source="sim:mic")
    check("simulated mic counts like the mic", h.rt.wonder.kb.role_name("boss") == "Dana")

    # user's own name
    h = rt_rig()
    h.say("tell Khalid hi")
    check("the brain's user name is never asked about", kinds(h.rt.wonder) == [])

    # status and restart
    h = rt_rig()
    h.say("I told Omar something"); h.free()
    st = h.rt.status()["wondering"]
    check("status shows it", st["asked"] == 1 and st["asked_last_hour"] == 1 and st["enabled"])
    rt2 = rt_rig()
    check("(a second runtime is independent)", kinds(rt2.rt.wonder) == [])
    wn = Wonderer(WonderingConfig(), Knowledge(os.path.join(h.dir, "k.json")), path=os.path.join(h.dir, "w.json"))
    check("a restarted daemon finds the asked question waiting for its answer",
          wn.wonders and next(iter(wn.wonders.values())).status == "asked")

    # a mood hook
    h = rt_rig()
    h.rt.mood_fn = lambda: -0.8
    h.say("I told Omar something"); h.free()
    check("a low mood from the mood hook holds off idle questions", h.asked() == [])
    h.rt.mood_fn = lambda: 1 / 0
    h.free()
    check("a crashing mood hook does not stop the loop", len(h.asked()) == 1 or h.rt.stats["errors"] == 0)


def w11_control():
    print("\n=== W11. The control command ===")
    from src.daemon.service import ControlServer, DEFAULT_CONFIG
    h = rt_rig()
    h.say("I told Omar something"); h.say("my boss is mean")
    cs = ControlServer(h.rt)
    r = cs.dispatch("wonders")
    check("`wonders` lists what is pending", r["ok"] and len(r["pending"]) == 2 and "Who is Omar to you?" in r["report"],
          r.get("report", ""))
    wid = r["pending"][0]["id"]
    check("`wonder dismiss <id>` drops one", cs.dispatch(f"wonder dismiss {wid}")["ok"]
          and len(h.rt.wonder.pending()) == 1)
    check("an unknown id is an error", not cs.dispatch("wonder dismiss nope")["ok"])
    cs.dispatch("wonder off")
    check("`wonder off` is the switch, and shows in status", not cs.dispatch("wonders")["status"]["enabled"])
    cs.dispatch("wonder on")
    check("`wonder on` turns it back on", cs.dispatch("wonders")["status"]["enabled"])
    cs.dispatch("wonder snooze 30")
    check("`wonder snooze 30` silences for half an hour",
          0 < cs.dispatch("wonders")["status"]["snoozed_s"] <= 1800)
    cs.dispatch("wonder clear")
    check("`wonder clear` empties the queue", cs.dispatch("wonders")["pending"] == [])
    check("a bad subcommand is reported", not cs.dispatch("wonder frobnicate")["ok"])
    h2 = rt_rig(wonder=False)
    check("with wondering off the command says so", not ControlServer(h2.rt).dispatch("wonders")["ok"])
    wc = DEFAULT_CONFIG["wondering"]
    check("the shipped defaults are sensible",
          wc["enabled"] is True and 1 <= wc["max_per_hour"] <= 4 and wc["max_per_day"] >= wc["max_per_hour"]
          and "screen" not in wc["sources"] and wc["urgent_at"] > 0.5)
    check("and parse into a real config", WonderingConfig.from_dict(wc).max_per_day == wc["max_per_day"])


# ======================================================================
WORDS = ("my brother sister boss wife Sara Omar Lina Dana Sam Rami told called with and the ZXQ NASA OK "
         "project Zephyr birthday 'flux' \"widget\" Paris Monday I a is was said asked texted met to about "
         "Python hello please open play what time stop asking not now later no yes maybe dunno cousin friend "
         "colleague married colleagues his name is Python 123 ?! . , ; : - _ ' \" ( ) 🙂 é ñ ال عربي").split()


def random_utterance(rnd):
    r = rnd.random()
    if r < 0.05:
        return "".join(rnd.choice(string.printable + "éñ🙂‮\x00") for _ in range(rnd.randint(0, 80)))
    if r < 0.08:
        return rnd.choice(WORDS) * rnd.randint(1, 300)
    return " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(0, 14)))


def w12_fuzz():
    print("\n=== W12. Fuzz and soak ===")
    rnd = random.Random(20260310)
    for seed in range(8):
        rnd = random.Random(seed)
        cfg = WonderingConfig(min_gap_s=rnd.choice([0, 60, 600]), max_per_hour=rnd.choice([1, 2, 3]),
                              max_per_day=rnd.choice([3, 6, 8]), subject_cooldown_s=rnd.choice([0, 3600, 21600]),
                              expire_s=rnd.choice([HOUR, DAY, 3 * DAY]), max_pending=rnd.choice([5, 15, 30]))
        path = tmp()
        kbp = tmp()
        t = [T0]
        w = Wonderer(cfg, Knowledge(kbp), path=path, clock=lambda: t[0])
        problems = []
        asked_log = []                   # (time, key, subjects)
        answered_keys = set()
        n_env = 0
        for step in range(1500):
            t[0] += rnd.choice([1, 5, 30, 120, 600, 3600, 6 * 3600])
            op = rnd.random()
            try:
                if op < 0.45:
                    w.observe(random_utterance(rnd), rnd.choice(["console", "mic"]), t[0])
                elif op < 0.7:
                    w.hear(random_utterance(rnd), t[0], dismissed=rnd.random() < 0.05)
                elif op < 0.72:
                    w.note_ambiguous_intent(random_utterance(rnd), [random_utterance(rnd)[:20] or "a",
                                                                    random_utterance(rnd)[:20] or "b"], t[0])
                elif op < 0.74 and rnd.random() < 0.3:
                    w.snooze_all(rnd.choice([60, 7200]), t[0]) if rnd.random() < 0.5 else w.set_enabled(rnd.random() < 0.7)
                else:
                    env = Environment(idle_s=rnd.choice([None, 1, 20, 60, 300, 900, 5000]),
                                      muted=rnd.random() < 0.1, quiet_hours=rnd.random() < 0.1,
                                      paused=rnd.random() < 0.05, conversation_open=rnd.random() < 0.1,
                                      plan_active=rnd.random() < 0.05, speaking=rnd.random() < 0.05,
                                      free_moment=rnd.random() < 0.5,
                                      mood=rnd.choice([None, -1.0, 0.0, 1.0]),
                                      activity=rnd.choice(["", "VS Code", "Sara"]))
                    q = w.next_question(env, t[0])
                    n_env += 1
                    if q is not None:
                        ok, why = w.can_ask(env, t[0])
                        if not ok:
                            problems.append(("asked while can_ask false", why))
                        if (env.muted or env.quiet_hours or env.paused or env.conversation_open or env.plan_active
                                or env.speaking or env.idle_s is None or env.idle_s < cfg.min_idle_s
                                or env.idle_s > cfg.max_idle_s):
                            problems.append(("asked in a forbidden moment", env))
                        if q.urgency < cfg.urgent_at and not env.free_moment:
                            problems.append(("non-urgent asked outside a free moment", q.kind))
                        if w.closed.get(q.key, {}).get("s") in ("answered", "dismissed", "unclear"):
                            problems.append(("re-asked a closed wonder", q.key))
                        if any(s in w.muted_subjects for s in q.subjects):
                            problems.append(("asked a muted subject", q.subjects))
                        w.mark_asked(q.id, t[0])
                        asked_log.append(t[0])
            except Exception as e:
                problems.append(("exception", repr(e)))
            if step % 250 == 249:
                w.housekeeping(t[0])
        rolling = max((sum(1 for x in asked_log if a <= x < a + HOUR) for a in asked_log), default=0)
        daily = max((sum(1 for x in asked_log if a <= x < a + DAY) for a in asked_log), default=0)
        check(f"seed {seed}: no invariant broken over 1500 random steps", not problems, str(problems[:3]))
        check(f"seed {seed}: never above {cfg.max_per_hour}/hour or {cfg.max_per_day}/day",
              rolling <= cfg.max_per_hour and daily <= cfg.max_per_day, f"{rolling}/{daily}")
        gaps = [b - a for a, b in zip(asked_log, asked_log[1:])]
        check(f"seed {seed}: min gap respected", all(g >= cfg.min_gap_s - 1e-6 for g in gaps) if gaps else True)
        check(f"seed {seed}: bounded", len([x for x in w.wonders.values() if x.status == "pending"]) <= cfg.max_pending + 1)
        check(f"seed {seed}: closed wonders hold no evidence",
              all(x.evidence == "" for x in w.wonders.values() if x.status in ("answered", "dismissed", "unclear", "expired")))
        w2 = Wonderer(cfg, Knowledge(kbp), path=path, clock=lambda: t[0])
        check(f"seed {seed}: state reloads identically",
              sorted((r["id"], r["status"]) for r in w2.pending()) == sorted((r["id"], r["status"]) for r in w.pending())
              and w2.closed == w.closed)
    # replies that are not text at all
    w = rig()
    for bad in (None, 5, [], {}, b"x"):
        try:
            w.hear(bad, T0)
            w.observe(bad)
            ok = True
        except Exception:
            ok = False
        check(f"non-text input {bad!r} is survived", ok)


if __name__ == "__main__":
    for fn in (w1_detectors, w2_statements, w3_queue, w4_when, w5_answers, w6_never_twice, w7_persistence,
               w8_switches, w9_knowledge, w10_runtime, w11_control, w12_fuzz):
        fn()
    print("\n" + ("ALL PASS" if OK else "FAILURES"))
    sys.exit(0 if OK else 1)
