"""
tst_guesses.py -- asking a yes/no guess about someone instead of an open question.

  G1  The company a sentence keeps: work, classes, family
  G2  When a guess is made: enough cues, one kind clearly ahead; otherwise the
      plain question
  G3  Reading the answer: yes, a role, "no, a friend from school", a bare no
      (then the open question), and "family"
  G4  Persistence of what was noticed
  G5  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tests/tst_guesses.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.wondering import (Environment, Knowledge, Wonderer, WonderingConfig,  # noqa: E402
                                  context_of)

check, rig = tw.check, tw.rig
T0 = tw.T0
FREE = dict(idle_s=60.0, free_moment=True)


def ask(w):
    q = w.next_question(Environment(**FREE), w.t[0])
    w.said = None
    if q is not None:
        w.said = w.render(q, w.t[0])
        w.mark_asked(q.id, w.t[0])
    return q


def reply(w, text):
    w.t[0] += 5
    return w.hear(text, w.t[0])


def g1_context():
    print("\n=== G1. The company a sentence keeps ===")
    check("work", context_of("I had a meeting with Sara about the project deadline") == {"work": 3})
    check("classes", context_of("Omar and I did the homework after the lecture") == {"school": 2})
    check("family", context_of("Lina came to the family dinner with my parents") == {"family": 2},
          str(context_of("Lina came to the family dinner with my parents")))
    check("nothing", context_of("I told Sara a joke") == {})


def work_rig(times=1, sentence="I told Sara about the meeting and the project deadline"):
    w = rig()
    for _ in range(times):
        w.observe(sentence)
    return w


def g2_when():
    print("\n=== G2. When a guess is made ===")
    w = work_rig()
    ask(w)
    check("clear work context: a yes/no guess", w.said == "You often mention Sara around work. Is Sara a colleague?",
          str(w.said))
    w = rig()
    w.observe("I told Sara about the meeting")
    ask(w)
    check("one cue is not enough: the plain question", w.said == "Who is Sara to you?", str(w.said))
    w = rig()
    w.observe("I told Sara about the meeting")
    w.observe("I told Sara about the project")
    ask(w)
    check("cues add up over mentions", w.said and w.said.startswith("You often mention Sara around work"),
          str(w.said))
    w = rig()
    w.observe("I told Sara about the meeting and the lecture")
    w.observe("I told Sara about the project and the homework")
    ask(w)
    check("mixed company: no guess", w.said == "Who is Sara to you?", str(w.said))
    w = rig()
    w.observe("I told Omar about the homework and the exam")
    ask(w)
    check("classes", w.said == "You often mention Omar around your classes. Is Omar a classmate?", str(w.said))
    w = rig()
    w.observe("I told Lina about the family dinner with my parents")
    ask(w)
    check("family has no article", w.said == "You often mention Lina with your family. Is Lina family?", str(w.said))
    check("no pronoun is ever guessed", all(p not in (w.said or "").split() for p in ("he", "she", "him", "her")))


def g3_answers():
    print("\n=== G3. Reading the answer ===")
    w = work_rig()
    ask(w)
    r = reply(w, "yes")
    check("yes stores the guess", r.outcome == "answered" and w.kb.relation_of("Sara") == "colleague", str(r))
    check("and is learned", r.learned == ["Sara is the user's colleague."])
    w = work_rig()
    ask(w)
    reply(w, "yeah, she's my manager actually")
    check("a role in the reply wins over the guess", w.kb.relation_of("Sara") == "manager")
    w = work_rig()
    ask(w)
    reply(w, "no, a friend from school")
    check("'no, a friend ...' records the relation it names", w.kb.relation_of("Sara") == "friend",
          str(w.kb.people.get("sara")))
    w = work_rig()
    ask(w)
    reply(w, "no, just someone from the gym")
    check("'no, <description>' keeps the description", w.kb.people["sara"]["desc"] == "from the gym"
          and w.kb.person_known("Sara"), str(w.kb.people.get("sara")))
    w = work_rig()
    ask(w)
    r = reply(w, "no")
    check("a bare no asks the open question", r.outcome == "followup" and r.say == "Okay -- who is Sara to you, then?",
          r.say)
    reply(w, "my cousin")
    check("and takes its answer", w.kb.relation_of("Sara") == "cousin")
    w = rig()
    w.observe("I told Lina about the family dinner with my parents")
    ask(w)
    reply(w, "yep")
    check("family: kept as a description, and Lina is known", w.kb.people["lina"]["desc"] == "family"
          and w.kb.person_known("Lina"))
    w = work_rig()
    ask(w)
    r = reply(w, "I don't know")
    check("'I don't know' lets it go", r.outcome == "dismissed")


def g4_persist():
    print("\n=== G4. Persistence ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "w.json")
    t = [T0]
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=p, clock=lambda: t[0])
    w.observe("I told Sara about the meeting")
    w2 = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), path=p, clock=lambda: t[0])
    w2.observe("I told Sara about the project")
    w2.t = t
    ask(w2)
    check("what was noticed survives a restart", w2.said and w2.said.startswith("You often mention Sara"),
          str(w2.said))


def g5_runtime():
    print("\n=== G5. The runtime ===")
    import tst_beliefs as tb
    h = tb.rt_rig()
    h.say("I told Sara about the meeting and the project deadline")
    h.free()
    check("asked as a guess", "You often mention Sara around work. Is Sara a colleague?" in h.said(), str(h.said()))
    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("yes", source="microphone", addressed=False)
    check("a yes is spent on it and stored", len(h.ash.calls) == n and h.rt.wonder.kb.relation_of("Sara") == "colleague",
          str(h.said()))


if __name__ == "__main__":
    for fn in (g1_context, g2_when, g3_answers, g4_persist, g5_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
