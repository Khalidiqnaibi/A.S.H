"""
tst_asking.py -- which question first, and asking for them.

  A1  Usefulness: a subject that comes back day after day outranks one said
      in a burst; a recent one outranks a stale one
  A2  "What are you wondering about?": answered there and then, one question
      after another riding on each acknowledgement, without the pacing caps;
      muted and closed ones never; "that's all" ends it; nothing open says so
  A3  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tst_asking.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check, rig, kinds = tw.check, tw.rig, tw.kinds
T0, DAY, HOUR = tw.T0, tw.DAY, tw.HOUR
FREE = dict(idle_s=60.0, free_moment=True)


def first(w):
    q = w.next_question(Environment(**FREE), w.t[0])
    return q.subject if q else None


def a1_usefulness():
    print("\n=== A1. Usefulness ===")
    w = rig()
    for d in range(3):
        w.t[0] = T0 + d * DAY
        w.observe("I told Omar about it")
    w.observe("I told Sara about it")
    w.observe("I told Sara again")
    w.observe("Sara called me")
    check("(same number of mentions)", w.mentions["person:omar"]["n"] == w.mentions["person:sara"]["n"] == 3)
    check("the one that came back over days is asked first", first(w) == "Omar", str(first(w)))
    check("its usefulness is higher", w.usefulness(w.wonders[[k for k, x in w.wonders.items()
                                                            if x.subject == "Omar"][0]], w.t[0]) >
          w.usefulness(w.wonders[[k for k, x in w.wonders.items() if x.subject == "Sara"][0]], w.t[0]))

    w = rig()
    w.observe("I told Omar about it")
    w.t[0] = T0 + 2.5 * DAY
    w.observe("I told Sara about it")
    check("one from today outranks one from days ago", first(w) == "Sara", str(first(w)))
    w = rig(expire_s=10 * DAY)
    w.observe("I told Omar about it")
    w.t[0] = T0 + 3.5 * DAY
    w.observe("I told Sara about it")
    w.t[0] = T0 + 5 * DAY
    check("one not heard of for days drops behind", first(w) == "Sara", str(first(w)))
    w.t[0] = T0 + 5 * DAY
    d = w.mentions["person:sara"]["days"]
    check("days are kept, bounded", len(d) == 1)
    for i in range(30):
        w.t[0] += DAY
        w.observe("I told Sara about it")
    check("(at most ten days kept)", len(w.mentions["person:sara"]["days"]) == 10)


def opened():
    w = rig()
    w.observe("I told Omar about it")
    w.observe("We should move the jobs to ZXQ")
    w.observe("ZXQ is slow")
    return w


def a2_batch():
    print("\n=== A2. What are you wondering about? ===")
    w = opened()
    r = w.hear("what are you wondering about?", T0 + 5)
    check("answered there and then", r.consumed and r.outcome == "batch"
          and r.say.startswith("A few things. First: "), r.say)
    first_q = r.say.split("First: ", 1)[1]
    check("with the best question", first_q in ("Who is Omar to you?", "What does ZXQ stand for?"), first_q)
    ans = "my cousin" if "Omar" in first_q else "zone exchange queue"
    r = w.hear(ans, T0 + 10)
    check("the answer is taken", r.outcome == "answered")
    check("and the next question rides on the acknowledgement", " Next: " in r.say, r.say)
    ans = "my cousin" if "Omar" in r.say.split(" Next: ")[1] else "zone exchange queue"
    r = w.hear(ans, T0 + 15)
    check("after the last, it says it is done", r.say.endswith("That's everything I was wondering about."), r.say)
    check("both stored", w.kb.relation_of("Omar") == "cousin" and w.kb.term_known("ZXQ"))

    w = opened()
    w.cfg.max_per_hour = 0
    w.cfg.min_gap_s = 99999
    r = w.hear("do you have any questions for me", T0 + 5)
    check("the pacing caps do not apply when asked", r.outcome == "batch" and "First:" in r.say, r.say)

    w = opened()
    w.hear("stop asking about Omar", T0 + 1)
    r = w.hear("any questions?", T0 + 5)
    check("a muted subject is never offered", "Omar" not in r.say and "ZXQ" in r.say, r.say)

    w = rig()
    r = w.hear("what are you wondering about?", T0 + 5)
    check("nothing open: says so", r.consumed and r.say.startswith("Nothing right now"), r.say)

    w = opened()
    w.hear("what are you wondering about", T0 + 5)
    r = w.hear("that's all for now", T0 + 10)
    check("'that's all' ends it", r.outcome == "batch_end" and r.say == "Okay, that's enough for now.")
    check("and the open question is kept for later", sorted(kinds(w)) == ["person", "term"]
          and all(x["status"] == "pending" for x in w.pending()), str(w.pending()))

    w = opened()
    w.hear("what are you wondering about", T0 + 5)
    r = w.hear("my cousin", T0 + 20 * 60 + 5)
    check("an answer after the batch window does not chain", " Next: " not in (r.say or ""), r.say)

    w = opened()
    w.set_enabled(False)
    r = w.hear("what are you wondering about", T0 + 5)
    check("wondering off: not taken", not r.consumed or r.outcome != "batch")

    w = opened()
    r = w.hear("so what were you wondering about when you looked at my screen earlier while I was out at lunch",
               T0 + 5)
    check("a long sentence that mentions it is not a request", r.outcome != "batch", r.say)

    w = rig()
    w.observe("I told Sara about it")
    b = w.raise_wonder("between", "Sara and Omar", "How do Sara and Omar know each other?", "between:omar|sara",
                       data={"a": "Sara", "b": "Omar"})
    b.subjects = ["sara", "omar"]
    w.mark_asked(b.id, w.t[0])
    w.hear("stop asking about that", w.t[0] + 5)
    check("(the question about Sara herself is still open)", any(x["subject"] == "Sara" for x in w.pending()))
    r = w.hear("what are you wondering about", w.t[0] + 10)
    check("muting it mutes both people for the batch too", "Sara" not in r.say and "Omar" not in r.say, r.say)

    w = rig()
    w.observe("I told Sara about the meeting and the project deadline")
    r = w.hear("what are you wondering about", T0 + 5)
    check("a batch question uses the guess too", r.say == "You often mention Sara around work. Is Sara a colleague?",
          r.say)

    w = opened()
    w.hear("what are you wondering about", T0 + 5)
    w._batch_until = T0 + 6
    r = w.hear("my cousin" if "Omar" in w.render(w.awaiting(T0 + 7), T0 + 7) else "zone exchange queue", T0 + 7)
    check("once the batch has closed, answers do not chain", r.outcome == "answered" and " Next: " not in r.say, r.say)


def a3_runtime():
    print("\n=== A3. The runtime ===")
    import tst_beliefs as tb
    h = tb.rt_rig()
    h.say("I told Omar about it")
    n = len(h.ash.calls)
    h.say("what are you wondering about?")
    check("asked for, without the model", len(h.ash.calls) == n and h.said()[-1] == "Who is Omar to you?",
          str(h.said()))
    h.say("my cousin", source="microphone", addressed=False)
    check("and answered", h.rt.wonder.kb.relation_of("Omar") == "cousin" and
          h.said()[-1].startswith("Got it -- Omar is your cousin."), str(h.said()))


if __name__ == "__main__":
    for fn in (a1_usefulness, a2_batch, a3_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
