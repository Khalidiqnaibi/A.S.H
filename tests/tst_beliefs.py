"""
tst_beliefs.py -- questions about what ASH already knows.

  B1  The rules: which two relations cannot both be true, which two
      definitions say different things
  B2  Contradicting relations: kept, not overwritten, and asked about
  B3  A replaced role: "Is Sam your boss now, instead of Dana?"
  B4  Contradicting definitions and dates
  B5  Stale relations: "Is Dana still your boss?" when Dana comes up again
      after a long while; yes / no / "she's my friend now" / not anymore
  B6  Persistence, and old knowledge files with no timestamps
  B7  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tests/tst_beliefs.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.wondering import (Environment, Knowledge, Wonderer, WonderingConfig,  # noqa: E402
                                  definitions_differ, relations_conflict)
from src.senses.base import PeripheryRegistry  # noqa: E402

check, rig, kinds = tw.check, tw.rig, tw.kinds
T0, HOUR, DAY = tw.T0, tw.HOUR, tw.DAY
FREE = dict(idle_s=60.0, free_moment=True)


def ask(w):
    q = w.next_question(Environment(**FREE), w.t[0])
    w.said = None
    if q is not None:
        w.said = w.render(q, w.t[0])
        w.mark_asked(q.id, w.t[0])
    return q


def reply(w, text, dt=5):
    w.t[0] += dt
    return w.hear(text, w.t[0])


def b1_rules():
    print("\n=== B1. The rules ===")
    check("cousin vs brother conflict", relations_conflict("cousin", "brother"))
    check("family vs not family conflict", relations_conflict("brother", "colleague"))
    check("colleague vs friend do not (both can be true)", not relations_conflict("colleague", "friend"))
    check("girlfriend vs fiance do not (one became the other)", not relations_conflict("girlfriend", "fiance"))
    check("sibling vs brother do not (one is the general word)", not relations_conflict("sibling", "brother")
          and not relations_conflict("brother", "sibling"))
    check("child vs daughter do not", not relations_conflict("child", "daughter"))
    check("the same, or nothing before, do not", not relations_conflict("cousin", "cousin")
          and not relations_conflict("", "cousin"))
    check("different definitions differ", definitions_differ("zone exchange queue", "the zero copy buffer"))
    check("the same reworded does not", not definitions_differ("zone exchange queue", "the zone exchange queue"))
    check("a fuller version does not", not definitions_differ("exchange queue", "the zone exchange queue we built"))
    check("mostly the same words do not", not definitions_differ("our deploy tool for staging",
                                                                 "the deploy tool for staging"))


def b2_relations():
    print("\n=== B2. Contradicting relations ===")
    w = rig()
    w.observe("Omar is my cousin.")
    check("(learned)", w.kb.relation_of("Omar") == "cousin")
    w.observe("Omar is my brother.")
    check("not overwritten", w.kb.relation_of("Omar") == "cousin")
    check("a conflict is noted", kinds(w) == ["conflict"])
    q = ask(w)
    check("and asked plainly", w.said == "Earlier you said Omar is your cousin, but just now your brother. Which is it?",
          str(w.said))
    r = reply(w, "brother, I misspoke")
    check("the answer is taken", r.outcome == "answered" and w.kb.relation_of("Omar") == "brother", str(r))
    check("and learned", r.learned == ["Omar is the user's brother."])

    w = rig()
    w.observe("Omar is my cousin."); w.observe("Omar is my brother.")
    ask(w)
    reply(w, "the first one")
    check("'the first one' keeps the old", w.kb.relation_of("Omar") == "cousin")

    w = rig()
    w.observe("Omar is my cousin."); w.observe("Omar is my brother.")
    ask(w)
    r = reply(w, "cousin")
    check("naming the old keeps it", w.kb.relation_of("Omar") == "cousin" and r.outcome == "answered")

    w = rig()
    w.observe("Sara is my colleague."); w.observe("Sara is my friend.")
    check("a relation that can change is replaced without a question",
          w.kb.relation_of("Sara") == "friend" and kinds(w) == [])

    w = rig()
    w.observe("Omar is my cousin."); w.observe("Omar is my cousin.")
    check("said twice is not a conflict", kinds(w) == [])

    w = rig()
    w.observe("Lina is my sibling."); w.observe("Lina is my sister.")
    check("the general word then the exact one is not a conflict", kinds(w) == []
          and w.kb.relation_of("Lina") == "sister")

    w = rig()
    w.observe("Omar is my cousin."); w.observe("Omar is my brother.")
    ask(w)
    reply(w, "both actually")
    check("'both' keeps the new and remembers the old", w.kb.relation_of("Omar") == "brother"
          and "cousin" in w.kb.people["omar"]["desc"])

    w = rig()
    w.observe("Omar is my cousin."); w.observe("Omar is my brother.")
    ask(w)
    r = reply(w, "hmm")
    check("an unclear reply gets the one follow-up", r.outcome == "followup" and "cousin or brother" in r.say,
          r.say)
    w.t[0] += 2 * HOUR
    w.hear("my brother Omar is visiting this weekend", w.t[0])
    check("a later mention is not read as the answer",
          w.kb.relation_of("Omar") == "cousin" and kinds(w) == ["conflict"], str(kinds(w)))


def b3_roles():
    print("\n=== B3. A replaced role ===")
    w = rig()
    w.observe("my boss Dana is away")
    check("(learned)", w.kb.role_name("boss") == "Dana")
    w.observe("my boss Sam wants the report")
    check("the new name is not stored yet", w.kb.role_name("boss") == "Dana" and w.kb.relation_of("Sam") == "")
    ask(w)
    check("asked as a change", w.said == "Is Sam your boss now, instead of Dana?", str(w.said))
    r = reply(w, "yes, Dana left")
    check("yes: Sam is the boss now", w.kb.role_name("boss") == "Sam" and w.kb.relation_of("Sam") == "boss", str(r))
    check("and Dana is remembered as the former one", w.kb.relation_of("Dana") == ""
          and w.kb.people["dana"]["desc"] == "your former boss")

    w = rig()
    w.observe("my boss Dana is away"); w.observe("my boss Sam wants the report")
    ask(w)
    r = reply(w, "no, Dana is still my boss")
    check("no: Dana stays", w.kb.role_name("boss") == "Dana" and r.outcome == "answered", str(r))

    w = rig()
    w.observe("my brother Omar is here"); w.observe("my brother Ali is here")
    check("a second brother is not a replacement", kinds(w) == [] and w.kb.relation_of("Ali") == "brother"
          and w.kb.relation_of("Omar") == "brother")


def b4_terms_facts():
    print("\n=== B4. Contradicting definitions and dates ===")
    w = rig()
    w.observe("ZXQ means zone exchange queue")
    check("(learned)", w.kb.terms["zxq"]["definition"] == "zone exchange queue")
    w.observe("ZXQ means the zero copy buffer")
    check("kept, and asked about", w.kb.terms["zxq"]["definition"] == "zone exchange queue"
          and kinds(w) == ["conflict"])
    ask(w)
    check("asked plainly", w.said == "You told me ZXQ means zone exchange queue, and now the zero copy buffer. "
                                     "Which is right?", str(w.said))
    reply(w, "the zero copy buffer one")
    check("the chosen one is stored", w.kb.terms["zxq"]["definition"] == "the zero copy buffer")

    w = rig()
    w.observe("ZXQ means zone exchange queue")
    w.observe("ZXQ means the zone exchange queue")
    check("the same definition reworded: no question", kinds(w) == [])

    w = rig()
    w.observe("my birthday is on 12 March")
    w.observe("my birthday is on 14 March")
    check("two dates: kept, and asked", w.kb.fact("birthday") == "12 March" and kinds(w) == ["conflict"])
    ask(w)
    reply(w, "14 March, sorry")
    check("the chosen date is stored", w.kb.fact("birthday") == "14 March")


def b5_stale():
    print("\n=== B5. Stale relations ===")
    w = rig()
    w.observe("my boss Dana is away")
    w.t[0] += 30 * DAY
    w.observe("I told Dana about the plan")
    check("a recent relation is not checked", kinds(w) == [])
    w.t[0] = T0 + 200 * DAY
    w.observe("I told Dana about the plan")
    check("an old changeable one is, when Dana comes up", kinds(w) == ["stale"])
    q = ask(w)
    check("asked as a check", w.said == "Is Dana still your boss?", str(w.said))
    check("(not urgent: only at a free moment)", q.urgency < w.cfg.urgent_at)
    r = reply(w, "yes")
    check("yes confirms", r.outcome == "answered" and w.kb.relation_age("Dana", w.t[0]) < 60)
    w.t[0] += 10 * DAY
    w.observe("I told Dana about it again")
    check("confirmed: not asked again soon", kinds(w) == [])
    w.t[0] += 200 * DAY
    w.observe("I told Dana about it again")
    check("but checked again after another long while", kinds(w) == ["stale"])

    def staled():
        w = rig()
        w.observe("my boss Dana is away")
        w.t[0] = T0 + 200 * DAY
        w.observe("I told Dana about the plan")
        ask(w)
        return w

    w = staled()
    reply(w, "no, she's my friend now")
    check("'no, she's my friend now' records the new relation", w.kb.relation_of("Dana") == "friend")
    w = staled()
    reply(w, "not anymore, she left")
    check("'not anymore' retires her", w.kb.relation_of("Dana") == "" and "former boss" in w.kb.people["dana"]["desc"]
          and w.kb.role_name("boss") is None)
    w = staled()
    r = reply(w, "no")
    check("a bare no asks what she is now", r.outcome == "followup" and r.say == "Okay -- what is Dana to you now?",
          r.say)
    reply(w, "a friend")
    check("and takes the answer", w.kb.relation_of("Dana") == "friend")
    w = staled()
    reply(w, "no")
    r = reply(w, "no")
    check("no twice: she is no longer the boss", r.outcome == "answered"
          and "former boss" in w.kb.people["dana"]["desc"], str(r))
    w = staled()
    reply(w, "she's still my boss")
    check("'still' confirms", w.kb.relation_of("Dana") == "boss" and w.kb.relation_age("Dana", w.t[0]) < 60)

    w = rig()
    w.observe("Omar is my cousin.")
    w.t[0] = T0 + 900 * DAY
    w.observe("I told Omar about it")
    check("family is never stale", kinds(w) == [])

    w = rig(stale_after_s=10 * DAY)
    w.observe("my boss Dana is away")
    w.t[0] = T0 + 11 * DAY
    w.observe("I told Dana")
    check("the age is configurable", kinds(w) == ["stale"])

    w = rig(kinds={"stale": False})
    w.observe("my boss Dana is away")
    w.t[0] = T0 + 400 * DAY
    w.observe("I told Dana")
    check("and the kind can be switched off", kinds(w) == [])


def b6_persist():
    print("\n=== B6. Persistence ===")
    d = tempfile.mkdtemp()
    kp = os.path.join(d, "k.json")
    t = [T0]
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(kp), clock=lambda: t[0])
    w.observe("my boss Dana is away")
    raw = json.load(open(kp, encoding="utf-8"))
    check("the time a relation was told is kept", raw["people"]["dana"]["t"] == T0)
    t[0] = T0 + 200 * DAY
    w2 = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(kp), clock=lambda: t[0])
    w2.observe("I told Dana")
    check("and survives a restart", kinds(w2) == ["stale"])

    legacy = os.path.join(d, "old.json")
    json.dump({"people": {"dana": {"name": "Dana", "relation": "boss", "desc": "", "between": {}}},
               "roles": {"boss": "Dana"}, "terms": {}, "facts": {}, "aliases": {}}, open(legacy, "w"))
    t[0] = T0
    w3 = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(legacy), clock=lambda: t[0])
    w3.observe("I told Dana")
    check("an old file with no times is not all stale at once", kinds(w3) == [])
    check("its clock starts when the person comes up", w3.kb.relation_age("Dana", T0) == 0)
    t[0] = T0 + 200 * DAY
    w3.observe("I told Dana")
    check("and is checked once enough time has passed", kinds(w3) == ["stale"])


def rt_rig():
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=True, wonder_cfg={"min_gap_s": 0},
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(tw.SensorEvent(source=source, modality=tw.Modality.TEXT, text=text, addressed=addressed,
                                 salience=0.9, urgency=0.6))
        rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0, ticks=1)
        step(15, 25.0)

    h.step, h.say, h.free = step, say, free
    h.said = lambda: [o[0] for o in out]
    return h


def b7_runtime():
    print("\n=== B7. The runtime ===")
    h = rt_rig()
    h.say("Omar is my cousin")
    h.say("Omar is my brother by the way")
    check("both sentences are answered normally", len(h.ash.calls) == 2)
    h.free()
    check("at a free moment ASH asks which", any("Which is it?" in s for s in h.said()), str(h.said()))
    n = len(h.ash.calls)
    h.step(5, 5.0)
    h.say("brother", source="microphone", addressed=False)
    check("the answer is spent on the question", len(h.ash.calls) == n and h.said()[-1] == "Got it -- brother.",
          str(h.said()))
    check("stored", h.rt.wonder.kb.relation_of("Omar") == "brother")
    check("and the concept network was told",
          any(s.get("source") == "wonder" and "brother" in s["text"] for s in h.ash.brain.seen))


if __name__ == "__main__":
    for fn in (b1_rules, b2_relations, b3_roles, b4_terms_facts, b5_stale, b6_persist, b7_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
