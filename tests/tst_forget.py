"""
tst_forget.py -- "undo that" and "forget what I said about Omar".

  U1  Reading the requests, and what is not one
  U2  The undo stack: newest first, within ten minutes, a failing step
  U3  Forgetting a subject: people, relations between them, roles, aliases,
      terms, routines, entity memory, open questions; nothing about it said
  U4  The runtime: undo a learned relation (and its entity), a definition, a
      follow-up, a reminder, a watch, a correction; forget a person; a guest
      cannot do either

Run:  PYTHONIOENCODING=utf-8 python tests/tst_forget.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.forget import UndoStack, forget_subject, wants_forget, wants_undo  # noqa: E402
from src.daemon.wondering import Knowledge  # noqa: E402

check = tw.check
T0 = tw.T0


def u1_reading():
    print("\n=== U1. Reading the requests ===")
    for text in ("undo that", "undo", "scratch that", "forget that", "never mind that", "hey ash, take that back"):
        check(f"undo: {text!r}", wants_undo(text))
    for text in ("undo the last commit in git please", "forget it", "", None):
        check(f"not undo: {str(text)[:30]!r}", not wants_undo(text))
    for text, want in (("forget what I said about Omar", "Omar"), ("forget Omar", "Omar"), ("forget ZXQ", "ZXQ"),
                       ("forget my morning check", "morning check"), ("please forget everything about Sara", "Sara"),
                       ("forget about the dentist", "dentist")):
        check(f"forget: {text!r}", wants_forget(text) == want, repr(wants_forget(text)))
    for text in ("forget it", "forget that", "forget my voice", "forget everything", "don't forget the milk", ""):
        check(f"not a subject to forget: {text!r}", wants_forget(text) is None)


def u2_stack():
    print("\n=== U2. The undo stack ===")
    t = [T0]
    s = UndoStack(clock=lambda: t[0])
    done = []
    s.push("first", lambda: done.append(1))
    s.push("second", lambda: done.append(2))
    check("newest first", s.undo() == "Undone: second." and done == [2])
    check("then the one before", s.undo() == "Undone: first." and done == [2, 1])
    check("then nothing", s.undo() == "There's nothing recent to undo.")
    s.push("old", lambda: done.append(3))
    t[0] += 700
    check("too old to undo", s.undo() == "There's nothing recent to undo." and done == [2, 1])
    s.push("bad", lambda: 1 / 0)
    check("a failing step says so", s.undo() == "I couldn't undo that one, sorry.")
    for i in range(30):
        s.push(str(i), lambda: None)
    check("bounded", len(s) == 20)


class Ent:
    """A stand-in for entity memory: an EntityManager-shaped store."""
    def __init__(self):
        from src.memory.entity.entity_manager import EntityManager
        self.m = EntityManager(os.path.join(tempfile.mkdtemp(), "e.json"))

    def __getattr__(self, k):
        return getattr(self.m, k)


def u3_forget():
    print("\n=== U3. Forgetting a subject ===")
    kb = Knowledge(entity_manager=None)
    kb.store_relation("Omar", role="cousin")
    kb.store_relation("Sara", role="boss")
    kb.store_between("Omar", "Sara", "friends")
    kb.store_alias("Om", "Omar")
    kb.store_term("ZXQ", "zone exchange queue")
    kb.store_procedure("morning check", ["Outlook", "Teams"])
    w = tw.rig()
    w.kb = kb
    w.observe("I told Omar about it")
    w.raise_wonder("between", "Omar and Lina", "How do Omar and Lina know each other?", "between:lina|omar",
                   data={"a": "Omar", "b": "Lina"}).subjects = ["omar", "lina"]
    gone = forget_subject("Omar", kb=kb, wonderer=w)
    check("the person is gone", "omar" not in kb.people and "who they are" in gone)
    check("and how they know others", "omar" not in kb.people["sara"]["between"] and "how they know others" in gone)
    check("their alias", "om" not in kb.aliases)
    check("questions about them are closed", not any(x.status in ("pending", "asked") and "omar" in x.subjects
                                                     for x in w.wonders.values()) and "the questions about it" in gone)
    check("Sara is untouched", kb.relation_of("Sara") == "boss" and kb.role_name("boss") == "Sara")
    gone = forget_subject("Sara", kb=kb)
    check("a role goes with the person", kb.role_name("boss") is None)
    check("a term", forget_subject("ZXQ", kb=kb) == ["what it means"] and not kb.term_known("ZXQ"))
    check("a routine", forget_subject("morning check", kb=kb) == ["the routine"] and kb.procedures == {})
    check("nothing to forget", forget_subject("Nobody", kb=kb) == [])
    ent = Ent()
    kb2 = Knowledge(entity_manager=ent)
    kb2.store_relation("Omar", role="cousin")
    check("(mirrored into entity memory)", len(ent.store.all()) == 1)
    gone = forget_subject("Omar", kb=kb2, entities=ent)
    check("entity memory forgets too", ent.store.all() == [] and "the memory entry" in gone)


def rt_rig():
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    import tst_watch as twa
    clock = [T0]
    out = []
    ash = tw.Ash()
    ash.entity_memory = Ent()
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, wonder=True,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json"),
                             followups=True, reminders=True, watch=True, watch_cfg={"disk_trend": False}),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.followups.clock = rt.reminders.clock = lambda: clock[0]
    rt.watcher.probe = twa.FakeProbe()
    rt.watcher.clock = lambda: clock[0]

    def say(text, **meta):
        clock[0] += 2
        rt.submit(SensorEvent(source="console" if not meta else "microphone", modality=Modality.TEXT, text=text,
                              addressed=True, salience=0.9, urgency=0.6, meta=meta))
        rt._tick()
    return rt, ash, out, say


def u4_runtime():
    print("\n=== U4. The runtime ===")
    rt, ash, out, say = rt_rig()
    say("Omar is my cousin")
    check("(learned, and mirrored)", rt.wonder.kb.relation_of("Omar") == "cousin"
          and len(ash.entity_memory.store.all()) == 1)
    n = len(ash.calls)
    say("undo that")
    check("undone, without the model", out[-1] == "Undone: Omar is your cousin." and len(ash.calls) == n, out[-1])
    check("the relation is gone", rt.wonder.kb.relation_of("Omar") == "")
    check("and the entity it created", ash.entity_memory.store.all() == [])
    say("Omar is my cousin")
    say("Lina is my sister")
    say("undo that")
    check("only the latest is undone", rt.wonder.kb.relation_of("Lina") == "" and
          rt.wonder.kb.relation_of("Omar") == "cousin")

    rt, ash, out, say = rt_rig()
    say("I have an interview tomorrow at 10am")
    say("undo that")
    check("a follow-up noted is dropped", out[-1] == "Undone: noting your interview." and
          rt.followups.status()["open"] == 0, out[-1])
    say("set a timer for 5 minutes")
    say("scratch that")
    check("a timer is cancelled", out[-1] == "Undone: the 5-minute timer." and rt.reminders.status()["active"] == 0, out[-1])
    say("remind me to call mom in 20 minutes")
    say("undo")
    check("a reminder is cancelled", out[-1] == "Undone: the reminder to call mom.", out[-1])
    say("tell me when chrome closes")
    say("undo that")
    check("a watch is cancelled", out[-1] == "Undone: watching for chrome." and rt.watcher.active() == [], out[-1])
    say("undo that")
    check("nothing left", out[-1] == "There's nothing recent to undo.")

    rt, ash, out, say = rt_rig()
    from src.brain import corrections
    old = corrections.default()
    try:
        corrections.configure(None)
        rt._last_turn = {"query": "what's the weather like in my code", "intent": "get_weather", "tool": None,
                         "t": rt.clock()}
        say("that's not what I asked")
        check("(a correction kept)", len(corrections.default().items) == 1)
        say("undo that")
        check("a correction is dropped", corrections.default().items == [] and out[-1] == "Undone: the correction.")
    finally:
        corrections._DEFAULT = old

    rt, ash, out, say = rt_rig()
    say("Omar is my cousin")
    say("I told Omar about the trip")
    say("forget what I said about Omar")
    check("forgotten", out[-1].startswith("Forgotten: who they are") and rt.wonder.kb.relation_of("Omar") == "", out[-1])
    check("from entity memory too", ash.entity_memory.store.all() == [])
    say("forget Nobody")
    check("nothing to forget: said so", out[-1] == "I don't have anything about Nobody to forget.")

    rt, ash, out, say = rt_rig()
    say("Omar is my cousin")
    say("forget Omar", guest=True)
    check("a guest cannot forget", rt.wonder.kb.relation_of("Omar") == "cousin" and "only do that for" in out[-1])
    say("undo that", guest=True)
    check("nor undo", rt.wonder.kb.relation_of("Omar") == "cousin")


if __name__ == "__main__":
    for fn in (u1_reading, u2_stack, u3_forget, u4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
