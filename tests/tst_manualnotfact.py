"""
tst_manualnotfact.py -- ASH's own manual is never a source of what the user said.

Live (25 Nov): "who is lieth" -> "From what you've said before: 'ASH manual, memory. "Remember Lieth is my friend" ...'"; and the knowledge file got
brother = Omar from the manual's example "remember my brother is Omar" (the user: "my brother is not omar where did you get that").

Run:  PYTHONIOENCODING=utf-8 python tests/tst_manualnotfact.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check

MANUAL = ('ASH manual, memory. "Remember Lieth is my friend", "remember my brother is Omar": parsed and stored as people and relations, '
          'and the reply says exactly what was stored.')


def m1_sentences():
    print("\n=== M1. What counts as 'said by the user' ===")
    import tst_effects as te
    rt = te.rig(wonder=True)
    dump = {"manual_memory": {"rule_id": "manual_memory", "category": "capability", "text": MANUAL},
            "manual_x": {"rule_id": "manual_x", "category": "constraint", "text": "remember my sister is Dana"},
            "core_user": {"rule_id": "core_user", "category": "standard", "text": "my favorite number is 4242"},
            "core_policy": {"rule_id": "core_policy", "category": "constraint", "text": "ASH must not invent facts."}}
    rt.ash.core_memory = NS(dump_all=lambda: dump)
    rt.ash.episodic_memory = NS(store=NS(episodes={}))
    got = rt._user_sentences()
    check("the manual is not in it, by id or by category", not any("Omar" in s or "Dana" in s for s in got), str(got))
    check("what the user asked to keep still is", "my favorite number is 4242" in got, str(got))


def m2_no_fact():
    print("\n=== M2. The manual's example does not become a fact ===")
    import tst_effects as te
    rt = te.rig(wonder=True)
    rt.ash.core_memory = NS(dump_all=lambda: {"manual_memory": {"rule_id": "manual_memory", "category": "capability", "text": MANUAL}})
    rt.ash.episodic_memory = NS(store=NS(episodes={}))
    w = rt.wonder
    w.memory_scan = rt._user_sentences
    q = None
    w.observe("my brother helped me with it yesterday")
    for r in w.pending():
        if r["kind"] == "role" and "brother" in r["question"].lower():
            q = r
    from src.daemon.wondering import Wonder
    wd = Wonder(id="r1", kind="role", subject="brother", question="What's your brother's name?", key="role:brother", subjects=["brother"], data={"role": "brother"})
    w.wonders["r1"] = wd
    got = w._from_memory(wd, w.clock())
    check("a role question is not answered from the manual", got is False and w.kb.role_name("brother") in (None, ""), str(w.kb.roles))
    check("nothing about Omar was stored", "omar" not in w.kb.people, str(list(w.kb.people)))


if __name__ == "__main__":
    m1_sentences()
    m2_no_fact()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
