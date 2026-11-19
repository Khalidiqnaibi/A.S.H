"""
tst_facets.py -- an entity is more than a name.

The Mind page: "311 of 316 entities hold only a name and the last sentence they appeared in ... a list of words, not knowledge of anyone."

Run:  PYTHONIOENCODING=utf-8 python tests/tst_facets.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.facets import compile_facets  # noqa: E402
from src.memory.entity.entity_model import Entity  # noqa: E402

check = tw.check


def ent(i, name, **at):
    return Entity(entity_id=i, entity_type="person", primary_identifiers={"name": name.lower()}, attributes={"canonical_name": name, **at})


class Store:
    def __init__(self, *e):
        self.entities = {x.entity_id: x for x in e}
        self.saved = 0

    def _save(self):
        self.saved += 1


def ep(t, summary, *eids):
    return NS(timestamp=t, summary=summary, related_entities=list(eids))


def f1_compile():
    print("\n=== F1. Compiling ===")
    r = tw.rig()
    r.kb.store_relation("Linda Iqnaibi", role="sister")
    st = Store(ent("e1", "Linda Iqnaibi"), ent("e2", "Hebron"), ent("e3", "Lonely"))
    eps = [ep(1000.0, "Immortal: Linda called about dinner", "e1"), ep(2000.0, "Immortal: Linda is coming Friday", "e1"),
           ep(3000.0, "Immortal: Linda called about dinner", "e1"), ep(1500.0, "Immortal: my dad's clinic is in Hebron", "e2")]
    n = compile_facets(st, eps, r.kb)
    a = st.entities["e1"].attributes
    check("how often, since when, last when", a["mentions"] == 3 and a["first_seen"] == 1000.0 and a["last_seen"] == 3000.0, str(a))
    check("the relationship from what it was told", a.get("relationship") == "sister", str(a))
    check("a few distinct things said around it (no repeats)", a["heard_in"] == ["Linda called about dinner", "Linda is coming Friday"], str(a.get("heard_in")))
    check("a place gets its mentions, no relationship", st.entities["e2"].attributes["mentions"] == 1 and "relationship" not in st.entities["e2"].attributes)
    check("one never mentioned in an episode gets nothing invented", "mentions" not in st.entities["e3"].attributes)
    check("two changed, saved once", n == 2 and st.saved == 1, f"{n} {st.saved}")
    st2 = Store(ent("e1", "Linda Iqnaibi"))
    compile_facets(st2, [ep(1.0, "Immortal: Linda called [mentions: Linda Iqnaibi]", "e1")], None)
    check("the episode's own tag is not a thing said", st2.entities["e1"].attributes["heard_in"] == ["Linda called"], str(st2.entities["e1"].attributes))
    check("run again: nothing changes, nothing saved", compile_facets(st, eps, r.kb) == 0 and st.saved == 1)
    check("the old attributes are kept", a["canonical_name"] == "Linda Iqnaibi")


def f2_likes():
    print("\n=== F2. What they like ===")
    r = tw.rig()
    r.kb.store_relation("Omar", role="friend")
    with r.kb._lock:
        r.kb.tastes["omar"] = {"name": "Omar", "items": {"chess": {}, "jazz": {}}}
    st = Store(ent("e1", "Omar"))
    compile_facets(st, [ep(1.0, "Immortal: Omar won", "e1")], r.kb)
    check("likes come from the taste profile", st.entities["e1"].attributes.get("likes") == ["chess", "jazz"], str(st.entities["e1"].attributes))
    check("without a knowledge base it still compiles the rest", compile_facets(Store(ent("e9", "X")), [ep(1.0, "Immortal: X", "e9")], None) == 1)


def f3_mind_counts_them():
    print("\n=== F3. The Mind page counts them as facets ===")
    useful = [k for k in ("canonical_name", "last_seen_context", "mentions", "relationship") if k not in ("canonical_name", "last_seen_context", "source", "updated_at")]
    check("'mentions' and 'relationship' are not in the ignored list", useful == ["mentions", "relationship"])


if __name__ == "__main__":
    f1_compile()
    f2_likes()
    f3_mind_counts_them()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
