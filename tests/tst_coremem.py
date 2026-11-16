"""
tst_coremem.py -- "remember that" is not a rule for ASH.

A user's "Remember that." became a HARD core constraint, and so did "my favorite number is 4242, remember that": core memory's hard
rules go into every prompt, so each answer carried them (and the model took them as instructions). Nothing in it: nothing is kept.
A fact in it: kept, not forced into every prompt. A real rule ("from now on ...", "always ...") is still hard.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_coremem.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.memory.memory_router import MemoryRouter  # noqa: E402
import memory_cleanup as mc  # noqa: E402

check = tw.check


def k1_classify():
    print("\n=== K1. What becomes a core memory ===")
    r = MemoryRouter.__new__(MemoryRouter)
    c = r._classify_core_candidate
    check("'Remember that.' keeps nothing", c("Remember that.") is None)
    check("'ok remember this' keeps nothing", c("Ok, remember this.") is None)
    got = c("my favorite number is 4242, remember that")
    check("a fact: kept, but not hard", got is not None and got[1] is False, str(got))
    got = c("Immortal is trying to build up Ash's concepts and make it better. Remember that.")
    check("another fact: not hard", got is not None and got[1] is False, str(got))
    check("a rule with 'from now on' is still hard", (c("From now on, always answer in one sentence") or (0, False))[1] is True)
    check("'remember that you must never share my address' is still hard", (c("remember that you must never share my address") or (0, False))[1] is True)
    check("a question is still not a rule", c("what is the rule here?") is None)
    check("an explicit 'Policy:' prefix is still hard", c("Policy: answer briefly") == ("constraint", True))


def k2_cleanup():
    print("\n=== K2. The repair ===")
    d = tempfile.mkdtemp()
    core = {"core_remember_that.": {"rule_id": "core_remember_that.", "category": "constraint", "text": "Remember that.", "hard": True},
            "core_fav": {"rule_id": "core_fav", "category": "constraint", "text": "my favorite number is 4242, remember that", "hard": True},
            "core_rule": {"rule_id": "core_rule", "category": "constraint", "text": "From now on always answer in one sentence. Remember that.", "hard": True},
            "manual_memory": {"rule_id": "manual_memory", "category": "capability", "text": "ASH manual: Remember Lieth is my friend", "hard": False}}
    json.dump(core, open(os.path.join(d, "core_memory.json"), "w"))
    p = mc.plan(d)
    check("the empty one is to go, the fact to be softened, the rule and the manual left",
          p["core"]["drop"] == ["core_remember_that."] and p["core"]["soften"] == ["core_fav"], str(p["core"]["drop"]) + str(p["core"]["soften"]))
    mc.apply(p)
    left = json.load(open(os.path.join(d, "core_memory.json")))
    check("the file: empty one gone, fact kept and soft, rule still hard",
          "core_remember_that." not in left and left["core_fav"]["hard"] is False and left["core_rule"]["hard"] is True, str(list(left)))
    check("a backup was made", any(f.startswith("core_memory.json.bak-") for f in os.listdir(d)))
    check("run again: nothing to do", not mc.plan(d)["core"]["drop"] and not mc.plan(d)["core"]["soften"])


def k3_loop_episodes():
    print("\n=== K3. Episodes that are a recogniser's loop ===")
    d = tempfile.mkdtemp()
    loop = "I'm not sure if I'm going to work with Ash, I'm not sure if I'm going to work with Ash, I'm not sure if I'm going to work with Ash."
    with open(os.path.join(d, "episodic_memory.jsonl"), "w") as f:
        for i, s in enumerate(["Immortal: " + loop, "Immortal: what is the time", "Immortal: remind me at five"]):
            f.write(json.dumps({"episode_id": f"e{i}", "timestamp": 1.0, "summary": s, "related_entities": []}) + "\n")
    p = mc.plan(d)
    check("the loop is to go, the two real lines stay", len(p["drop_episodes"]) == 1 and "going to work" in p["drop_episodes"][0]["summary"], str(p["drop_episodes"]))


if __name__ == "__main__":
    k1_classify()
    k2_cleanup()
    k3_loop_episodes()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
