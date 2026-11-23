"""
tst_prune.py -- pruning episodic memory: importance is read, and the daemon adding episodes meanwhile is not lost.

prune() dropped anything older than a week that had never been retrieved -- a 0.95 alert or something the user said included, although
journal.py says importance decides whether an episode survives. And it ran on the maintenance thread with no lock while the daemon loop was
adding episodes: one added mid-prune was missing from the survivors (gone from RAM and from the rewritten file), and iterating the dict
could raise "dict changed size".

Run:  PYTHONIOENCODING=utf-8 python tests/tst_prune.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def mem():
    from src.memory.episodic.episodic_manager import EpisodicMemory
    return EpisodicMemory(None, path=os.path.join(tempfile.mkdtemp(), "e.jsonl"))


def p1_importance():
    print("\n=== P1. What survives ===")
    m = mem()
    old = time.time() - 30 * 86400
    keep = {}
    for name, imp, accessed in [("dull", 0.3, 0), ("alert", 0.95, 0), ("said", 0.7, 0), ("used", 0.3, 2), ("mid", 0.69, 0)]:
        ep = m.add_episode(f"old {name}", "interaction", importance=imp)
        ep.timestamp = old
        ep.access_count = accessed
        keep[name] = ep.episode_id
    fresh = m.add_episode("fresh and dull", "interaction", importance=0.1)
    n = m.prune(min_age_seconds=7 * 86400)
    alive = {k for k, v in keep.items() if v in m.store.episodes}
    check("an old, never-used, unimportant episode goes (and just those)", n == 2 and alive == {"alert", "said", "used"}, f"pruned {n}, alive {sorted(alive)}")
    check("a fresh one stays", fresh.episode_id in m.store.episodes)
    from src.memory.episodic.episodic_manager import EpisodicMemory
    m2 = EpisodicMemory(None, path=m.store.path)
    check("what survives is what is on disk", {e.summary for e in m2.store.all()} == {e.summary for e in m.store.all()})
    check("the importance bar can be moved", mem().prune(keep_importance=0.0) == 0)


def p2_concurrent():
    print("\n=== P2. Adding while pruning ===")
    m = mem()
    old = time.time() - 30 * 86400
    for i in range(200):
        ep = m.add_episode(f"old dull {i}", "interaction", importance=0.2)
        ep.timestamp = old
    errors, added = [], []
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set() and i < 400:
            try:
                added.append(m.add_episode(f"new {i}", "interaction", importance=0.5).episode_id)
            except Exception as e:                       # noqa: BLE001
                errors.append(repr(e))
            i += 1
    th = threading.Thread(target=writer)
    th.start()
    for _ in range(25):
        try:
            m.prune(min_age_seconds=7 * 86400)
        except Exception as e:                           # noqa: BLE001
            errors.append(repr(e))
    stop.set()
    th.join()
    from src.memory.episodic.episodic_manager import EpisodicMemory
    disk = EpisodicMemory(None, path=m.store.path)
    lost_ram = [e for e in added if e not in m.store.episodes]
    lost_disk = [e for e in added if e not in disk.store.episodes]
    check("no error on either thread", not errors, str(errors[:2]))
    check("every episode added meanwhile is still in memory", not lost_ram, f"{len(lost_ram)} of {len(added)} lost")
    check("...and on disk after a reload", not lost_disk, f"{len(lost_disk)} of {len(added)} lost")
    check("the old dull ones are gone", not any("old dull" in e.summary for e in m.store.all()))
    check("the index has one row per episode (in step)", len(m.index.episodes) == len(m.store.episodes), f"{len(m.index.episodes)} vs {len(m.store.episodes)}")


if __name__ == "__main__":
    p1_importance()
    p2_concurrent()
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)
