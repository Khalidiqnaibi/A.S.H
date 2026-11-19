"""
tst_nap.py -- a light consolidation in any ten quiet minutes and on every clean stop; the night's pruning stays for the night.

Learning used to be consolidated only by a 25-minute-idle phase at most every six hours, which a PC that is used or switched off
rarely reached. A nap replays what was learned and writes it down; it does not downscale, prune, merge or clear working memory.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_nap.py
"""

import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.brain import Brain  # noqa: E402
from src.brain.consolidation import consolidate  # noqa: E402
from src.concepts.sleep import SleepCycle, SleepConfig  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402

check = tw.check


class Rec:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, k))
            return {}
        return f


def s1_cycle():
    print("\n=== S1. The sleep cycle, light ===")
    calls = []
    plast = NS(replay_mode=False, _recent=NS(clear=lambda: None), _snapshot=lambda: 1, _restore=lambda s: None,
               _prune=lambda: calls.append("prune") or 0, _homeostatic_scaling=lambda: calls.append("scale") or 0, _cap_inputs=lambda: calls.append("cap") or 0, _audit_causal=lambda: calls.append("audit") or 0,
               _merge_converged=lambda: calls.append("merge") or 0)
    hpc = NS(replay_batch=lambda size, new_fraction: [], decay=lambda hours: calls.append(("decay", hours)) or 0, status=lambda: {})
    net = NS()
    sc = SleepCycle(net, plast, hpc, None, SleepConfig())
    sc._snapshot_stats = lambda: {}
    sc._delta = lambda b: {}
    sc._downscale = lambda: calls.append("downscale") or {}
    stages = []
    sc._nrem = lambda: stages.append("n") or {}
    sc._rem = lambda: stages.append("r") or {}
    rep = sc.run(hours_since_last=0.2, light=True)
    check("a nap is one cycle", stages == ["n", "r"], str(stages))
    check("no downscale, prune or scale; near-duplicates are merged", "merge" in calls and "audit" in calls and not any(c in calls for c in ("downscale", "prune", "scale", "cap")), str(calls))
    check("the traces fade by the real time", ("decay", 0.2) in calls, str(calls))
    stages.clear(); calls.clear()
    sc.run(hours_since_last=8.0)
    check("a night is three cycles with all of it", len(stages) == 6 and {"downscale", "prune", "scale", "merge", "cap", "audit"} <= set(calls), str((stages, calls)))


def s2_consolidate():
    print("\n=== S2. consolidate(light) ===")
    rec = Rec()
    brain = NS(predictive=NS(consolidate=lambda epochs: rec.calls.append(("replay", epochs)) or {}, save=lambda: rec.calls.append("psave")),
               concepts=NS(sleep=lambda light=False: rec.calls.append(("csleep", light)) or {}),
               system1=NS(save=lambda: rec.calls.append("s1save"), invalidate_reflex=lambda: rec.calls.append("reflex")),
               memory=NS(clear_working=lambda: rec.calls.append("clear")),
               homeostasis=NS(recover=lambda s: rec.calls.append("recover"), drives=NS(as_dict=lambda: {})),
               executive=NS(constitution=NS(refresh=lambda: None)))
    rep = consolidate(brain, light=True)
    names = [c if isinstance(c, str) else c[0] for c in rec.calls]
    check("replayed (briefly) and the concepts napped", ("replay", 2) in rec.calls and ("csleep", True) in rec.calls, str(rec.calls))
    check("written down", rep.get("persisted") is True and "s1save" in names and "psave" in names)
    check("working memory, reflexes and drives untouched", not {"clear", "reflex", "recover"} & set(names), str(names))


def s3_gap():
    print("\n=== S3. Brain.nap keeps its distance ===")
    runs = []
    import src.brain.consolidation as cons
    orig = cons.consolidate
    cons.consolidate = lambda b, light=False, **k: runs.append(light) or {"light": light}
    try:
        b = NS()
        Brain.nap(b)
        Brain.nap(b, min_gap_s=60)
        check("a second nap within the gap is skipped", runs == [True], str(runs))
        b._last_consolidation = time.time()
        r = Brain.nap(NS(_last_consolidation=time.time()), min_gap_s=60)
        check("so is one right after a night", "skipped" in r)
        Brain.nap(NS())
        check("with no gap asked, it runs (the stop)", runs == [True, True], str(runs))
    finally:
        cons.consolidate = orig


def s4_trigger():
    print("\n=== S4. The runtime naps after ten quiet minutes ===")
    napped = threading.Event()
    r = AmbientRuntime.__new__(AmbientRuntime)
    r.cfg = NS(sleep_after_idle_s=1500.0, nap_after_idle_s=600.0, nap_min_interval_s=2700.0, sleep_min_interval_s=3600.0, sleep_brain_gap_s=21600.0)
    r.brain = NS(nap=lambda min_gap_s=0.0: napped.set() or {"light": True})
    r.stats = {}
    r.last_sleep = 0.0
    r._user_busy = 0.0
    now = 10 ** 6
    r._last_activity = now - 300
    AmbientRuntime._maybe_sleep(r, now)
    check("five minutes quiet: nothing", not napped.wait(0.3))
    r._last_activity = now - 700
    r._user_busy = 0.6
    AmbientRuntime._maybe_sleep(r, now)
    check("the user at the keyboard: nothing", not napped.wait(0.3))
    r._user_busy = 0.0
    AmbientRuntime._maybe_sleep(r, now)
    check("ten minutes quiet: a nap", napped.wait(2.0))
    napped.clear()
    AmbientRuntime._maybe_sleep(r, now + 60)
    check("not again a minute later", not napped.wait(0.3))


if __name__ == "__main__":
    s1_cycle()
    s2_consolidate()
    s3_gap()
    s4_trigger()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
