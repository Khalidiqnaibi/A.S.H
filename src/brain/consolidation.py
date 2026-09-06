"""
src/brain/consolidation.py

Sleep / maintenance phase for the brain layer.

ASH already had a MaintenanceScheduler that pruned episodic memory on idle.
The brain adds three things worth doing while nobody is watching, each of
which is too expensive to do inline:

  1. REPLAY  -- retrain the Predictive Model's forward model on its replay
     buffer for several epochs. Online SGD during a live turn gets one
     gradient step per transition; replay gets many, in shuffled order,
     which is the difference between a model that tracks and a model that
     generalizes. This is hippocampal replay in the loosest but most
     load-bearing sense: the day's transitions get re-experienced offline.

  2. RECOVER -- repay accumulated fatigue and relax all drives back toward
     their homeostatic setpoints, so a session that ended frustrated doesn't
     start the next one frustrated.

  3. PERSIST -- flush the critic weights and forward model to disk.

Call `consolidate(brain)` from MaintenanceScheduler.run_sweep().
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger("ash.brain.consolidation")


def consolidate(brain, seconds_idle: float = 1800.0, epochs: int = 6) -> Dict[str, Any]:
    t0 = time.perf_counter()
    report: Dict[str, Any] = {"started": time.time()}

    # 1. Replay.
    try:
        report["replay"] = brain.predictive.consolidate(epochs=epochs)
    except Exception:
        logger.exception("Consolidation: replay failed")
        report["replay"] = {"error": True}

    # 1b. Synaptic consolidation. Only place w_slow is ever written, and the
    # only place concepts merge or synapses are pruned -- see
    # src/concepts/plasticity.py for why this is gated on sleep rather than
    # running online.
    try:
        concepts = getattr(brain, "concepts", None)
        if concepts is not None:
            report["concepts"] = concepts.sleep()
    except Exception:
        logger.exception("Consolidation: concept graph failed")

    # 2. Homeostatic recovery.
    try:
        brain.homeostasis.recover(seconds_idle)
        report["drives_after"] = brain.homeostasis.drives.as_dict()
    except Exception:
        logger.exception("Consolidation: recovery failed")

    # 3. Clear volatile state that shouldn't survive a long gap.
    try:
        brain.system1.invalidate_reflex()
        brain.memory.clear_working()
        brain.executive.constitution.refresh()   # pick up any new hard rules
        report["cleared"] = ["reflex_cache", "working_memory", "constitution_cache"]
    except Exception:
        logger.exception("Consolidation: cleanup failed")

    # 4. Persist.
    try:
        brain.system1.save()
        brain.predictive.save()
        report["persisted"] = True
    except Exception:
        logger.exception("Consolidation: persist failed")
        report["persisted"] = False

    report["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("Brain consolidation complete in %.0fms: %s", report["duration_ms"], report.get("replay"))
    return report
