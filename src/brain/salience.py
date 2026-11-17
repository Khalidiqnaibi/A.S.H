"""
src/brain/salience.py

How strongly a turn is laid down. A brain remembers what surprised it and lets the routine blur; ASH laid down everything at the
caller's flat 0.5. Now a turn the predictive model did not expect (or that is unlike anything seen) is written stronger -- it is
what recall and pruning keep -- and a routine, familiar one a little weaker. A caller that asked for more than the default
(an explicit "remember this") is never lowered.
"""

DEFAULT = 0.5
SURPRISE_GAIN = 0.30        # at full surprise, up to this much more
FAMILIAR_LOSS = 0.10        # a familiar, unsurprising turn, this much less
CAP = 0.68                  # plus the slow path's 0.15 stays under the 0.85 "always recall" line: kept, not forced into every prompt


def remembered_importance(importance: float, surprise: float = 0.0, novelty: float = 0.0) -> float:
    """`importance` adjusted by how surprising (0..1) and how novel (0..1) the turn was."""
    base = float(importance)
    s = max(0.0, min(1.0, float(surprise or 0.0)))
    n = max(0.0, min(1.0, float(novelty or 0.0)))
    up = SURPRISE_GAIN * max(s, 0.6 * n)
    out = base + up
    if up < 0.02 and s < 0.1 and n < 0.15 and base <= DEFAULT:
        out = base - FAMILIAR_LOSS
    if base > DEFAULT:
        out = max(base, out)                                  # asked for on purpose: never lowered
    return round(max(0.2, min(max(CAP, base), out)), 3)
