"""
tools/concept_cleanup.py

Take the concepts that were grown from the machine's own readings ("memory at 93%", "screen changed while in ...", "switched to X
(was on Y ...)", battery, motion) out of the saved concept graph, with their wires.

Before the telemetry rule (src/concepts/telemetry.py) every such reading grew a concept; there were 21 of them, one fired 12,000 times
and took a quarter of all activation. Only concepts that were GROWN from such a text are removed -- never one that came with ASH
(intents, seeded ones) and never one a person's sentence made.

    python tools/concept_cleanup.py            # look
    python tools/concept_cleanup.py --apply    # back up state/concepts, then remove (with ASH stopped)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_OWN = __import__("re").compile(
    r"^(?:you(?:'ve| have) been on\b|anything from it you want me to keep|want today's|do you want me to|want me to\b|are you "
    r"still\b)", __import__("re").I)


def is_own_remark(text: str) -> bool:
    """A sentence ASH itself says to the user (a nudge, an offer), grown into a concept when it came back through the loop."""
    return bool(_OWN.search(str(text or "").strip()))


def plan(directory: str) -> Dict[str, Any]:
    import json
    from src.concepts.telemetry import is_telemetry
    meta_path = os.path.join(directory, "graph.json")
    out: Dict[str, Any] = {"dir": directory, "drop": [], "total": 0}
    if not os.path.exists(meta_path):
        return out
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    cs: List[Dict[str, Any]] = meta.get("concepts", [])
    out["total"] = len(cs)
    for c in cs:
        if c.get("origin") != "grown" or c.get("frozen"):
            continue
        text = str(c.get("seed_text") or c.get("label") or "")
        if is_telemetry(text, "") or is_own_remark(text):
            out["drop"].append({"id": int(c["id"]), "text": text[:70], "fired": c.get("activations", 0)})
    return out


def apply(p: Dict[str, Any]) -> str:
    from src.concepts.network import ConceptNetwork
    d = p["dir"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = d.rstrip("\\/") + ".bak-" + stamp
    shutil.copytree(d, backup)
    import json
    with open(os.path.join(d, "graph.json"), "r", encoding="utf-8") as fh:
        dim = int(json.load(fh).get("dim", 384))
    net = ConceptNetwork(dim=dim)
    if not net.load(d):
        raise SystemExit("could not load the saved graph; nothing changed")
    for c in p["drop"]:
        net.remove_concept(c["id"])
    net.save(d)
    return backup


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Remove concepts grown from the machine's own readings.")
    ap.add_argument("--dir", default=os.path.join(os.getcwd(), "state", "concepts"), help="the concept state folder")
    ap.add_argument("--apply", action="store_true", help="do it (after a backup)")
    a = ap.parse_args(argv)
    p = plan(a.dir)
    print(f"{p['total']} concepts in {a.dir}; grown from readings of the machine: {len(p['drop'])}")
    for c in p["drop"]:
        print(f"   c{c['id']:<4} fired {c['fired']!s:>6}  {c['text']!r}".encode("ascii", "replace").decode())
    if not a.apply:
        print("\nNothing was changed. Run again with --apply (with ASH stopped) to back up and clean.")
        return 0
    print("\nDone. Backup:", apply(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
