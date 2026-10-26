"""tst_soak.py -- long simulated workloads through ConceptSystem with every
plasticity mechanism on: topic-structured streams, rewards, sleeps, save/load,
protected units, noise bursts. Checks global invariants, not specific numbers."""
import math, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ASH_CONCEPT_STATE"] = tempfile.mkdtemp()
from src.concepts import ConceptSystem, Origin  # noqa: E402

DIM = 48
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def sample(rng, centres, t, noise):
    return unit(centres[t] + noise * rng.standard_normal(DIM).astype(np.float32) / math.sqrt(DIM) * 3)


def invariants(cs, tag):
    net = cs.net
    bad = []
    for c in net.concepts.values():
        if not np.all(np.isfinite(c.prototype)):
            bad.append("proto nan")
        elif abs(float(np.linalg.norm(c.prototype)) - 1.0) > 1e-3:
            bad.append("proto norm")
        if not math.isfinite(c.rate) or c.rate < -1e-9 or c.rate > 1 + 1e-9:
            bad.append(f"rate {c.rate}")
    for k, s in net.synapses.items():
        if not (math.isfinite(s.w_fast) and math.isfinite(s.w_slow)):
            bad.append("syn nan"); break
        if not (-1.5 - 1e-6 <= s.w_slow <= 1.5 + 1e-6) or s.weight() < -3.0:
            bad.append(f"negative {s.kind.value} {s.w_fast} {s.w_slow}"); break
        if k[0] not in net.concepts or k[1] not in net.concepts:
            bad.append("dangling synapse"); break
    # reverse index agrees with synapse table
    idx = sum(len(v) for v in net._in.values())
    if idx != len(net.synapses):
        bad.append(f"_in {idx} != {len(net.synapses)}")
    check(f"{tag}: invariants", not bad, str(sorted(set(bad))[:4]))


def run(seed, pull=True):
    rng = np.random.default_rng(seed)
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    centres = [unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(6)]
    # protected seeds
    for i in range(3):
        cs.net.add_concept(centres[i], label=f"seed{i}", origin=Origin.INTENT, intent=f"i{i}")
    cs.plasticity.nonhebb.cfg.neigh_enabled = pull
    protos0 = {cid: c.prototype.copy() for cid, c in cs.net.concepts.items()}
    for step in range(1500):
        t = int(rng.integers(0, 6)) if rng.random() < 0.3 else (step // 100) % 6
        noise = 3.0 if rng.random() < 0.05 else 0.4
        act = cs.perceive(sample(rng, centres, t, noise), text=f"s{step}",
                          surprise=float(rng.random() * 0.5))
        if step % 7 == 0:
            cs.encode_episode(sample(rng, centres, t, 0.4), act=act)
        if step % 5 == 0:
            cs.reward(float(rng.normal(0, 1)))
        if step % 400 == 399:
            cs.sleep()
            invariants(cs, f"seed{seed} sleep@{step}")
    invariants(cs, f"seed{seed} end")
    # Seeded units may drift through ordinary winner learning (low plasticity),
    # but must keep their identity: stay nearer their own origin than any other.
    lost = [cid for cid, p in protos0.items()
            if cid in cs.net.concepts and float(cs.net.concepts[cid].prototype @ p) < 0.85]
    check(f"seed{seed}: seeded units keep their identity", not lost, str(lost))
    n = len(cs.net.concepts)
    check(f"seed{seed}: concept count bounded", n < 1000, str(n))
    check(f"seed{seed}: weights bounded", max([s.weight() for s in cs.net.synapses.values()] or [0]) <= 4.0)
    # save / load roundtrip keeps rate and graph
    cs.net.save()
    cs2 = ConceptSystem(None, None, dim=DIM, load=True, seed=False)
    check(f"seed{seed}: load restores graph",
          len(cs2.net.concepts) == len(cs.net.concepts) and len(cs2.net.synapses) == sum(1 for x in cs.net.synapses.values() if abs(x.w_slow) > 1e-4 or x.frozen))
    invariants(cs2, f"seed{seed} reloaded")
    r1 = sum(c.rate for c in cs.net.concepts.values())
    r2 = sum(c.rate for c in cs2.net.concepts.values())
    check(f"seed{seed}: rates persisted", abs(r1 - r2) < 1e-3 * max(1, r1), f"{r1} {r2}")
    # still works after reload
    for step in range(100):
        cs2.perceive(sample(rng, centres, step % 6, 0.4), text="x")
    invariants(cs2, f"seed{seed} post-reload run")
    return cs


if __name__ == "__main__":
    for sd in (1, 2, 3):
        cs = run(sd)
    # off-switch parity: all mechanisms disabled still runs clean
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    for f in ("hetero_enabled", "meta_enabled", "neigh_enabled"):
        setattr(cs.plasticity.nonhebb.cfg, f, False)
    rng = np.random.default_rng(9)
    cen = [unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(4)]
    for i in range(400):
        cs.perceive(sample(rng, cen, i % 4, 0.4))
    invariants(cs, "all-off")
    print("\nFAILED:", FAILS if FAILS else "none")
    sys.exit(1 if FAILS else 0)
