"""
tst_nonhebbian.py -- tests for heterosynaptic depression, metaplasticity and
neighbourhood plasticity.

Every one of these mechanisms exists to stop something specific going wrong in
a self-rewiring graph, so the tests target the failure each one prevents and
the invariants it must never break, not just "does the number move":

  Reverse index        inputs() stays correct through create / remove / prune / load
  Heterosynaptic       depresses INACTIVE inputs onto a strongly active unit, in
                       proportion to their weight, and nothing else:
                       never below the trigger, never frozen / route / inhibitory
                       edges, never across zero; fast (one cycle), so inputs compete
  Metaplasticity       recent activity slides the learning rate down and recovers,
                       is bounded, lazy-decays exactly, survives a save/load, and
                       slows runaway growth
  Neighbourhood        spreads a strengthened edge onto prototype-neighbours of
                       both ends, proportionally to closeness, always strictly
                       weaker than the trigger, only provisionally (needs reward to
                       consolidate), and pulls neighbour prototypes with a ceiling
                       so a neighbourhood never collapses; protected units are
                       never pulled
  Interactions         long random workloads stay finite and bounded, are
                       deterministic, and the three can each be switched off
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())

from src.concepts import (  # noqa: E402
    ConceptNetwork, ConceptSystem, NonHebbian, NonHebbianConfig, Origin,
    Plasticity, PlasticityConfig, SynType,
)
from src.concepts.model import Activation  # noqa: E402

DIM = 32


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def rnd(seed):
    return unit(np.random.default_rng(seed).standard_normal(DIM))


def at_cos(base, c, seed):
    """A unit vector whose cosine with `base` is exactly `c`."""
    r = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    r = r - float(r @ base) * base
    r = unit(r)
    return unit(c * base + math.sqrt(max(0.0, 1.0 - c * c)) * r)


def make(nh=None, **kw):
    """A network and a Plasticity with habituation off, so activations are
    exactly what the test says they are."""
    net = ConceptNetwork(dim=DIM, k=8, vigilance=0.99, persist=False)
    cfg = PlasticityConfig(adaptation_strength=0.0, **kw)
    if nh is not None:
        cfg.nonhebbian = nh
    return net, Plasticity(net, cfg)


def add(net, v, label=None, origin=Origin.GROWN, frozen=False):
    return net.add_concept(v, label=label, origin=origin, frozen=frozen).id


def cycle(net, pl, spread):
    """One live cycle with the given activations."""
    net.cycles += 1
    for cid, v in spread.items():
        net.concepts[cid].touch(v)
    act = Activation(initial=dict(spread), spread=dict(spread))
    net._record_cooccurrence(act)
    return pl.potentiate(act)


def main():
    ok = True
    n_checks = 0

    def check(label, cond, detail=""):
        nonlocal ok, n_checks
        n_checks += 1
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and bool(cond)

    # ===================================================================
    print("\n=== 1. Reverse index (inputs onto a unit) ===")
    net = ConceptNetwork(dim=DIM, persist=False)
    a, b, c, d = (add(net, rnd(i)) for i in range(4))
    net.link(a, c, 0.5)
    net.link(b, c, 0.4, bidirectional=False)
    check("inputs() lists every source", set(net.inputs(c)) == {a, b}, str(sorted(net.inputs(c))))
    check("inputs() agrees with the synapse table",
          all(net.inputs(dd)[s] is net.synapses[(s, dd)] for (s, dd) in net.synapses))
    check("a unit with no inputs has none", net.inputs(d) == {})
    net.remove_concept(a)
    check("removing a unit drops its edges from the index", set(net.inputs(c)) == {b}
          and a not in net.inputs(c) and net.inputs(a) == {})
    pl = Plasticity(net, PlasticityConfig())
    net.synapses[(b, c)].w_slow = 0.0
    net.synapses[(b, c)].w_fast = 0.0
    pl._prune()
    check("pruning an edge drops it from the index", b not in net.inputs(c))
    big, pl_big = make()
    ids = [add(big, rnd(100 + i)) for i in range(12)]
    for i in ids:
        for j in ids:
            if i != j:
                big.link(i, j, 0.1 + 0.05 * ((i * 7 + j) % 9), bidirectional=False)
    pl_big.cfg.max_synapses = 40
    pl_big._emergency_prune()
    consistent = (len(big.synapses) == sum(len(v) for v in big._in.values())
                  and all(big._in[dd][s] is syn for (s, dd), syn in big.synapses.items()))
    check("emergency prune rebuilds the index consistently", consistent,
          f"{len(big.synapses)} synapses")
    d_dir = tempfile.mkdtemp()
    big.persist = True
    big.save(d_dir)
    re = ConceptNetwork(dim=DIM, persist=False)
    re.load(d_dir)
    check("loading rebuilds the index",
          all(re._in[dd][s] is syn for (s, dd), syn in re.synapses.items())
          and len(re.synapses) == sum(len(v) for v in re._in.values()))

    # ===================================================================
    print("\n=== 2. Heterosynaptic depression ===")
    nh_cfg = NonHebbianConfig(neigh_enabled=False, meta_enabled=False)

    def hetero_net():
        n, p = make(nh=NonHebbianConfig(neigh_enabled=False, meta_enabled=False))
        post = add(n, rnd(200))
        act_in = add(n, rnd(201))
        quiet = add(n, rnd(202))
        quiet2 = add(n, rnd(203))
        n.link(act_in, post, 0.6, bidirectional=False)
        n.link(quiet, post, 0.6, bidirectional=False)
        n.link(quiet2, post, 0.2, bidirectional=False)
        return n, p, post, act_in, quiet, quiet2

    n, p, post, act_in, quiet, quiet2 = hetero_net()
    p.nonhebb.heterosynaptic({post: 1.0, act_in: 1.0}, {post: 1.0, act_in: 1.0}, 1.0)
    w_q = n.synapses[(quiet, post)].weight()
    w_a = n.synapses[(act_in, post)].weight()
    check("an inactive input onto a firing unit is depressed", w_q < 0.6, f"0.600 -> {w_q:.4f}")
    check("an input that was active is left alone", abs(w_a - 0.6) < 1e-9, f"{w_a:.4f}")
    fr_q2 = (0.2 - n.synapses[(quiet2, post)].weight()) / 0.2
    fr_q = (0.6 - w_q) / 0.6
    check("depression is proportional to the weight (same fraction)", abs(fr_q - fr_q2) < 1e-6,
          f"{fr_q:.4f} vs {fr_q2:.4f}")
    check("the stronger input loses more in absolute terms",
          (0.6 - w_q) > (0.2 - n.synapses[(quiet2, post)].weight()))
    expected = 0.6 - (nh_cfg.hetero_rate * 1.0 * 0.6) * (1.0 + nh_cfg.hetero_slow_share)
    check("the amount matches the rule", abs(w_q - expected) < 1e-6, f"{w_q:.5f} vs {expected:.5f}")

    n, p, post, act_in, quiet, quiet2 = hetero_net()
    p.nonhebb.heterosynaptic({post: 0.6}, {post: 0.6}, 1.0)
    check("a burst at exactly the trigger depresses nothing",
          abs(n.synapses[(quiet, post)].weight() - 0.6) < 1e-9)
    p.nonhebb.heterosynaptic({post: 0.4}, {post: 0.4}, 1.0)
    check("a burst below the trigger depresses nothing",
          abs(n.synapses[(quiet, post)].weight() - 0.6) < 1e-9)

    amounts = []
    for a_post in (0.7, 0.85, 1.0):
        n, p, post, act_in, quiet, quiet2 = hetero_net()
        p.nonhebb.heterosynaptic({post: a_post}, {post: a_post}, 1.0)
        amounts.append(0.6 - n.synapses[(quiet, post)].weight())
    check("a stronger burst depresses more", amounts[0] < amounts[1] < amounts[2],
          str([round(x, 4) for x in amounts]))

    n, p, post, act_in, quiet, quiet2 = hetero_net()
    n.synapses[(quiet, post)].frozen = True
    n.synapses[(quiet2, post)].kind = SynType.ROUTE
    inh = add(n, rnd(204))
    n.link(inh, post, -0.8, bidirectional=False, kind=SynType.INHIBIT)
    sub = add(n, rnd(205))
    n.link(sub, post, 0.5, bidirectional=False, kind=SynType.PART_OF)
    p.nonhebb.heterosynaptic({post: 1.0}, {post: 1.0}, 1.0)
    check("a frozen synapse is never depressed", abs(n.synapses[(quiet, post)].weight() - 0.6) < 1e-9)
    check("a route into an intent is not eroded",
          abs(n.synapses[(quiet2, post)].weight() - 0.2) < 1e-9)
    check("an inhibitory edge is left alone", abs(n.synapses[(inh, post)].weight() + 0.8) < 1e-9)
    check("a part-of edge is left alone", abs(n.synapses[(sub, post)].weight() - 0.5) < 1e-9)

    n, p, post, act_in, quiet, quiet2 = hetero_net()
    weak = add(n, rnd(206))
    n.link(weak, post, 0.01, bidirectional=False)
    p.nonhebb.heterosynaptic({post: 1.0}, {post: 1.0}, 1.0)
    check("an input already near zero is left to decay, not driven below",
          abs(n.synapses[(weak, post)].weight() - 0.01) < 1e-9)

    n, p, post, act_in, quiet, quiet2 = hetero_net()
    for k in range(2000):
        p.nonhebb.heterosynaptic({post: 1.0}, {post: 1.0}, 1.0)
    ws = [s.weight() for (s_, d_), s in n.synapses.items() if d_ == post]
    check("repeated bursts never push a weight across zero", min(ws) >= 0.0,
          f"min weight {min(ws):.5f}")
    check("a weight settles near the floor instead of vanishing exactly",
          abs(n.synapses[(quiet, post)].weight()) < 0.05)

    # A switched-off mechanism does nothing.
    net_off, pl_off = make(nh=NonHebbianConfig(hetero_enabled=False, neigh_enabled=False,
                                               meta_enabled=False))
    po, qi = add(net_off, rnd(210)), add(net_off, rnd(211))
    net_off.link(qi, po, 0.6, bidirectional=False)
    check("a disabled mechanism does nothing",
          pl_off.nonhebb.heterosynaptic({po: 1.0}, {po: 1.0}, 1.0) == 0
          and abs(net_off.synapses[(qi, po)].weight() - 0.6) < 1e-9)

    # Through the real learning path.
    net_h, pl_h = make(nh=NonHebbianConfig(neigh_enabled=False, meta_enabled=False))
    po, qi = add(net_h, rnd(220)), add(net_h, rnd(221))
    net_h.link(qi, po, 0.6, bidirectional=False)
    cycle(net_h, pl_h, {po: 1.0})
    check("one strongly active unit (a cycle with no co-active partner) still triggers it",
          net_h.synapses[(qi, po)].weight() < 0.6,
          f"{net_h.synapses[(qi, po)].weight():.4f}")
    check("and the effect is immediate: no consolidation was needed",
          net_h.synapses[(qi, po)].w_slow <= 0.6 and pl_h.consolidations == 0)

    net_r, pl_r = make(nh=NonHebbianConfig(neigh_enabled=False, meta_enabled=False))
    po, qi = add(net_r, rnd(222)), add(net_r, rnd(223))
    net_r.link(qi, po, 0.6, bidirectional=False)
    pl_r.replay_mode = True
    cycle(net_r, pl_r, {po: 1.0})
    check("replay during sleep does not trigger it",
          abs(net_r.synapses[(qi, po)].weight() - 0.6) < 1e-9)

    # Competition between inputs.
    def compete(hetero):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=hetero, neigh_enabled=False,
                                        meta_enabled=False))
        A, B, P, X, Y = (add(n, rnd(230 + i)) for i in range(5))
        n.link(B, P, 0.5, bidirectional=False)
        n.link(A, P, 0.1, bidirectional=False)
        for i in range(120):
            cycle(n, p, {A: 1.0, P: 1.0} if i % 2 == 0 else {X: 1.0, Y: 1.0})
        return n.synapses[(A, P)].weight(), n.synapses[(B, P)].weight()

    wa_on, wb_on = compete(True)
    wa_off, wb_off = compete(False)
    check("an input that stays silent loses ground to the one that fires",
          wb_on < 0.5 * wb_off, f"B->P {wb_off:.3f} (off) vs {wb_on:.3f} (on)")
    check("the active input ends up the stronger one only with competition",
          wa_on > wb_on, f"A->P {wa_on:.3f} vs B->P {wb_on:.3f}")
    check("competition does not weaken the input that is learning",
          wa_on >= wa_off - 0.02, f"{wa_on:.3f} vs {wa_off:.3f}")

    # ===================================================================
    print("\n=== 3. Metaplasticity (sliding threshold) ===")
    n3, p3 = make(nh=NonHebbianConfig(hetero_enabled=False, neigh_enabled=False))
    u = add(n3, rnd(300))
    nh3 = p3.nonhebb
    cfg3 = nh3.cfg
    check("a never-active unit has zero recent activity", nh3.current_rate(u) == 0.0)
    check("a quiet unit learns faster than normal", nh3.meta_scale(u) > 1.0,
          f"{nh3.meta_scale(u):.3f}")

    n3.concepts[u].rate = cfg3.meta_target
    n3.concepts[u].rate_cycle = n3.cycles
    check("at the target activity the multiplier is exactly 1",
          abs(nh3.meta_scale(u) - 1.0) < 1e-9, f"{nh3.meta_scale(u):.6f}")

    scales = []
    for r in (0.0, 0.05, 0.15, 0.2, 0.3, 0.6, 1.0):
        n3.concepts[u].rate = r
        n3.concepts[u].rate_cycle = n3.cycles
        scales.append(nh3.meta_scale(u))
    check("the multiplier falls as activity rises, until it reaches its floor",
          all(x > y for x, y in zip(scales[:5], scales[1:5]))
          and all(x >= y for x, y in zip(scales, scales[1:])), str([round(s, 3) for s in scales]))
    check("it stays inside its bounds",
          all(cfg3.meta_min - 1e-9 <= s <= cfg3.meta_max + 1e-9 for s in scales))
    n3.concepts[u].rate = 50.0
    n3.concepts[u].rate_cycle = n3.cycles
    check("an absurd rate is clamped to the floor", abs(nh3.meta_scale(u) - cfg3.meta_min) < 1e-9)

    n3.concepts[u].rate = 0.0
    n3.concepts[u].rate_cycle = n3.cycles
    for i in range(400):
        n3.cycles += 1
        nh3.observe({u: 1.0})
    r_hot = nh3.current_rate(u)
    check("sustained activity raises the recent-activity average toward its level",
          0.8 < r_hot <= 1.0, f"{r_hot:.3f}")
    n3.cycles += 600
    r_cool = nh3.current_rate(u)
    exact = r_hot * (1.0 - 1.0 / cfg3.meta_tau) ** 600
    check("it decays lazily, exactly as if every cycle had been applied",
          abs(r_cool - exact) < 1e-6 and r_cool < 0.1 * r_hot, f"{r_hot:.3f} -> {r_cool:.4f}")
    check("reading the rate does not modify the unit",
          n3.concepts[u].rate_cycle < n3.cycles)
    check("a rested unit is back to learning at or above the normal rate",
          nh3.meta_scale(u) > 0.9, f"{nh3.meta_scale(u):.3f}")
    n3.concepts[u].rate = 0.99
    n3.concepts[u].rate_cycle = n3.cycles
    n3.cycles += 1
    nh3.observe({u: 50.0})
    check("an activity above 1 cannot push the average past 1",
          nh3.current_rate(u) <= 1.0, f"{nh3.current_rate(u):.4f}")

    n3.concepts[u].rate = 0.9
    n3.concepts[u].rate_cycle = n3.cycles
    p_off = Plasticity(n3, PlasticityConfig(nonhebbian=NonHebbianConfig(meta_enabled=False)))
    check("a disabled sliding threshold is exactly 1", p_off.nonhebb.meta_scale(u) == 1.0)

    # A busy post-synaptic unit is harder to potentiate than a quiet one.
    def pair_net():
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, neigh_enabled=False))
        A, P, X, Y = (add(n, rnd(310 + i)) for i in range(4))
        # A short history: the weight is still far from the Oja equilibrium,
        # where a smaller learning rate would shrink the pull-back instead.
        for i in range(8):
            cycle(n, p, {A: 1.0, P: 1.0} if i % 2 == 0 else {X: 1.0, Y: 1.0})
        return n, p, A, P, X, Y

    ns, ps, A_s, P_s, X_s, Y_s = pair_net()
    nb, pb, A_b, P_b, X_b, Y_b = pair_net()
    check("two identical histories have identical activity records",
          abs(ns.concepts[P_s].rate - nb.concepts[P_b].rate) < 1e-12)
    nb.concepts[P_b].rate = 0.7
    nb.concepts[P_b].rate_cycle = nb.cycles
    before_s = ns.synapses[(A_s, P_s)].w_fast
    before_b = nb.synapses[(A_b, P_b)].w_fast
    cycle(ns, ps, {A_s: 1.0, P_s: 1.0})
    cycle(nb, pb, {A_b: 1.0, P_b: 1.0})
    gain_s = ns.synapses[(A_s, P_s)].w_fast - before_s
    gain_b = nb.synapses[(A_b, P_b)].w_fast - before_b
    check("the same coincidence potentiates a busy unit less than a quiet one",
          0.0 < gain_b < gain_s, f"busy {gain_b:.5f} vs quiet {gain_s:.5f}")
    ratio = gain_b / gain_s
    expect_ratio = nb.concepts[P_b].rate and (pb.nonhebb.meta_scale(P_b) / ps.nonhebb.meta_scale(P_s))
    check("and by about the ratio of the two multipliers (Oja term aside)",
          abs(ratio - expect_ratio) < 0.12, f"{ratio:.3f} vs {expect_ratio:.3f}")

    # Slows runaway growth.
    def grow(meta, rate0, cycles=100):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, neigh_enabled=False,
                                        meta_enabled=meta), eta_fast=0.01)
        A, B, X, Y = (add(n, rnd(330 + i)) for i in range(4))
        n.concepts[B].rate = rate0
        n.concepts[B].rate_cycle = n.cycles
        for i in range(cycles):
            cycle(n, p, {A: 1.0, B: 1.0} if i % 2 == 0 else {X: 1.0, Y: 1.0})
        return n.synapses[(A, B)].weight()

    w_busy, w_plain, w_quiet = grow(True, 0.6), grow(False, 0.6), grow(True, 0.0)
    check("a unit that has been busy gains weight more slowly than with no threshold",
          w_busy < w_plain, f"{w_busy:.3f} vs {w_plain:.3f}")
    check("and a quiet unit more quickly (the threshold slides both ways)",
          w_quiet > w_plain, f"{w_quiet:.3f} vs {w_plain:.3f}")
    check("busy < unchanged < quiet", w_busy < w_plain < w_quiet)
    check("the busy unit's association still forms", w_busy > 0.01, f"{w_busy:.3f}")
    check("and every variant stays bounded", max(w_busy, w_plain, w_quiet) < 5.0)

    # Persistence of the activity record.
    n3.concepts[u].rate = 0.42
    n3.concepts[u].rate_cycle = 77
    d_dir = tempfile.mkdtemp()
    n3.persist = True
    n3.save(d_dir)
    n3b = ConceptNetwork(dim=DIM, persist=False)
    n3b.load(d_dir)
    check("the activity record survives a save and load",
          abs(n3b.concepts[u].rate - 0.42) < 1e-4 and n3b.concepts[u].rate_cycle == 77,
          f"{n3b.concepts[u].rate} @ {n3b.concepts[u].rate_cycle}")

    # ===================================================================
    print("\n=== 4. Neighbourhood: finding neighbours ===")
    nn, pn = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False))
    base = rnd(400)
    B0 = add(nn, base)
    n_near = add(nn, at_cos(base, 0.92, 401))
    n_mid = add(nn, at_cos(base, 0.75, 402))
    n_edge = add(nn, at_cos(base, 0.60, 403))
    n_out = add(nn, at_cos(base, 0.50, 404))
    n_far = add(nn, at_cos(base, 0.10, 405))
    n_frozen = add(nn, at_cos(base, 0.95, 406), frozen=True)
    n_extra = add(nn, at_cos(base, 0.58, 407))
    nbrs = pn.nonhebb.neighbours_of(B0)
    ids_ = [i for i, _ in nbrs]
    check("neighbours are returned nearest first",
          [s for _, s in nbrs] == sorted([s for _, s in nbrs], reverse=True), str([round(s, 2) for _, s in nbrs]))
    check("a close unit is a neighbour and a far one is not",
          n_near in ids_ and n_far not in ids_ and n_out not in ids_)
    check("a frozen unit is never a neighbour", n_frozen not in ids_)
    check("a unit is not its own neighbour", B0 not in ids_)
    check("the count is capped", len(ids_) <= pn.nonhebb.cfg.neigh_max, f"{len(ids_)}")
    check("a unit just past the floor is included, one just below is not",
          n_edge in ids_ and n_out not in ids_)
    ex = pn.nonhebb.neighbours_of(B0, exclude={n_near})
    check("excluded units are skipped and the next-nearest takes their place",
          n_near not in [i for i, _ in ex] and n_mid in [i for i, _ in ex])
    check("an unknown unit has no neighbours", pn.nonhebb.neighbours_of(99999) == [])
    lone, pl_lone = make()
    only = add(lone, rnd(408))
    check("a network of one has no neighbours", pl_lone.nonhebb.neighbours_of(only) == [])
    share_vals = [pn.nonhebb._share(s) for s in (0.55, 0.65, 0.8, 0.95, 1.0)]
    check("the share is zero at the floor and rises with closeness",
          share_vals[0] == 0.0 and all(x < y for x, y in zip(share_vals, share_vals[1:])),
          str([round(x, 3) for x in share_vals]))
    check("below the floor the share is exactly zero, never negative",
          pn.nonhebb._share(0.3) == 0.0 and pn.nonhebb._share(-1.0) == 0.0)
    check("the share never exceeds the configured gain",
          max(share_vals) <= pn.nonhebb.cfg.neigh_gain + 1e-9)

    # ===================================================================
    print("\n=== 5. Neighbourhood: synaptic spread ===")
    def spread_net(**nh_kw):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False, **nh_kw))
        ba, bb = rnd(500), rnd(501)
        a_ = add(n, ba)
        b_ = add(n, bb)
        na_ = add(n, at_cos(ba, 0.90, 502))
        nb_ = add(n, at_cos(bb, 0.78, 503))
        far_ = add(n, rnd(504))
        syn = n.synapse(a_, b_)
        syn.w_fast = 0.5
        return n, p, a_, b_, na_, nb_, far_

    n5, p5, a5, b5, na5, nb5, far5 = spread_net()
    credits = p5.nonhebb.spread_synapses([(a5, b5, 0.05, 1.0)], {a5, b5}, 1.0)
    w_na = n5.synapses[(na5, b5)].weight() if (na5, b5) in n5.synapses else 0.0
    w_nb = n5.synapses[(a5, nb5)].weight() if (a5, nb5) in n5.synapses else 0.0
    sh_na = p5.nonhebb._share(float(n5.concepts[na5].prototype @ n5.concepts[a5].prototype))
    sh_nb = p5.nonhebb._share(float(n5.concepts[nb5].prototype @ n5.concepts[b5].prototype))
    check("a neighbour of the source end gets the edge", w_na > 0.0, f"{w_na:.5f}")
    check("a neighbour of the target end gets the edge", w_nb > 0.0, f"{w_nb:.5f}")
    check("the amount is the share of the change", abs(w_na - sh_na * 0.05) < 1e-6
          and abs(w_nb - sh_nb * 0.05) < 1e-6, f"{w_na:.5f} vs {sh_na * 0.05:.5f}")
    check("a closer neighbour gets more", w_na > w_nb)
    check("an unrelated unit gets nothing",
          (far5, b5) not in n5.synapses and (a5, far5) not in n5.synapses)
    check("each neighbour edge returns a credit entry for the traces",
          len(credits) == 2 and all(c > 0 for _, c in credits))
    check("neighbour edges are new synapses created for the purpose",
          p5.nonhebb.stats["neigh_new_edges"] == 2)
    check("the triggering edge itself is untouched by the spread",
          abs(n5.synapses[(a5, b5)].weight() - 0.5) < 1e-9)

    n5b, p5b, a5b, b5b, na5b, nb5b, far5b = spread_net(neigh_new_edges=False)
    p5b.nonhebb.spread_synapses([(a5b, b5b, 0.05, 1.0)], {a5b, b5b}, 1.0)
    check("with new edges disallowed, nothing is created",
          (na5b, b5b) not in n5b.synapses and (a5b, nb5b) not in n5b.synapses)
    n5b.link(na5b, b5b, 0.1, bidirectional=False)
    p5b.nonhebb.spread_synapses([(a5b, b5b, 0.05, 1.0)], {a5b, b5b}, 1.0)
    check("but an existing neighbour edge is still strengthened",
          n5b.synapses[(na5b, b5b)].weight() > 0.1)

    n5c, p5c, a5c, b5c, na5c, nb5c, far5c = spread_net()
    n5c.link(na5c, b5c, 0.1, bidirectional=False, frozen=True)
    p5c.nonhebb.spread_synapses([(a5c, b5c, 0.05, 1.0)], {a5c, b5c}, 1.0)
    check("a frozen neighbour edge is never touched",
          abs(n5c.synapses[(na5c, b5c)].weight() - 0.1) < 1e-9)
    n5d, p5d, a5d, b5d, na5d, nb5d, far5d = spread_net()
    n5d.link(na5d, b5d, -0.7, bidirectional=False, kind=SynType.INHIBIT)
    p5d.nonhebb.spread_synapses([(a5d, b5d, 0.05, 1.0)], {a5d, b5d}, 1.0)
    check("an inhibitory neighbour edge is never strengthened",
          abs(n5d.synapses[(na5d, b5d)].weight() + 0.7) < 1e-9)

    n5e, p5e, a5e, b5e, na5e, nb5e, far5e = spread_net()
    p5e.nonhebb.spread_synapses([(a5e, b5e, 0.05, 1.0)], {a5e, b5e, na5e}, 1.0)
    check("a unit that is itself active this cycle is not given a spread edge",
          (na5e, b5e) not in n5e.synapses)

    n5f, p5f, a5f, b5f, na5f, nb5f, far5f = spread_net()
    check("a non-positive change spreads nothing",
          p5f.nonhebb.spread_synapses([(a5f, b5f, -0.05, 1.0), (a5f, b5f, 0.0, 1.0)],
                                      {a5f, b5f}, 1.0) == [] and len(n5f.synapses) == 1)
    check("an empty update list spreads nothing",
          p5f.nonhebb.spread_synapses([], {a5f, b5f}, 1.0) == [])
    n5g, p5g, a5g, b5g, na5g, nb5g, far5g = spread_net()
    n5g.synapses[(a5g, b5g)].w_fast = 0.0
    check("a trigger whose edge has no strength spreads nothing",
          p5g.nonhebb.spread_synapses([(a5g, b5g, 0.05, 1.0)], {a5g, b5g}, 1.0) == [])
    n5h, p5h, a5h, b5h, na5h, nb5h, far5h = spread_net()
    check("a trigger whose edge no longer exists spreads nothing",
          p5h.nonhebb.spread_synapses([(a5h, 99, 0.05, 1.0)], {a5h}, 1.0) == [])

    # The invariant that matters most: a neighbour is never as strong as its trigger.
    rng = np.random.default_rng(7)
    viol = 0
    changed_total = 0
    for trial in range(60):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False,
                                        neigh_max_triggers=5, neigh_gain=0.9))
        centers = [rnd(600 + trial * 10 + i) for i in range(3)]
        ids_t = []
        for ci, cv in enumerate(centers):
            for k in range(4):
                ids_t.append(add(n, at_cos(cv, 0.85 + 0.03 * k, 700 + trial * 100 + ci * 10 + k)))
        for _ in range(8):
            s, t = rng.choice(ids_t, size=2, replace=False)
            n.synapse(int(s), int(t)).w_fast = float(rng.uniform(0.05, 1.0))
        before = {k: v.weight() for k, v in n.synapses.items()}
        ups = []
        for _ in range(4):
            s, t = rng.choice(ids_t, size=2, replace=False)
            sy = n.synapse(int(s), int(t))
            sy.w_fast = max(sy.w_fast, 0.05)
            before[(int(s), int(t))] = sy.weight()
            ups.append((int(s), int(t), float(rng.uniform(0.01, 2.0)), 1.0))
        refs = {(u_[0], u_[1]): n.synapses[(u_[0], u_[1])].weight() for u_ in ups}
        max_ref = max(refs.values())
        p.nonhebb.spread_synapses(ups, set(), 1.0)
        for k, v in n.synapses.items():
            if k in refs:
                continue
            old = before.get(k, 0.0)
            if v.weight() > old + 1e-12:
                changed_total += 1
                if v.weight() > max_ref + 1e-9:
                    viol += 1
    check("fuzz: a spread edge never exceeds the strongest trigger", viol == 0 and changed_total > 50,
          f"{changed_total} edges changed, {viol} violations")

    # Provisional: no consolidation without reward.
    def consolidate_neigh(reward):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False))
        ba, bb = rnd(800), rnd(801)
        A, B = add(n, ba), add(n, bb)
        NA = add(n, at_cos(ba, 0.92, 802))
        X, Y = add(n, rnd(803)), add(n, rnd(804))
        for i in range(60):
            cycle(n, p, {A: 1.0, B: 1.0} if i % 2 == 0 else {X: 1.0, Y: 1.0})
            if reward and i % 2 == 0:
                p.reward(1.0)
        has_edge = (NA, B) in n.synapses
        p.consolidate()
        syn = n.synapses.get((NA, B))
        return has_edge, (syn.w_slow if syn else 0.0), n.synapses[(A, B)].w_slow

    has_u, slow_u, trig_u = consolidate_neigh(False)
    has_r, slow_r, trig_r = consolidate_neigh(True)
    check("live learning created a neighbour edge in both runs", has_u and has_r)
    check("without reward the neighbour edge does not become permanent", slow_u <= 1e-9,
          f"w_slow={slow_u:.5f}")
    check("the trigger edge still consolidates without reward", trig_u > 0.0, f"{trig_u:.4f}")
    check("with reward the neighbour edge consolidates", slow_r > 0.0, f"w_slow={slow_r:.5f}")
    check("and stays weaker than the edge that taught it", slow_r < trig_r,
          f"{slow_r:.4f} < {trig_r:.4f}")

    # Generalisation: the behavioural point of the whole mechanism.
    def teach(neigh):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False,
                                        neigh_enabled=neigh))
        topic = rnd(900)
        tool = rnd(901)
        phr = add(n, topic)                          # the phrasing it was taught
        sib = add(n, at_cos(topic, 0.88, 902))       # a sibling phrasing never heard with the tool
        stranger = add(n, at_cos(topic, 0.05, 903))  # unrelated topic
        t_ = add(n, tool)
        filler = [add(n, rnd(910 + i)) for i in range(2)]
        for i in range(80):
            cycle(n, p, {phr: 1.0, t_: 1.0} if i % 2 == 0 else {filler[0]: 1.0, filler[1]: 1.0})
        w = lambda x: n.synapses[(x, t_)].weight() if (x, t_) in n.synapses else 0.0
        return w(phr), w(sib), w(stranger)

    w_phr, w_sib, w_str = teach(True)
    w_phr0, w_sib0, w_str0 = teach(False)
    check("a phrasing that was taught is associated with the tool", w_phr > 0.05, f"{w_phr:.3f}")
    check("a sibling phrasing that was never taught is associated too", w_sib > 0.02,
          f"{w_sib:.3f}")
    check("without the neighbourhood mechanism it is not", w_sib0 == 0.0, f"{w_sib0:.3f}")
    check("an unrelated topic gets nothing either way", w_str == 0.0 and w_str0 == 0.0)
    check("the sibling is learned more weakly than the taught phrasing", w_sib < w_phr)

    # ===================================================================
    print("\n=== 6. Neighbourhood: prototype pull ===")
    def pull_net(**kw):
        n, p = make(nh=NonHebbianConfig(hetero_enabled=False, meta_enabled=False, **kw))
        base = rnd(1000)
        W = add(n, base)                                       # the winner
        G = add(n, at_cos(base, 0.75, 1001))                   # grown neighbour
        Gc = add(n, at_cos(base, 0.93, 1002))                  # grown, past the ceiling
        S = add(n, at_cos(base, 0.78, 1003), origin=Origin.INTENT)   # protected neighbour
        Fz = add(n, at_cos(base, 0.80, 1004), frozen=True)
        Far = add(n, rnd(1005))
        for cid in (W, G, Gc, S, Fz, Far):
            n.concepts[cid].plasticity = 0.08
        v = at_cos(base, 0.97, 1006)                           # the input
        return n, p, v, W, G, Gc, S, Fz, Far

    n6, p6, v6, W6, G6, Gc6, S6, Fz6, Far6 = pull_net()
    cos0 = {c: float(n6.concepts[c].prototype @ v6) for c in (W6, G6, Gc6, S6, Fz6, Far6)}
    old = {c: n6.concepts[c].prototype.copy() for c in cos0}
    moved = p6.nonhebb.pull_prototypes(v6, {W6: 0.97})
    cos1 = {c: float(n6.concepts[c].prototype @ v6) for c in cos0}
    check("a grown neighbour of the winner drifts toward the input", cos1[G6] > cos0[G6],
          f"{cos0[G6]:.4f} -> {cos1[G6]:.4f}")
    check("the pull is gentle", cos1[G6] - cos0[G6] < 0.05, f"{cos1[G6] - cos0[G6]:.4f}")
    check("the winner itself is not moved by the pull", np.allclose(old[W6], n6.concepts[W6].prototype))
    check("a seeded (protected) neighbour is never pulled",
          np.allclose(old[S6], n6.concepts[S6].prototype))
    check("a frozen neighbour is never pulled", np.allclose(old[Fz6], n6.concepts[Fz6].prototype))
    check("an unrelated unit is never pulled", np.allclose(old[Far6], n6.concepts[Far6].prototype))
    check("a neighbour already past the ceiling is not pulled closer",
          np.allclose(old[Gc6], n6.concepts[Gc6].prototype))
    check("the pull reports how many it moved and marks the matrix stale", moved == 1 and n6._dirty)
    check("prototypes stay unit length",
          all(abs(float(np.linalg.norm(c.prototype)) - 1.0) < 1e-5 for c in n6.concepts.values()))

    n6b, p6b, v6b, W6b, G6b, *_ = pull_net()
    sims = []
    for i in range(2000):
        p6b.nonhebb.pull_prototypes(v6b, {W6b: 0.97})
        sims.append(float(n6b.concepts[G6b].prototype @ n6b.concepts[W6b].prototype))
    check("repeated pulls stop at the ceiling: a neighbourhood never collapses",
          max(sims) < p6b.nonhebb.cfg.neigh_pull_ceiling + 0.02, f"max sim to winner {max(sims):.4f}")
    check("and the neighbour did get closer before stopping", sims[-1] > 0.80, f"{sims[-1]:.4f}")

    n6c, p6c, v6c, W6c, G6c, *_ = pull_net(neigh_enabled=False)
    o = n6c.concepts[G6c].prototype.copy()
    check("with the mechanism off nothing is pulled",
          p6c.nonhebb.pull_prototypes(v6c, {W6c: 0.97}) == 0
          and np.allclose(o, n6c.concepts[G6c].prototype))
    n6d, p6d, v6d, W6d, G6d, *_ = pull_net(neigh_pull=0.0)
    check("a zero pull strength moves nothing", p6d.nonhebb.pull_prototypes(v6d, {W6d: 0.97}) == 0)
    n6e, p6e, v6e, W6e, G6e, *_ = pull_net()
    check("a weak bottom-up match does not spread",
          p6e.nonhebb.pull_prototypes(v6e, {W6e: 0.1}) == 0)
    check("an empty input pulls nothing", p6e.nonhebb.pull_prototypes(v6e, {}) == 0)
    check("a zero-length input vector is ignored",
          p6e.nonhebb.pull_prototypes(np.zeros(DIM), {W6e: 0.9}) == 0)
    n6f, p6f, v6f, W6f, G6f, *_ = pull_net()
    lo = p6f.nonhebb.pull_prototypes(v6f, {W6f: 0.35})
    c_lo = float(n6f.concepts[G6f].prototype @ v6f)
    n6g, p6g, v6g, W6g, G6g, *_ = pull_net()
    p6g.nonhebb.pull_prototypes(v6g, {W6g: 0.97})
    c_hi = float(n6g.concepts[G6g].prototype @ v6g)
    n6h, _p6h, _v6h, _W6h, G6h, *_ = pull_net()
    c_start = float(n6h.concepts[G6h].prototype @ v6f)
    check("a stronger match pulls harder", (c_hi - c_start) > (c_lo - c_start) > 0,
          f"{c_lo - c_start:.5f} < {c_hi - c_start:.5f}")

    # Through ConceptSystem, the real entry point.
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    cs.net.k = 1
    cs.net.vigilance = 0.3
    cs.plasticity.cfg.adaptation_strength = 0.0
    sb = rnd(1100)
    seed_c = cs.net.add_concept(sb, label="seed", origin=Origin.INTENT, intent="seed")
    prot = cs.net.add_concept(at_cos(sb, 0.80, 1101), label="prot", origin=Origin.INTENT, intent="prot")
    grown = cs.net.add_concept(at_cos(sb, 0.76, 1102), origin=Origin.GROWN)
    grown.plasticity = 0.08
    g0 = grown.prototype.copy()
    p0 = prot.prototype.copy()
    q = at_cos(sb, 0.97, 1103)
    act = cs.perceive(q)
    check("ConceptSystem.perceive applies the pull to a grown neighbour",
          seed_c.id in act.initial and not np.allclose(g0, grown.prototype))
    check("and leaves a seeded neighbour where the catalog put it", np.allclose(p0, prot.prototype))
    pulls_before = cs.plasticity.nonhebb.stats["prototype_pulls"]
    cs.perceive(q, learn=False)
    check("learn=False applies no pull", cs.plasticity.nonhebb.stats["prototype_pulls"] == pulls_before)
    cs.enabled = False
    cs.perceive(q)
    check("a disabled concept system applies no pull",
          cs.plasticity.nonhebb.stats["prototype_pulls"] == pulls_before)

    # ===================================================================
    print("\n=== 7. Interactions and stability ===")
    def workload(seed, nh, cycles=1500):
        rng = np.random.default_rng(seed)
        n, p = make(nh=nh)
        centers = [rnd(2000 + i) for i in range(4)]
        ids_w = []
        for ci, cv in enumerate(centers):
            for k in range(5):
                ids_w.append(add(n, at_cos(cv, 0.82 + 0.02 * k, 2100 + ci * 10 + k)))
        hub = ids_w[0]
        for i in range(cycles):
            k = int(rng.integers(2, 5))
            members = [int(x) for x in rng.choice(ids_w, size=k, replace=False)]
            if rng.random() < 0.5 and hub not in members:
                members.append(hub)
            cycle(n, p, {m: float(rng.uniform(0.5, 1.0)) for m in members})
            if i % 50 == 0:
                p.reward(float(rng.uniform(-0.5, 1.0)))
            if i % 500 == 499:
                p.consolidate()
        return n, p, hub

    nh_on = NonHebbianConfig()
    nh_off = NonHebbianConfig(hetero_enabled=False, meta_enabled=False, neigh_enabled=False)
    n_on, p_on, hub_on = workload(5, nh_on)
    n_off, p_off2, hub_off = workload(5, nh_off)
    w_on = np.array([s.weight() for s in n_on.synapses.values()])
    w_off = np.array([s.weight() for s in n_off.synapses.values()])
    check("a long random workload stays finite with everything on", np.isfinite(w_on).all(),
          f"{len(w_on)} synapses")
    check("weights stay bounded", w_on.max() < 3.0 and w_on.min() > -3.0,
          f"[{w_on.min():.2f}, {w_on.max():.2f}]")
    check("no activity record is NaN or negative",
          all(math.isfinite(c.rate) and c.rate >= 0.0 for c in n_on.concepts.values()))
    check("nothing is below -1.5 or above 1.5 in the persistent part (the clip)",
          all(-1.5 - 1e-6 <= s.w_slow <= 1.5 + 1e-6 for s in n_on.synapses.values()))
    check("the mechanisms actually ran in the workload",
          p_on.nonhebb.stats["hetero_events"] > 0 and p_on.nonhebb.stats["neigh_edges"] > 0,
          str(p_on.nonhebb.stats))
    check("and did not run when switched off",
          all(v == 0 for v in p_off2.nonhebb.stats.values()))

    def out_total(net, cid):
        return sum(abs(s.weight()) for s in net._out.get(cid, {}).values())

    def top_share(net):
        tot = {c: out_total(net, c) for c in net.concepts}
        s = sum(tot.values()) or 1.0
        return max(tot.values()) / s

    check("hub domination is no worse with the extra mechanisms on",
          top_share(n_on) <= top_share(n_off) + 0.05,
          f"top node share {top_share(n_on):.3f} vs {top_share(n_off):.3f}")
    n_on2, p_on2, _ = workload(5, nh_on)
    same = (len(n_on.synapses) == len(n_on2.synapses)
            and all(k in n_on2.synapses and abs(n_on.synapses[k].weight() - n_on2.synapses[k].weight()) < 1e-9
                    for k in n_on.synapses))
    check("the same workload produces the same graph (deterministic)", same)
    check("scaling at sleep still caps every node's outgoing weight",
          all(out_total(n_on, c) <= p_on.cfg.weight_budget + 0.05 or
              all(s.frozen for s in n_on._out.get(c, {}).values()) for c in n_on.concepts))

    d_json = json.dumps(p_on.status(), default=str)
    check("the status report serialises and carries the new counters",
          "nonhebbian" in json.loads(d_json) and "hetero_events" in d_json)

    # Each can be switched off independently, and each alone changes behaviour.
    base_cfg = dict(hetero_enabled=False, meta_enabled=False, neigh_enabled=False)
    results = {}
    for name in ("hetero_enabled", "meta_enabled", "neigh_enabled"):
        cfg = dict(base_cfg)
        cfg[name] = True
        n_x, p_x, _ = workload(5, NonHebbianConfig(**cfg), cycles=600)
        results[name] = np.array([s.weight() for s in n_x.synapses.values()])
    n_base, p_base, _ = workload(5, NonHebbianConfig(**base_cfg), cycles=600)
    wb = np.array([s.weight() for s in n_base.synapses.values()])

    def differs(a, b):
        return a.shape != b.shape or not np.allclose(a, b)

    check("heterosynaptic depression alone changes the outcome", differs(results["hetero_enabled"], wb))
    check("metaplasticity alone changes the outcome", differs(results["meta_enabled"], wb))
    check("neighbourhood plasticity alone changes the outcome", differs(results["neigh_enabled"], wb))

    # Rollback still works with the new edges in play.
    n_rb, p_rb = make()
    ia = add(n_rb, rnd(3000), origin=Origin.INTENT)
    ib = add(n_rb, at_cos(n_rb.concepts[ia].prototype, 0.9, 3001))
    n_rb.link(ia, ib, 0.5, bidirectional=False)
    w0 = n_rb.synapses[(ia, ib)].w_slow
    scores = iter([1.0, 0.1])
    for i in range(10):
        cycle(n_rb, p_rb, {ia: 1.0, ib: 1.0} if i % 2 == 0 else {ia: 1.0})
    rep = p_rb.consolidate(evaluator=lambda: next(scores, 0.1))
    check("a regressing sleep pass still rolls back", rep.get("ROLLED_BACK") is True)
    check("and the pre-existing edge is restored", abs(n_rb.synapses[(ia, ib)].w_slow - w0) < 1e-9)

    # Developmental / neuromodulatory gain still scale everything consistently.
    n_g, p_g = make()
    A, B, X, Y = (add(n_g, rnd(3100 + i)) for i in range(4))
    for i in range(20):
        cycle(n_g, p_g, {A: 1.0, B: 1.0} if i % 2 == 0 else {X: 1.0, Y: 1.0})
    check("maturity and neuromodulation are unaffected by the new mechanisms",
          0.0 <= p_g.maturity() <= 1.0 and p_g.developmental_gain() >= 1.0)

    # Cost: bounded work per cycle.
    import time as _t
    n_c, p_c = make()
    big_ids = []
    for ci in range(30):
        cv = rnd(4000 + ci)
        for k in range(10):
            big_ids.append(add(n_c, at_cos(cv, 0.8 + 0.01 * k, 4100 + ci * 20 + k)))
    rng2 = np.random.default_rng(11)
    t0 = _t.perf_counter()
    for i in range(150):
        ms = [int(x) for x in rng2.choice(big_ids, size=6, replace=False)]
        cycle(n_c, p_c, {m: 0.9 for m in ms})
    per = (_t.perf_counter() - t0) / 150 * 1000
    check("a cycle over a 300-unit graph stays cheap", per < 60.0, f"{per:.1f} ms per cycle")

    print(f"\n     {n_checks} checks")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
