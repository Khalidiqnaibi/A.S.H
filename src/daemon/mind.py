"""
src/daemon/mind.py

A read-only picture of what ASH has learned and how it keeps it: the concept graph (every concept is a point in
384-dimensional meaning space, wired to others by typed, weighted synapses), the people and things it knows, the
episodes it lived through, what it is told never to do, and how its own learning is going.

Nothing here writes anything and nothing here asks a model. It reads the live network when the daemon is running, or the
saved files (state/concepts, entities.json, episodic_memory.jsonl, core_memory.json, state/brain) when it is not, and
returns one JSON-able dict for the /mind page and `python ashctl.py mind`.

It also judges what it reads: `findings` are things in the data that look wrong (one concept swallowing every
activation, system telemetry stored as "knowledge", causal arrows that never had a direction, entities that are one-word
noise) and `ideas` are the concrete changes that would make ASH feel more like BMO and Jarvis, each tied to the finding
that argues for it.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ash.mind")

MAX_CONCEPTS = 1500          # the page draws every one; beyond this keep the busiest
MAX_EDGES = 900
_SYSTEM_RX = re.compile(
    r"\b\d{1,3}\s*%|\bpower\s+(?:connected|disconnected|unplugged)|\bbattery\b|\bcpu\b|\bmemory\s+(?:at|usage)\b"
    r"|\bscreen\s+changed\b|\bidle\s+screen\b|\bswitched\s+to\b|\bwindow\s+(?:opened|closed|changed)\b|\bdisk\b"
    r"|\bnetwork\s+(?:up|down)\b|\bwifi\b|\bcharging\b", re.I)


# ======================================================================
# Reading the graph
# ======================================================================
def graph_from_network(net) -> Optional[Dict[str, Any]]:
    """The live ConceptNetwork as plain data (no copy of the network, nothing is saved)."""
    try:
        cons = list(net.concepts.values())
        if not cons:
            return None
        return {
            "cycles": int(getattr(net, "cycles", 0)),
            "concepts": [{**c.as_dict(), "plasticity": c.plasticity, "last_active": float(c.last_active or 0.0)}
                         for c in cons],
            "synapses": [s.as_dict() for s in net.synapses.values() if abs(s.w_slow) > 1e-4 or abs(s.w_fast) > 1e-3
                         or s.frozen],
            "protos": np.stack([np.asarray(c.prototype, dtype=np.float32) for c in cons]),
            "live": True,
        }
    except Exception:
        logger.debug("live graph unreadable", exc_info=True)
        return None


def graph_from_files(directory: str) -> Optional[Dict[str, Any]]:
    """The saved graph (state/concepts), read-only."""
    gp, pp = os.path.join(directory, "graph.json"), os.path.join(directory, "prototypes.npz")
    if not (os.path.exists(gp) and os.path.exists(pp)):
        return None
    try:
        with open(gp, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        d = np.load(pp)
        ids = [int(i) for i in d["ids"]]
        protos = np.asarray(d["protos"], dtype=np.float32)
        by_id = {int(c["id"]): c for c in meta.get("concepts", [])}
        concepts, rows = [], []
        for i, cid in enumerate(ids):
            if cid in by_id and i < len(protos):
                concepts.append(by_id[cid])
                rows.append(i)
        if not concepts:
            return None
        return {"cycles": int(meta.get("cycles", 0)), "concepts": concepts, "synapses": meta.get("synapses", []),
                "protos": protos[rows], "live": False}
    except Exception:
        logger.debug("saved graph unreadable", exc_info=True)
        return None


# ======================================================================
# Geometry: where each concept sits in meaning space
# ======================================================================
def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def project(protos: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """2-D PCA of the prototypes in [-1, 1], and how much of the spread the two axes keep. The honest caveat of any
    flat map of 384 dimensions: points that look close here may be far apart, so `similar` lists the real neighbours."""
    n = len(protos)
    if n == 0:
        return np.zeros((0, 2)), [0.0, 0.0]
    x = protos - protos.mean(axis=0, keepdims=True)
    if n < 3 or float(np.abs(x).max()) < 1e-9:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1) * 0.6, [0.0, 0.0]
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    total = float((s ** 2).sum()) or 1.0
    kept = [round(float(s[i] ** 2) / total, 4) if i < len(s) else 0.0 for i in range(2)]
    span = float(np.percentile(np.abs(xy), 96)) or float(np.abs(xy).max()) or 1.0   # a few outliers must not squash the rest
    return np.clip(xy / span, -1.25, 1.25).astype(np.float32), kept


def clusters(protos: np.ndarray, k: Optional[int] = None, iters: int = 14) -> np.ndarray:
    """Spherical k-means with a deterministic farthest-first start: which concepts are about the same sort of thing."""
    n = len(protos)
    if n < 6:
        return np.zeros(n, dtype=int)
    k = k or int(min(10, max(2, round((n / 2.0) ** 0.5))))
    u = _unit(protos)
    centres = [0]
    for _ in range(k - 1):
        sim = (u @ u[centres].T).max(axis=1)
        centres.append(int(np.argmin(sim)))
    c = u[centres].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        lab = np.argmax(u @ c.T, axis=1)
        for j in range(k):
            m = u[lab == j]
            if len(m):
                v = m.mean(axis=0)
                c[j] = v / max(float(np.linalg.norm(v)), 1e-9)
    return lab


# ======================================================================
# Names and flags
# ======================================================================
def concept_name(c: Dict[str, Any]) -> str:
    if c.get("label"):
        return str(c["label"])
    ex = c.get("exemplars") or []
    seed = re.sub(r"\s+", " ", str(ex[0][1] if ex and len(ex[0]) == 2 else c.get("seed_text") or "")).strip()
    return (seed[:44] + ("..." if len(seed) > 44 else "")) if seed else f"c{c.get('id')}"


def is_system(c: Dict[str, Any]) -> bool:
    """A concept grown from a reading of the machine or the room: the shared telemetry rule (src/concepts/telemetry.py,
    the one the brain uses to refuse to grow them) or one of the extra words this page also flags."""
    if c.get("origin") not in ("grown", "merged"):
        return False
    text = str(c.get("seed_text") or "")
    try:
        from src.concepts.telemetry import is_telemetry
        if is_telemetry(text, ""):
            return True
    except Exception:
        pass
    return bool(_SYSTEM_RX.search(text))


# ======================================================================
# The concept graph
# ======================================================================
def analyze_graph(g: Dict[str, Any]) -> Dict[str, Any]:
    cons: List[Dict[str, Any]] = list(g["concepts"])
    protos: np.ndarray = g["protos"]
    if len(cons) > MAX_CONCEPTS:
        keep = sorted(range(len(cons)), key=lambda i: -int(cons[i].get("activations", 0)))[:MAX_CONCEPTS]
        keep.sort()
        cons = [cons[i] for i in keep]
        protos = protos[keep]
    n = len(cons)
    idx = {int(c["id"]): i for i, c in enumerate(cons)}
    xy, kept = project(protos)
    cl = clusters(protos)
    u = _unit(protos)
    sim = u @ u.T if n <= 2500 else None

    out_deg, in_deg = collections.Counter(), collections.Counter()
    wired: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    edges: List[Dict[str, Any]] = []
    for s in g["synapses"]:
        a, b = idx.get(int(s["src"])), idx.get(int(s["dst"]))
        if a is None or b is None:
            continue
        w = float(s.get("w", float(s.get("slow", 0.0)) + float(s.get("fast", 0.0))))
        out_deg[a] += 1
        in_deg[b] += 1
        wired[a].append((b, w))
        wired[b].append((a, w))
        cos = float(sim[a, b]) if sim is not None else 0.0
        edges.append({"s": a, "d": b, "k": s.get("kind", "assoc"), "w": round(w, 4),
                      "slow": round(float(s.get("slow", 0.0)), 4), "fast": round(float(s.get("fast", 0.0)), 4),
                      "o": round(float(s.get("order", 0.0)), 3), "n": int(s.get("n", 0)), "fz": bool(s.get("frozen")),
                      "cos": round(cos, 3)})
    acts = np.array([max(0, int(c.get("activations", 0))) for c in cons], dtype=float)
    total_act = float(acts.sum()) or 1.0

    concepts: List[Dict[str, Any]] = []
    for i, c in enumerate(cons):
        nb = sorted(wired[i], key=lambda t: -t[1])[:6]
        similar: List[Dict[str, Any]] = []
        if sim is not None and n > 1:
            row = sim[i].copy()
            row[i] = -2
            for j in np.argsort(-row)[:5]:
                similar.append({"i": int(j), "cos": round(float(row[j]), 3)})
        concepts.append({
            "i": i, "id": int(c["id"]), "name": concept_name(c), "label": c.get("label"), "origin": c.get("origin"),
            "intent": c.get("intent"), "seed": str(c.get("seed_text") or "")[:160],
            "named": bool(c.get("exemplars")), "exemplars": [str(e[1])[:120] for e in (c.get("exemplars") or []) if len(e) == 2],
            "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "cluster": int(cl[i]),
            "act": int(acts[i]), "share": round(float(acts[i]) / total_act, 4),
            "rate": round(float(c.get("rate", 0.0)), 4), "plasticity": round(float(c.get("plasticity", 0.0)), 3),
            "frozen": bool(c.get("frozen")), "system": is_system(c),
            "out": int(out_deg[i]), "in": int(in_deg[i]),
            "wired": [{"i": j, "w": round(w, 3)} for j, w in nb], "similar": similar,
        })

    # Name each cluster after the member closest to its centre, plus what else is in it.
    cluster_info = []
    for j in sorted(set(int(x) for x in cl)):
        members = [i for i in range(n) if int(cl[i]) == j]
        if not members:
            continue
        cen = u[members].mean(axis=0)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-9)
        order = sorted(members, key=lambda i: -float(u[i] @ cen))
        cluster_info.append({
            "id": j, "size": len(members), "name": concepts[order[0]]["name"],
            "examples": [concepts[i]["name"] for i in order[:4]],
            "act": int(sum(acts[i] for i in members)), "system": sum(1 for i in members if concepts[i]["system"]),
        })

    edges.sort(key=lambda e: -abs(e["w"]))
    shown = edges[:MAX_EDGES]
    seen = {(e["s"], e["d"]) for e in shown}
    for i in range(n):                              # every concept keeps its strongest wire, so none float free
        best = max((e for e in edges if e["s"] == i or e["d"] == i), key=lambda e: abs(e["w"]), default=None) \
            if len(edges) <= 6000 else None
        if best is not None and (best["s"], best["d"]) not in seen:
            shown.append(best)
            seen.add((best["s"], best["d"]))

    kinds = collections.Counter(e["k"] for e in edges)
    origins = collections.Counter(str(c.get("origin")) for c in cons)
    return {
        "live": bool(g.get("live")), "cycles": int(g.get("cycles", 0)), "dim": int(protos.shape[1]) if n else 0,
        "variance_kept": kept, "concepts": concepts, "edges": shown, "edge_total": len(edges),
        "clusters": cluster_info, "kinds": dict(kinds), "origins": dict(origins),
        "stats": {"concepts": n, "synapses": len(edges), "frozen": sum(1 for e in edges if e["fz"]),
                  "activations": int(total_act), "system_concepts": sum(1 for c in concepts if c["system"]),
                  "named": sum(1 for c in cons if c.get("label")), "mean_weight": round(float(np.mean(
                      [abs(e["w"]) for e in edges])), 4) if edges else 0.0},
        "_cons": cons, "_sim": sim, "_edges": edges,
    }


# ======================================================================
# What it learned
# ======================================================================
def learned(a: Dict[str, Any]) -> Dict[str, Any]:
    cs, edges = a["concepts"], a["_edges"]

    def nm(i):
        return cs[i]["name"]

    def pair(e):
        return {"s": e["s"], "d": e["d"], "from": nm(e["s"]), "to": nm(e["d"]), "kind": e["k"], "w": e["w"],
                "slow": e["slow"], "fast": e["fast"], "n": e["n"], "cos": e["cos"], "order": e["o"]}

    solid = sorted((e for e in edges if e["slow"] > 0 and not e["fz"]), key=lambda e: -e["slow"])[:14]
    # The links meaning alone would not predict: strong wires between concepts that are far apart in embedding space.
    beyond = sorted((e for e in edges if e["cos"] < 0.40 and e["w"] > 0 and not e["fz"]),
                    key=lambda e: -(e["w"] * (1.0 - max(e["cos"], 0.0))))[:14]
    # Causal chains: follow the strongest directed, order-evidenced edges a -> b -> c -> d.
    nxt: Dict[int, Tuple[int, float]] = {}
    for e in edges:
        if e["k"] == "causal" and e["o"] > 0.5 and e["w"] > 0:
            if e["s"] not in nxt or e["w"] > nxt[e["s"]][1]:
                nxt[e["s"]] = (e["d"], e["w"])
    chains, used = [], set()
    for s0, _ in sorted(nxt.items(), key=lambda kv: -kv[1][1]):
        if s0 in used:
            continue
        path, cur = [s0], s0
        while cur in nxt and nxt[cur][0] not in path and len(path) < 5:
            cur = nxt[cur][0]
            path.append(cur)
        if len(path) >= 3:
            chains.append({"path": path, "names": [nm(i) for i in path]})
            used.update(path)
        if len(chains) >= 8:
            break
    grown = [c for c in cs if c["origin"] in ("grown", "merged")]
    busiest = sorted(cs, key=lambda c: -c["act"])[:12]
    hot = sorted((c for c in cs if c["rate"] > 0), key=lambda c: -c["rate"])[:12]
    cold = sorted((c for c in grown if c["act"] > 0 and c["rate"] < 0.002), key=lambda c: c["rate"])[:10]
    newest = sorted(grown, key=lambda c: -c["id"])[:12]
    return {
        "solid": [pair(e) for e in solid], "beyond_meaning": [pair(e) for e in beyond], "chains": chains,
        "busiest": [{"i": c["i"], "name": c["name"], "act": c["act"], "share": c["share"], "system": c["system"]}
                    for c in busiest],
        "hot": [{"i": c["i"], "name": c["name"], "rate": c["rate"]} for c in hot],
        "faded": [{"i": c["i"], "name": c["name"], "act": c["act"]} for c in cold],
        "newest": [{"i": c["i"], "name": c["name"], "seed": c["seed"]} for c in newest],
        "grown": len(grown),
    }


# ======================================================================
# The rest of memory
# ======================================================================
def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def memory_overview(entities_path: str, episodic_path: str, core_path: str, brain_dir: str,
                    now: Optional[float] = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    out: Dict[str, Any] = {}

    ents = _load_json(entities_path)
    ents = list(ents.values()) if isinstance(ents, dict) else []
    by_type = collections.Counter(str(e.get("entity_type")) for e in ents)
    names: Dict[str, List[str]] = collections.defaultdict(list)
    thin = 0
    rows = []
    for e in ents:
        at = e.get("attributes") or {}
        nm_ = str((e.get("primary_identifiers") or {}).get("name") or at.get("canonical_name") or "")
        names[re.sub(r"[^a-z0-9]", "", nm_.lower())].append(nm_)
        useful = [k for k in at if k not in ("canonical_name", "last_seen_context", "source", "updated_at")]
        thin += 0 if useful else 1
        rows.append({"name": at.get("canonical_name") or nm_, "type": e.get("entity_type"), "facets": len(useful),
                     "seen": float(at.get("updated_at") or e.get("updated_at") or 0.0),
                     "context": str(at.get("last_seen_context") or "")[:140]})
    rows.sort(key=lambda r: (-r["facets"], -r["seen"]))
    dup = [v for k, v in names.items() if k and len(v) > 1]
    out["entities"] = {"total": len(ents), "by_type": dict(by_type.most_common()), "no_facets": thin,
                       "duplicates": dup[:8], "top": rows[:14], "recent": sorted(rows, key=lambda r: -r["seen"])[:8]}

    eps: List[Dict[str, Any]] = []
    try:
        with open(episodic_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        eps.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    types = collections.Counter(str(e.get("event_type")) for e in eps)
    days: Dict[str, int] = collections.Counter()
    for e in eps:
        t = float(e.get("timestamp") or 0.0)
        if t:
            days[time.strftime("%Y-%m-%d", time.localtime(t))] += 1
    last_days = sorted(days.items())[-30:]
    linked = sum(1 for e in eps if e.get("related_entities"))
    cdb = sum(1 for e in eps if e.get("concepts") or e.get("concept_ids"))
    never = sum(1 for e in eps if not e.get("access_count"))
    recent = sorted(eps, key=lambda e: -float(e.get("timestamp") or 0.0))[:10]
    out["episodes"] = {
        "total": len(eps), "by_type": dict(types.most_common()), "by_day": last_days, "entity_linked": linked,
        "concept_linked": cdb, "never_recalled": never,
        "important": sorted(({"t": float(e.get("timestamp") or 0.0), "importance": float(e.get("importance") or 0.0),
                              "text": str(e.get("summary") or "")[:140]} for e in eps),
                            key=lambda r: -r["importance"])[:8],
        "recent": [{"t": float(e.get("timestamp") or 0.0), "type": e.get("event_type"),
                    "text": str(e.get("summary") or "")[:140]} for e in recent],
        "first": min((float(e.get("timestamp") or now) for e in eps), default=0.0),
    }

    core = _load_json(core_path)
    rules = list(core.values()) if isinstance(core, dict) else []
    cat = collections.Counter(str(r.get("category")) for r in rules)
    out["core"] = {"total": len(rules), "by_category": dict(cat),
                   "rules": [{"category": r.get("category"), "text": str(r.get("text") or "")[:200],
                              "priority": r.get("priority"), "hard": bool(r.get("hard"))}
                             for r in sorted(rules, key=lambda r: -float(r.get("priority") or 0))[:30]]}

    brain: Dict[str, Any] = {}
    crit = _load_json(os.path.join(brain_dir, "critic.json"))
    if isinstance(crit, dict):
        brain["critic"] = {"updates": int(crit.get("updates", 0)), "weights": [round(float(x), 3) for x in crit.get("w", [])]}
    outc = _load_json(os.path.join(brain_dir, "outcomes.json"))
    if isinstance(outc, dict):
        brain["outcomes"] = [{"tool": k, "tries": float(v[0]), "success": float(v[1]) if len(v) > 1 else 0.0}
                             for k, v in outc.items() if isinstance(v, (list, tuple)) and v]
    out["brain"] = brain
    return out


def learning_overview(state_dir: str) -> Dict[str, Any]:
    """What the newer learners hold, when they exist: asks timing, dwell, camera detections."""
    out: Dict[str, Any] = {}
    timing = _load_json(os.path.join(state_dir, "asks", "timing.json"))
    if isinstance(timing, dict) and timing:
        out["timing_cells"] = len(timing.get("cells", timing))
    ledger = os.path.join(state_dir, "asks", "asks.jsonl")
    if os.path.exists(ledger):
        c = collections.Counter()
        try:
            with open(ledger, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        c[str(json.loads(line).get("outcome") or "open")] += 1
                    except ValueError:
                        pass
        except OSError:
            pass
        out["asks"] = dict(c)
    dw = _load_json(os.path.join(state_dir, "dwell.json"))
    if isinstance(dw, dict):
        out["dwell_days"] = len(dw.get("days", {})) if isinstance(dw.get("days"), dict) else 0
    return out


# ======================================================================
# Judgement
# ======================================================================
def _f(level: str, fid: str, title: str, detail: str, fix: str, examples: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"id": fid, "level": level, "title": title, "detail": detail, "fix": fix, "examples": examples or []}


def findings(a: Dict[str, Any], mem: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cs, edges, cons = a["concepts"], a["_edges"], a["_cons"]
    n = len(cs)
    if n:
        top = sorted(cs, key=lambda c: -c["act"])
        if top[0]["share"] >= 0.20 or sum(c["share"] for c in top[:3]) >= 0.45:
            share3 = sum(c["share"] for c in top[:3])
            out.append(_f("bad", "hub", "One idea is swallowing everything",
                          f"“{top[0]['name']}” took {top[0]['share']:.0%} of every activation ever recorded "
                          f"(the top three took {share3:.0%}). A concept that fires for everything tells the network "
                          "nothing, and its wires drag unrelated ideas together.",
                          "Do not let routine system messages become concepts, and cap how much one unit may fire "
                          "(a soft inhibition on a concept whose share of activity keeps climbing).",
                          [f"{c['name']} ({c['share']:.0%})" for c in top[:3]]))
        sysn = [c for c in cs if c["system"]]
        grown = [c for c in cs if c["origin"] in ("grown", "merged")]
        if grown and len(sysn) / len(grown) >= 0.12:
            out.append(_f("bad", "telemetry", "System telemetry is being learned as if it were knowledge",
                          f"{len(sysn)} of {len(grown)} grown concepts are battery, memory, power or screen messages "
                          f"({sum(c['act'] for c in sysn):,} activations). They are readings, not ideas.",
                          "New readings no longer grow concepts (src/concepts/telemetry.py). Take the ones already there "
                          "out with `python tools/concept_cleanup.py` (a look first, then --apply with ASH stopped).",
                          [c["name"] for c in sorted(sysn, key=lambda c: -c["act"])[:5]]))
        if grown:
            unnamed = [c for c in grown if not c["label"] and not c.get("named")]
            if len(unnamed) / len(grown) > 0.6:
                out.append(_f("warn", "unnamed", "Almost every learned concept has no name",
                              f"{len(unnamed)} of {len(grown)} grown concepts are known only by the sentence that "
                              "created them, so ASH cannot say what it has learned in its own words.",
                              "From now on every concept keeps its three most typical sentences and is named by the "
                              "best one; concepts already in the graph gain them as they are used again.",
                              [c["name"] for c in unnamed[:4]]))
        if a.get("_sim") is not None and n > 1:
            sim = a["_sim"]
            gi = [c["i"] for c in grown]
            pairs = []
            for x in range(len(gi)):
                for y in range(x + 1, len(gi)):
                    v = float(sim[gi[x], gi[y]])
                    if v >= 0.93:
                        pairs.append((v, gi[x], gi[y]))
            if pairs:
                pairs.sort(reverse=True)
                out.append(_f("warn", "duplicates", "Concepts that mean the same thing are kept apart",
                              f"{len(pairs)} pairs of grown concepts are 93% or more alike in meaning, so what ASH "
                              "learns about one is not shared with the other.",
                              "Merge near-duplicate concepts in the sleep pass (the machinery exists: the MERGED "
                              "origin) and add their wires together.",
                              [f"{cs[x]['name']}  =  {cs[y]['name']}  ({v:.0%})" for v, x, y in pairs[:4]]))
        dead = [c for c in cs if c["origin"] in ("intent", "tool") and c["act"] == 0]
        ints = [c for c in cs if c["origin"] in ("intent", "tool")]
        if ints and len(dead) / len(ints) >= 0.5:
            out.append(_f("info", "dead_intents", "Most skills have never been reached through the graph",
                          f"{len(dead)} of {len(ints)} intent concepts never fired.",
                          "Either the phrases for them are too far from how you talk (grow variants from the phrasing "
                          "that did work), or they are rarely wanted. Check against the usage log before pruning.",
                          [c["name"] for c in dead[:6]]))
        ages = [float(c.get("age_h", 0.0)) for c in cons]
        if len(ages) > 20 and float(np.std(ages)) < 0.02:
            out.append(_f("warn", "forgotten_time", "The graph forgets when anything happened",
                          "Every concept shows the same age and its running activation total reads zero: creation "
                          "time, last use and mean strength are not saved, so after each restart ASH cannot tell "
                          "what is new, what is old, or what it has not touched in a month.",
                          "Save created, last_active and activation_sum with each concept (and last_potentiated on "
                          "edges), then derive recency, forgetting and 'this week I learned' from real time."))
    if edges:
        kinds = a["kinds"]
        causal = [e for e in edges if e["k"] == "causal"]
        weak = [e for e in causal if e["o"] < 0.5]
        ceiling = [e for e in causal if e["o"] >= 1.9]
        if len(causal) / len(edges) > 0.35:
            out.append(_f("warn", "causal_flood", "Nearly half the wires claim to be cause and effect",
                          f"{len(causal)} of {len(edges)} edges are causal ({len(causal) / len(edges):.0%}); "
                          f"{len(ceiling)} sit at the ceiling of ordering evidence and {len(weak)} have almost none, "
                          "so the evidence is not separating real sequences from things that merely happen one after "
                          "another in a log.",
                          "Require repeated ordering across different days before an edge is causal, and require an "
                          "outcome (something changed) rather than mere adjacency."))
        hubs = sorted(range(n), key=lambda i: -cs[i]["in"])
        if n >= 20 and cs[hubs[0]]["in"] >= 0.5 * n:
            out.append(_f("warn", "wiring_hub", "Most arrows point at the same few concepts",
                          f"“{cs[hubs[0]]['name']}” is wired to by {cs[hubs[0]]['in']} of {n - 1} other concepts. "
                          "A link that everything has says nothing about this one thing, and spreading activation "
                          "floods through it.",
                          "Scale each incoming wire by how many wires the target already has (a degree penalty), "
                          "so a concept can only be a hub if it earns it.",
                          [f"{cs[i]['name']} (in {cs[i]['in']})" for i in hubs[:3]]))
        for kind, why in (("route", "concept to skill"), ("inhibit", "never-do rules")):
            if not kinds.get(kind):
                out.append(_f("warn" if kind == "route" else "info", f"no_{kind}",
                              f"The graph has no {kind} wires",
                              f"There are no {kind} edges ({why}). "
                              + ("Without them the graph cannot nominate a skill on its own, only the classifier can."
                                 if kind == "route" else
                                 "Core rules are enforced elsewhere, not wired into what the graph can reach."),
                              "Seed route wires from concepts to intents when a request is routed correctly and the "
                              "tool succeeds; seed frozen inhibit wires from each hard core rule." if kind == "route" else
                              "Wire frozen inhibit edges from each hard core rule to the intents it forbids."))
        pending = [e for e in edges if e["fast"] > 0.05 and e["fast"] > 2 * max(e["slow"], 1e-4)]
        if len(pending) > 25:
            out.append(_f("info", "unconsolidated", "A lot of learning is waiting for sleep",
                          f"{len(pending)} wires are strong only in the fast (this-session) weight; if they are not "
                          "consolidated they fade within hours.",
                          "Run the sleep pass when you go idle for a while, not only on a long timer, so the day's "
                          "useful links become permanent.",
                          [f"{cs[e['s']]['name']} -> {cs[e['d']]['name']}" for e in sorted(
                              pending, key=lambda e: -e["fast"])[:3]]))
    ent = mem.get("entities", {})
    if ent.get("total", 0) >= 20:
        noise_types = ("proper", "org", "norp", "work_of_art", "fac", "product", "date", "gpe")
        noisy = sum(v for k, v in ent["by_type"].items() if k in noise_types)
        if ent["no_facets"] / ent["total"] > 0.6:
            out.append(_f("bad", "entity_noise", "People and things are stored as names with nothing about them",
                          f"{ent['no_facets']} of {ent['total']} entities hold only a name and the last sentence they "
                          f"appeared in ({noisy} are tagged as organisation, proper noun, artwork and similar guesses). "
                          "That is a list of words, not knowledge of anyone.",
                          "Keep an entity only after it recurs, give it facets (relationship, likes, routines, last "
                          "seen, how you feel about it) and merge aliases; drop one-off capitalised words.",
                          [r["name"] for r in ent["recent"][:5]]))
        if ent["duplicates"]:
            out.append(_f("warn", "entity_duplicates", "The same name is stored more than once",
                          f"{len(ent['duplicates'])} names appear in several entries.",
                          "Resolve aliases (spelling, case, nicknames) into one entity with several names.",
                          [" / ".join(d[:3]) for d in ent["duplicates"][:4]]))
    ep = mem.get("episodes", {})
    if ep.get("total", 0) >= 50:
        if ep["concept_linked"] == 0:
            out.append(_f("warn", "episodes_unlinked", "Episodes are not linked to the concepts that were active",
                          f"None of the {ep['total']} episodes record which concepts were lit when they were "
                          f"stored, and only {ep['entity_linked']} name an entity. Recall can only match text, not "
                          "follow what the moment was about.",
                          "Store the top five active concept ids on every episode, so recall can walk the graph "
                          "(which the associative layer is built for)."))
        if ep["never_recalled"] / ep["total"] > 0.7:
            out.append(_f("info", "never_recalled", "Most memories are never brought back",
                          f"{ep['never_recalled']} of {ep['total']} episodes were never recalled.",
                          "Surface a memory unprompted when the present moment resembles it ('last time you were on "
                          "this you...'), which is what makes an assistant feel like it was there."))
    br = mem.get("brain", {})
    oc = br.get("outcomes") or []
    if oc and len(oc) <= 4:
        out.append(_f("info", "thin_outcomes", "ASH has seen how very few of its skills turn out",
                      f"Outcome records exist for only {len(oc)} skills, so it cannot yet tell which actions help.",
                      "Record outcomes for every action, including the camera, asks, offers and reminders."))
    order = {"bad": 0, "warn": 1, "info": 2}
    out.sort(key=lambda f: order.get(f["level"], 3))
    return out


# Each idea says what to build, why it makes ASH more like BMO or Jarvis, and which finding argues for it.
IDEAS: List[Dict[str, Any]] = [
    {"id": "facets", "title": "Give every person and thing a set of facets, not a name",
     "like": "Jarvis", "when": ["entity_noise", "entity_duplicates"],
     "what": "Entity = {names, relationship, likes/dislikes, routines, last seen, mood around them, how much to "
             "trust}. Filled from what you say and do, never asked twice. A friend who knows you has a file on you; a "
             "list of capitalised words is not one."},
    {"id": "valence", "title": "Every concept carries a feeling: familiarity, valence, arousal",
     "like": "BMO", "when": ["hub", "unnamed"],
     "what": "Three numbers per concept from real outcomes: how often you meet it, whether things around it went well "
             "(asks accepted, replies warm), how charged it is. BMO reacts to ideas, not only to events; this is where "
             "that comes from, and it costs no model."},
    {"id": "context", "title": "Concept-in-context cells, like the timing model already has",
     "like": "Jarvis", "when": ["telemetry"],
     "what": "Learn what you want when coding at night, versus reading in the morning: a hierarchy of cells "
             "(part of day, app, mode) with a Beta estimate per concept or skill. The asks timing model is the "
             "template; generalise it from 'should I ask now' to 'what would help now'."},
    {"id": "telemetry", "title": "Keep telemetry out of the concept network",
     "like": "Jarvis", "when": ["telemetry", "hub"],
     "what": "Battery, memory and power are the body, not the mind. Feed them to homeostasis as a state vector and let "
             "concepts stand for language and activity. The map becomes readable and the real links stop being "
             "crowded out."},
    {"id": "names", "title": "Name a concept from its three best exemplars, and say it",
     "like": "BMO", "when": ["unnamed"],
     "what": "'I've noticed that when you talk about X you usually mean Y' needs a name for X. Keep exemplars, "
             "re-name as they change, and let ASH narrate the week's new links in a sentence."},
    {"id": "time", "title": "Remember when: creation, last use, forgetting curve",
     "like": "Jarvis", "when": ["forgotten_time"],
     "what": "Persist time on concepts and edges. Then 'new this week', 'fading', and a real forgetting curve are a "
             "few lines, and ASH can say what it learned recently instead of nothing."},
    {"id": "episodes", "title": "Link episodes to the concepts that were active, and bring them back",
     "like": "BMO", "when": ["episodes_unlinked", "never_recalled"],
     "what": "Store the active concept ids on every episode. Recall walks the graph, and a resemblance between now "
             "and a past moment becomes an unprompted 'last time this happened, you...'. Memories that are never "
             "used are not memories."},
    {"id": "causal", "title": "Earn the arrows: causal only after outcomes, across days",
     "like": "Jarvis", "when": ["causal_flood", "no_route", "wiring_hub"],
     "what": "A causal edge should mean 'when A, B follows, and B was worth it'. Require repeated order on different "
             "days plus an outcome; keep the rest as association. Then 'you usually do B after A' can be trusted."},
    {"id": "curiosity", "title": "Ask about the lonely and the surprising",
     "like": "BMO", "when": ["dead_intents"],
     "what": "A concept with few wires, or a wire between two things that are not alike (the 'beyond meaning' list), "
             "is exactly what a curious friend asks about. Feed those into the wondering batch and record the answer "
             "against the concept, so the next question is better."},
    {"id": "why", "title": "Show the reasoning: which wires fired for this reply",
     "like": "Jarvis", "when": [],
     "what": "On request, say 'I thought of Y because you said X and they are linked from the Tuesdays you...'. The "
             "spread path is already computed; surface it, so ASH can be corrected where it is wrong."},
    {"id": "sleep", "title": "Sleep when you are away, and tell the story",
     "like": "BMO", "when": ["unconsolidated"],
     "what": "Consolidate when you go idle for a while, then say one line on return: 'while you were away I tidied "
             "up, and I linked A with B'. This is where being away and 'welcome back' meet what ASH learned."},
]


def ideas(found: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ids = {f["id"] for f in found}
    out = []
    for it in IDEAS:
        hit = [w for w in it["when"] if w in ids]
        out.append({**it, "argued_by": hit, "priority": len(hit)})
    out.sort(key=lambda i: -i["priority"])
    return out


# ======================================================================
# Entry points
# ======================================================================
def build(net=None, concept_dir: Optional[str] = None, state_dir: str = "state", entities_path: Optional[str] = None,
          episodic_path: Optional[str] = None, core_path: Optional[str] = None, brain_dir: Optional[str] = None,
          now: Optional[float] = None) -> Dict[str, Any]:
    """Everything the page shows, as JSON-ready data. `net` (a live ConceptNetwork) wins over the saved files."""
    from src.memory.paths import memory_file
    g = graph_from_network(net) if net is not None else None
    if g is None:
        g = graph_from_files(concept_dir or os.path.join(state_dir, "concepts"))
    mem = memory_overview(entities_path or memory_file("entities.json"),
                          episodic_path or memory_file("episodic_memory.jsonl"),
                          core_path or memory_file("core_memory.json"),
                          brain_dir or os.path.join(state_dir, "brain"), now=now)
    res: Dict[str, Any] = {"ok": True, "time": time.time() if now is None else now, "memory": mem,
                           "learners": learning_overview(state_dir)}
    if g is None:
        res.update(graph=None, learned=None, findings=findings({"concepts": [], "_edges": [], "_cons": [], "kinds": {}},
                                                               mem))
    else:
        a = analyze_graph(g)
        res["learned"] = learned(a)
        res["findings"] = findings(a, mem)
        res["graph"] = {k: v for k, v in a.items() if not k.startswith("_")}
    res["ideas"] = ideas(res["findings"])
    return res


def standalone_html(data: Dict[str, Any], template_path: str) -> str:
    """The /mind page with its data embedded, so it opens from a file with no daemon running."""
    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    blob = json.dumps(data, default=str, separators=(",", ":")).replace("</", "<\/")
    return (html.replace("/*{{THEMES}}*/", "").replace("{{TOKEN}}", "").replace("{{THEME}}", "nebula")
                .replace("/*{{DATA}}*/null", blob, 1))
