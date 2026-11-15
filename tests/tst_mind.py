"""
tst_mind.py -- the mind page: how ASH keeps what it learned, as a map, as lists, and as a list of what is wrong with it.
Everything is read-only and rule-based; the graphs here are made up in temp folders and no real memory file is read.

  M1  the map: every concept placed, clustered and named; every concept keeps a wire; real neighbours listed
  M2  the judgement: a hub, telemetry, unnamed concepts, causal flood and missing wires are found; a healthy graph is not accused
  M3  what it learned: permanent wires, links meaning would not predict, causal chains
  M4  the rest of memory: entities, episodes, core rules, outcomes; missing files are fine
  M5  the live network and the saved files give the same picture; nothing is written
  M6  the page, the server routes, and the offline export
  M7  the later audits: causal both ways, a concept at the centre of everything, orphans, one skill, stale concepts, no route wires

Run:  PYTHONIOENCODING=utf-8 python tests/tst_mind.py
"""

import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import numpy as np  # noqa: E402

import tst_wondering as tw  # noqa: E402
from src.daemon import mind  # noqa: E402

check = tw.check
HTML = os.path.join(HERE, "src", "ui", "static", "mind.html")
DIM = 24


def unit(seed, base=None, noise=0.15):
    r = np.random.RandomState(seed)
    v = (base if base is not None else r.randn(DIM)) + noise * r.randn(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_graph(folder, hub=True, telemetry=True, named=False, causal=True):
    """3 topics of 8 grown concepts (+2 intents, +1 core); a hub that fires for everything; telemetry; many causal arrows."""
    os.makedirs(folder, exist_ok=True)
    topics = [np.random.RandomState(100 + k).randn(DIM) for k in range(3)]
    cons, protos = [], []

    def add(label, origin, seed_text, vec, act, intent=None):
        cons.append({"id": len(cons), "label": label, "origin": origin, "intent": intent, "activations": act, "mean_activation": 0.1,
                     "age_h": 0.5, "frozen": False, "seed_text": seed_text, "rate": 0.2, "rate_cycle": 5, "plasticity": 0.08})
        protos.append(vec)

    add("get_time", "intent", "get time", unit(1), 0, "get_time")
    add("play_media", "intent", "play media", unit(2), 40, "play_media")
    add("rule:no lying", "core", "ASH must not invent facts", unit(3), 5)
    for k in range(3):
        for j in range(8):
            add(("topic%d_%d" % (k, j)) if named else None, "grown", "talk about topic %d item %d" % (k, j), unit(10 * k + j, topics[k]), 30 + j)
    hub_id = None
    if hub:
        hub_id = len(cons)
        add(None, "grown", "memory at 93%", unit(777), 5000)
    if telemetry:
        for t in ("power connected", "battery at 20%", "cpu high 90%"):
            add(None, "grown", t, unit(len(cons) + 900), 60)
    n = len(cons)
    syn = []
    rs = np.random.RandomState(5)
    for i in range(n):
        for j in rs.choice(n, 4, replace=False):
            if i != j:
                kind = "causal" if causal and rs.rand() < 0.6 else "assoc"
                syn.append({"src": i, "dst": int(j), "kind": kind, "w": 0.08, "slow": 0.06, "fast": 0.02,
                            "order": 2.0 if kind == "causal" else 0.0, "n": 9, "frozen": False})
    if hub:
        for i in range(n):
            if i != hub_id:
                syn.append({"src": i, "dst": hub_id, "kind": "assoc", "w": 0.05, "slow": 0.05, "fast": 0.0, "order": 0.0, "n": 3, "frozen": False})
    # one strong wire between two things that mean nothing alike: what experience taught it
    syn.append({"src": 3, "dst": 3 + 8, "kind": "causal", "w": 0.5, "slow": 0.5, "fast": 0.0, "order": 1.5, "n": 400, "frozen": False})
    np.savez_compressed(os.path.join(folder, "prototypes.npz"), ids=np.arange(n, dtype=np.int64), protos=np.stack(protos))
    with open(os.path.join(folder, "graph.json"), "w") as f:
        json.dump({"dim": DIM, "next_id": n, "cycles": 100, "concepts": cons, "synapses": syn}, f)
    return n


def write_memory(root, entities=30, episodes=60):
    ents = {}
    for i in range(entities):
        eid = "e%d" % i
        ents[eid] = {"entity_id": eid, "entity_type": "proper" if i % 3 else "person", "primary_identifiers": {"name": "name%d" % i},
                     "attributes": {"canonical_name": "Name%d" % i, "last_seen_context": "ctx", "updated_at": 1.0e9 + i}}
    ents["d1"] = {"entity_id": "d1", "entity_type": "org", "primary_identifiers": {"name": "name1"}, "attributes": {"canonical_name": "NAME1"}}
    with open(os.path.join(root, "entities.json"), "w") as f:
        json.dump(ents, f)
    with open(os.path.join(root, "episodic_memory.jsonl"), "w") as f:
        for i in range(episodes):
            f.write(json.dumps({"episode_id": str(i), "timestamp": 1.78e9 + i * 3000, "summary": "ep %d" % i,
                                "event_type": "interaction" if i % 5 else "activity",
                                "related_entities": ["e1"] if i % 4 == 0 else [], "importance": 0.5,
                                "access_count": 1 if i % 10 == 0 else 0}) + "\n")
    with open(os.path.join(root, "core_memory.json"), "w") as f:
        json.dump({"r1": {"rule_id": "r1", "category": "constraint", "text": "never lie", "priority": 9, "hard": True}}, f)
    os.makedirs(os.path.join(root, "state", "brain"), exist_ok=True)
    with open(os.path.join(root, "state", "brain", "outcomes.json"), "w") as f:
        json.dump({"get_time": [10.0, 9.0]}, f)
    with open(os.path.join(root, "state", "brain", "critic.json"), "w") as f:
        json.dump({"w": [0.1, -0.2], "updates": 7}, f)


def build_from(root, **kw):
    return mind.build(state_dir=os.path.join(root, "state"), entities_path=os.path.join(root, "entities.json"),
                      episodic_path=os.path.join(root, "episodic_memory.jsonl"), core_path=os.path.join(root, "core_memory.json"), **kw)


def m1_map():
    print("\n=== M1. The map ===")
    root = tempfile.mkdtemp()
    n = make_graph(os.path.join(root, "state", "concepts"))
    write_memory(root)
    r = build_from(root)
    g = r["graph"]
    check("it reads the saved graph: every concept is there", g["stats"]["concepts"] == n and len(g["concepts"]) == n and not g["live"])
    xs = [c["x"] for c in g["concepts"]] + [c["y"] for c in g["concepts"]]
    check("every concept has a place in [-1.25, 1.25] and the axes' share of the spread is reported",
          max(abs(v) for v in xs) <= 1.25 + 1e-6 and 0 < sum(g["variance_kept"]) < 1)
    check("the three topics land in different clusters (the grown concepts of a topic stay together)",
          len({g["concepts"][3 + j]["cluster"] for j in range(8)}) <= 2 and g["concepts"][3]["cluster"] != g["concepts"][3 + 8]["cluster"])
    check("clusters have sizes and a name from a member", sum(c["size"] for c in g["clusters"]) == n and all(c["name"] for c in g["clusters"]))
    check("a concept with no label is named after what created it, never 'None'",
          all(c["name"] and c["name"] != "None" for c in g["concepts"]) and any("talk about topic" in c["name"] for c in g["concepts"]))
    deg = {i: 0 for i in range(n)}
    for e in g["edges"]:
        deg[e["s"]] += 1
        deg[e["d"]] += 1
    check("every concept keeps at least one wire in the picture", all(v > 0 for v in deg.values()))
    c0 = g["concepts"][3]
    check("each concept lists its wired neighbours and its real neighbours in meaning (cosine)",
          c0["wired"] and len(c0["similar"]) == 5 and all(-1.0 <= s["cos"] <= 1.0 for s in c0["similar"])
          and c0["similar"][0]["cos"] >= c0["similar"][-1]["cos"])
    check("the result is JSON-clean, with no private keys", json.loads(json.dumps(r)) is not None and not any(k.startswith("_") for k in g))
    small = mind.build(state_dir=tempfile.mkdtemp(), entities_path="nope", episodic_path="nope", core_path="nope")
    check("no graph at all is fine: ok, graph None, still a result", small["ok"] and small["graph"] is None and small["learned"] is None)
    one = tempfile.mkdtemp()
    os.makedirs(os.path.join(one, "c"))
    np.savez_compressed(os.path.join(one, "c", "prototypes.npz"), ids=np.array([0, 1]), protos=np.stack([unit(1), unit(2)]))
    with open(os.path.join(one, "c", "graph.json"), "w") as f:
        json.dump({"dim": DIM, "concepts": [{"id": 0, "origin": "grown"}, {"id": 1, "origin": "grown"}], "synapses": []}, f)
    t = mind.build(concept_dir=os.path.join(one, "c"), state_dir=one, entities_path="x", episodic_path="x", core_path="x")
    check("two concepts and no wires does not break the projection", t["graph"]["stats"]["concepts"] == 2)


def m2_findings():
    print("\n=== M2. The judgement ===")
    root = tempfile.mkdtemp()
    make_graph(os.path.join(root, "state", "concepts"))
    write_memory(root, entities=60, episodes=80)
    r = build_from(root)
    ids = {f["id"]: f for f in r["findings"]}
    check("the concept that fires for everything is called out, by name and share",
          "hub" in ids and "memory at 93%" in ids["hub"]["detail"] and ids["hub"]["level"] == "bad")
    check("system telemetry as knowledge is called out, with examples",
          "telemetry" in ids and any("power connected" in x or "battery" in x or "cpu" in x for x in ids["telemetry"]["examples"]))
    check("unnamed grown concepts are called out", "unnamed" in ids)
    check("a causal flood is called out", "causal_flood" in ids)
    check("a concept everything points at is called out as a wiring hub", "wiring_hub" in ids)
    check("missing route and inhibit wires are called out", "no_route" in ids and "no_inhibit" in ids)
    check("entities that are only names are called out (types are guesses)", "entity_noise" in ids and ids["entity_noise"]["level"] == "bad")
    check("the same name stored twice is called out", "entity_duplicates" in ids)
    check("episodes with no concept link are called out", "episodes_unlinked" in ids)
    check("the saved age that is the same for everything is called out", "forgotten_time" in ids)
    check("every finding says what to do about it", all(f["fix"] and f["detail"] and f["title"] for f in r["findings"]))
    order = {"bad": 0, "warn": 1, "info": 2}
    check("worst first", [order[f["level"]] for f in r["findings"]] == sorted(order[f["level"]] for f in r["findings"]))
    check("ideas are ranked by how much the findings argue for them, each tied to its findings",
          r["ideas"][0]["priority"] >= r["ideas"][-1]["priority"] and r["ideas"][0]["argued_by"]
          and all(i["what"] and i["like"] in ("BMO", "Jarvis") for i in r["ideas"]))
    clean = tempfile.mkdtemp()
    make_graph(os.path.join(clean, "state", "concepts"), hub=False, telemetry=False, named=True, causal=False)
    c = build_from(clean)
    cid = {f["id"] for f in c["findings"]}
    check("a healthy graph is not accused of a hub, telemetry, unnamed concepts or causal flood",
          not ({"hub", "telemetry", "unnamed", "causal_flood", "wiring_hub"} & cid), str(sorted(cid)))
    check("the telemetry words are what the rule catches, not ordinary talk",
          mind.is_system({"origin": "grown", "seed_text": "battery at 20%"})
          and not mind.is_system({"origin": "grown", "seed_text": "what is the weather like"})
          and not mind.is_system({"origin": "intent", "seed_text": "battery"}))


def m3_learned():
    print("\n=== M3. What it learned ===")
    root = tempfile.mkdtemp()
    make_graph(os.path.join(root, "state", "concepts"))
    write_memory(root)
    L = build_from(root)["learned"]
    check("the strongest permanent wire is the one with the most slow weight", L["solid"] and L["solid"][0]["w"] == 0.5 and L["solid"][0]["slow"] == 0.5)
    top = L["beyond_meaning"][0]
    check("the link meaning would not predict is found: strong, causal, and the two are not alike", top["w"] == 0.5 and top["cos"] < 0.4)
    check("every listed wire names both ends", all(p["from"] and p["to"] for p in L["solid"] + L["beyond_meaning"]))
    check("chains have 3 or more steps, no repeats", all(len(c["path"]) >= 3 and len(set(c["path"])) == len(c["path"]) for c in L["chains"]))
    check("what fires most is led by the hub", L["busiest"][0]["act"] == 5000)
    check("newest are grown concepts, and the count of grown ones is reported", L["grown"] > 0 and all(x["seed"] for x in L["newest"]))


def m4_memory():
    print("\n=== M4. The rest of memory ===")
    root = tempfile.mkdtemp()
    write_memory(root, entities=30, episodes=60)
    M = mind.memory_overview(os.path.join(root, "entities.json"), os.path.join(root, "episodic_memory.jsonl"), os.path.join(root, "core_memory.json"),
                             os.path.join(root, "state", "brain"))
    check("entities: counted by type, with duplicates and names with nothing else",
          M["entities"]["total"] == 31 and M["entities"]["by_type"]["proper"] > 10 and M["entities"]["duplicates"] and M["entities"]["no_facets"] == 31)
    check("episodes: total, by kind, by day, linked and never recalled",
          M["episodes"]["total"] == 60 and M["episodes"]["by_type"]["interaction"] > 40 and M["episodes"]["by_day"]
          and M["episodes"]["entity_linked"] == 15 and M["episodes"]["never_recalled"] == 54 and M["episodes"]["concept_linked"] == 0)
    check("core rules and their weight", M["core"]["total"] == 1 and M["core"]["rules"][0]["hard"])
    check("outcomes and the critic", M["brain"]["outcomes"][0]["tool"] == "get_time" and M["brain"]["critic"]["updates"] == 7)
    E = mind.memory_overview("missing", "missing", "missing", "missing")
    check("missing files are an empty picture, not an error", E["entities"]["total"] == 0 and E["episodes"]["total"] == 0 and E["core"]["total"] == 0)
    bad = os.path.join(root, "broken.jsonl")
    with open(bad, "w") as f:
        f.write('{"timestamp": 1}\nnot json\n{"timestamp": 2, "summary": "ok"}\n')
    B = mind.memory_overview("missing", bad, "missing", "missing")
    check("a corrupt line is skipped", B["episodes"]["total"] == 2)


def m5_live_and_readonly():
    print("\n=== M5. Live network, and nothing is written ===")
    from src.concepts.model import Concept, Origin, Synapse, SynType

    class Net:
        pass

    net = Net()
    net.cycles = 9
    net.concepts = {i: Concept(id=i, prototype=unit(i), label=("a%d" % i) if i < 2 else None, origin=Origin.GROWN if i > 1 else Origin.INTENT,
                               activations=10 * i, seed_text="s%d" % i) for i in range(8)}
    net.synapses = {(0, 1): Synapse(src=0, dst=1, w_slow=0.3, kind=SynType.CAUSAL, order_evidence=1.5, potentiations=4),
                    (2, 3): Synapse(src=2, dst=3, w_slow=0.0, w_fast=0.2)}
    live = mind.build(net=net, state_dir=tempfile.mkdtemp(), entities_path="x", episodic_path="x", core_path="x")
    g = live["graph"]
    check("a live network is read directly: marked live, concepts and wires carried over",
          g["live"] and g["stats"]["concepts"] == 8 and g["stats"]["synapses"] == 2 and g["cycles"] == 9)
    check("fast-only wires (this session) count", any(e["fast"] > 0 for e in g["edges"]))
    root = tempfile.mkdtemp()
    make_graph(os.path.join(root, "state", "concepts"))
    write_memory(root)

    def snap():
        return {os.path.join(dp, f): (os.path.getmtime(os.path.join(dp, f)), os.path.getsize(os.path.join(dp, f)))
                for dp, _d, fs in os.walk(root) for f in fs}

    before = snap()
    time.sleep(0.05)
    build_from(root)
    check("building the picture changes no file and creates none", before == snap())
    src = open(os.path.join(HERE, "src", "daemon", "mind.py"), encoding="utf-8").read()
    body = src.split("def standalone_html")[0]
    check("the analysis has no write, no network and no model call in it", not re.search(r"open\([^)]*['\"][wa]b?['\"]|requests|urllib|llm\.|\.save\(", body))


def m6_page():
    print("\n=== M6. The page ===")
    page = open(HTML, encoding="utf-8").read()
    check("a name for a title; a viewport; one column on a narrow screen",
          "<title>ASH Mind</title>" in page and 'name="viewport"' in page and "max-width: 900px" in page)
    check("theme tokens, light and dark", all(t in page for t in ("var(--text", "var(--accent", "var(--line", "var(--warn", "var(--bad)".replace(")", ""))) and "color-scheme: dark" in page
          and '[data-theme="paper"], [data-theme="blossom"] { color-scheme: light; }' in page)
    check("no innerHTML, eval or document.write: a concept name can never be markup",
          "innerHTML" not in page and "eval(" not in page and "document.write" not in page)
    check("four tabs: map, learned, everything else, what is wrong",
          all(t in page for t in ('["map", "Map"]', "What it learned", "Everything else it keeps", "what is wrong")))
    check("the map has a meaning layout and a wiring layout, a search and a way to see only surprising links",
          'value="wiring"' in page and 'id="find"' in page and 'id="beyond"' in page)
    check("labelled for a screen reader", 'role="img"' in page and 'role="tablist"' in page and "aria-live" in page and "aria-selected" in page)
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    check("one script, balanced", len(scripts) == 1 and scripts[0].count("{") == scripts[0].count("}") and scripts[0].count("(") == scripts[0].count(")"))
    import tst_vision_camera as vc
    from src.ui import ConsoleServer, UIBus, attach_ui
    rt = vc.rt_with()
    bus = UIBus()
    on_say, on_control, extra = attach_ui(rt, bus)
    srv = ConsoleServer(bus, port=0, theme="forest", on_say=on_say, on_control=on_control, state_extra=extra)
    if not srv.start():
        check("server started", False)
        return
    try:
        port = int(srv.url.rstrip("/").rsplit(":", 1)[1])

        def get(path, host=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.request("GET", path, headers={"Host": host or "127.0.0.1:%d" % port})
            r = c.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.read()

        st, ct, raw = get("/mind")
        html = raw.decode("utf-8")
        check("GET /mind: html with the themes filled in and the forest theme chosen",
              st == 200 and ct.startswith("text/html") and "/*{{THEMES}}*/" not in html and 'data-theme="forest"' in html
              and all('[data-theme="%s"]' % t in html for t in ("nebula", "blossom", "paper", "forest")))
        check("also /mind.html; the console, hud and vision still work",
              get("/mind.html")[0] == 200 and get("/")[0] == 200 and get("/hud")[0] == 200 and get("/vision")[0] == 200)
        st, ct, raw = get("/api/mind")
        j = json.loads(raw)
        check("GET /api/mind: JSON, ok, with findings and ideas, even when the daemon has no concept network",
              st == 200 and ct.startswith("application/json") and j.get("ok") and "findings" in j and j["ideas"] and "memory" in j)
        check("a foreign Host header is refused for the page and the data (DNS rebinding)",
              get("/mind", host="evil.example")[0] == 403 and get("/api/mind", host="evil.example")[0] == 403)
    finally:
        srv.stop()
    root = tempfile.mkdtemp()
    make_graph(os.path.join(root, "state", "concepts"))
    write_memory(root)
    out = os.path.join(root, "out", "mind.html")
    p = subprocess.run([sys.executable, os.path.join(HERE, "ashctl.py"), "mind", "--root", root, "--out", out], capture_output=True, text=True, cwd=HERE,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=120)
    check("`ashctl.py mind --root X --out F` writes the page", p.returncode == 0 and os.path.exists(out), p.stdout[-200:] + p.stderr[-300:])
    if os.path.exists(out):
        h = open(out, encoding="utf-8").read()
        m = re.search(r"const EMBED = (\{.*?\});\nlet D", h, re.S)
        check("the data is embedded as JSON and parses back, with no '</' that could close the script",
              bool(m) and json.loads(m.group(1))["graph"]["stats"]["concepts"] > 20 and "</" not in m.group(1))
        check("no placeholder is left", "{{" not in h and "/*{{DATA}}*/" not in h)


def m7_more_audits():
    print("\n=== M7. More that can be wrong with the concepts ===")
    now = time.time()

    def con(i, origin="grown", act=10, **kw):
        d = {"id": i, "label": kw.pop("label", None), "origin": origin, "intent": None, "activations": act, "mean_activation": 0.1,
             "age_h": float(i), "frozen": False, "seed_text": "thing %d" % i, "rate": 0.1, "plasticity": 0.08, "created": now - i * 3600.0,
             "last_active": now - 3600.0}
        d.update(kw)
        return d

    def run(cons, syn, protos=None):
        n = len(cons)
        pr = protos if protos is not None else np.stack([unit(100 + i) for i in range(n)])
        a = mind.analyze_graph({"concepts": cons, "synapses": syn, "protos": pr, "live": False, "cycles": 1})
        return {f["id"]: f for f in mind.findings(a, {})}

    def edge(s_, d_, kind="assoc", order=0.0, w=0.1):
        return {"src": s_, "dst": d_, "kind": kind, "w": w, "slow": w, "fast": 0.0, "order": order, "n": 5, "frozen": False}

    # causal both ways
    cons = [con(i) for i in range(14)]
    syn = []
    for i in range(0, 12, 2):
        syn += [edge(i, i + 1, "causal", 1.9), edge(i + 1, i, "causal", 1.8)]
    f = run(cons, syn)
    check("causal arrows in both directions are called out, with the count", "causal_both_ways" in f and "6 pairs" in f["causal_both_ways"]["detail"])
    f = run(cons, [edge(i, i + 1, "causal", 1.9) for i in range(0, 12, 2)])
    check("one-way arrows are not", "causal_both_ways" not in f)
    # orphans
    cons = [con(i) for i in range(15)]
    f = run(cons, [edge(0, 1), edge(1, 2)])
    check("learned concepts with no wires at all are called out", "orphans" in f and len(f["orphans"]["examples"]) > 0)
    f = run(cons, [edge(i, (i + 1) % 15) for i in range(15)])
    check("a connected graph has no orphans", "orphans" not in f)
    # a busy concept at the centre of everything
    base = [unit(200 + i) for i in range(14)]
    centre = np.mean(np.stack(base), axis=0)
    centre = (centre / np.linalg.norm(centre)).astype(np.float32)
    cons = [con(i, act=10) for i in range(14)] + [con(14, act=400)]
    f = run(cons, [edge(i, (i + 1) % 15) for i in range(15)], np.stack(base + [centre]))
    check("a busy concept at the average of all of them is called out", "centroid" in f and "thing 14" in f["centroid"]["examples"][0])
    cons[14]["activations"] = 10
    f = run(cons, [edge(i, (i + 1) % 15) for i in range(15)], np.stack(base + [centre]))
    check("a quiet one at the centre is not", "centroid" not in f)
    # one skill
    cons = [con(i, origin="intent", act=a_, label="s%d" % i, intent="s%d" % i) for i, a_ in enumerate((900, 30, 20, 10))] + [con(4 + i) for i in range(10)]
    f = run(cons, [edge(i, i + 1) for i in range(13)])
    check("one skill that gets nearly all the skill activation is called out", "one_skill" in f and "s0" in f["one_skill"]["detail"])
    cons[0]["activations"] = 40
    check("an even spread is not", "one_skill" not in run(cons, [edge(i, i + 1) for i in range(13)]))
    # stale
    cons = [con(i, act=50, last_active=now - 40 * 86400) for i in range(14)]
    f = run(cons, [edge(i, i + 1) for i in range(13)])
    check("concepts that have not fired in three weeks are called out", "stale" in f)
    check("a saved graph with no times is not accused of staleness", "stale" not in run([con(i, act=50, last_active=0.0) for i in range(14)], [edge(i, i + 1) for i in range(13)]))
    # route wires
    f = run([con(i) for i in range(14)], [edge(i, i + 1) for i in range(13)])
    check("no route wires: told it fills in as skills work, nothing to do by hand", "no_route" in f and "fills in" in f["no_route"]["fix"])
    f = run([con(i) for i in range(14)], [edge(i, i + 1, "route") for i in range(13)])
    check("with route wires it is not told", "no_route" not in f)
    check("the inhibit note is accurate: only for a rule that names a skill", "no_inhibit" in f and "names a skill" in f["no_inhibit"]["detail"])


if __name__ == "__main__":
    for fn in (m1_map, m2_findings, m3_learned, m4_memory, m5_live_and_readonly, m6_page, m7_more_audits):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
