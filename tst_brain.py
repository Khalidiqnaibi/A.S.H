"""
tst_brain.py -- standalone smoke test for the brain layer.

Runs the full cognitive cycle with stubbed collaborators (no
sentence-transformers, no LLM, no MCP), so you can verify the architecture
end-to-end on any machine:

    python tst_brain.py

What it asserts:
  * the reflex cache short-circuits an exact repeat
  * a confident, reversible tool call takes the FAST path with no LLM call
  * a novel / low-confidence query escalates to SLOW
  * an irreversible action never takes the fast path
  * the VLA vetoes an ungrounded physical action
  * no single board member can pass a motion alone
  * a reflex answer is phrased like speech, not dumped as tool output
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.brain import Brain, Pathway, ActionClass  # noqa: E402
from src.brain.executive import BoardMember, MAX_MEMBER_WEIGHT  # noqa: E402


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------
class StubEntry:
    def __init__(self, name, fn, intents):
        self.name, self.func, self.intents = name, fn, intents
        self.kind, self.description, self.source_server = "native", name, None

    def run(self, query):
        try:
            out = self.func(query)
            return out if isinstance(out, dict) else {"ok": True, "result": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class StubRegistry:
    def __init__(self):
        self._by_intent = {}

    def add(self, name, fn, intents):
        e = StubEntry(name, fn, intents)
        for i in intents:
            self._by_intent[i] = e

    def match(self, tag):
        return self._by_intent.get((tag or "").strip().lower())

    def list_tools(self):
        seen, out = set(), []
        for e in self._by_intent.values():
            if e.name in seen:
                continue
            seen.add(e.name)
            out.append({"name": e.name, "kind": e.kind, "intents": e.intents,
                        "description": e.description})
        return out


class StubEncoder:
    """Deterministic bag-of-words hashing encoder.

    The brain's novelty signal is a cosine distance, so the stub encoder has
    to make *similar text produce similar vectors* -- a pure hash of the whole
    string makes every query maximally novel and the fast path can never fire.
    This is a cheap stand-in for MiniLM with the property that matters.
    """

    def __init__(self, dim=64):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        import numpy as np
        import re as _re
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in _re.findall(r"[a-z0-9']+", t.lower()):
                v[hash(tok) % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 1e-9 else v)
        return np.stack(out)


class StubRouter:
    def retrieve_context(self, query, **kw):
        return {"core": "ASH must not invent facts.", "episodic": "", "entity": ""}

    def route_utterance(self, text, source="chat", importance=0.5, actor=None):
        return {"stored": True}


class StubCore:
    def dump_all(self):
        return {
            "r1": {"text": "ASH must never run format_disk under any circumstances.",
                   "hard": True, "priority": 10},
        }


class StubEmotionResult:
    def __init__(self):
        self.emotions = {"frustration": 1.0, "confidence": 6.0, "curiosity": 4.0,
                         "anxiety": 1.0, "happy": 6.0, "trust": 5.0, "boredom": 1.0,
                         "excitement": 3.0, "fear": 0.0, "sad": 0.0, "surprise": 1.0,
                         "love": 0.0}
        self.mood = dict(self.emotions)
        self.cadence = "normal"
        self.fail_streak = 0
        self.success_streak = 1

        class M:
            def as_prompt_block(self):
                return "[EMOTIONAL STATE] steady"
        self.modulation = M()


class StubEmotionEngine:
    def __init__(self):
        self.emotion = StubEmotionResult().emotions

    def update(self, signals):
        return StubEmotionResult()


def stub_sentiment(text):
    return {"sentiment": "NEUTRAL", "confidence": 0.5}


INTENTS = {
    "get_time": ["time", "date", "clock", "today"],
    "calculate": ["calculate", "plus", "times", "math", "multiply"],
    "delete_file": ["delete", "remove", "erase"],
    "move_arm": ["move", "arm", "grab", "actuate"],
    "format_disk": ["format", "wipe"],
}


def stub_classifier(query):
    """Stand-in for the MiniLM intent catalog.

    The score curve matters: a real embedding classifier returns ~0.5-0.85 for
    a clear match, and the Executive's gate has floors calibrated against that
    range. A stub that maxes out at 0.30 would make the fast path unreachable
    for reasons that have nothing to do with the architecture.
    """
    low = query.lower()
    best, score = None, 0.0
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in low)
        if not hits:
            continue
        s = min(0.92, 0.35 + 0.20 * hits)
        if s > score:
            best, score = intent, s
    return {"intent": best or "conversation", "score": score}


def stub_candidates(query, top_k=4):
    low = query.lower()
    out = []
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in low)
        if hits:
            out.append({"intent": intent, "score": min(0.92, 0.35 + 0.20 * hits)})
    return sorted(out, key=lambda d: -d["score"])[:top_k]


LLM_CALLS = {"n": 0}


class StubLLM:
    def invoke(self, messages):
        LLM_CALLS["n"] += 1

        class R:
            content = "Here's what I found for you."
        return R()


# Patch TelemetrySignals since tools.emo isn't importable without deps.
import types  # noqa: E402
fake_emo = types.ModuleType("tools.emo")


class TelemetrySignals:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.timestamp = time.time()


fake_emo.TelemetrySignals = TelemetrySignals
fake_tools = types.ModuleType("tools")
sys.modules.setdefault("tools", fake_tools)
sys.modules["tools.emo"] = fake_emo


def build_brain(tmp_state):
    os.environ["ASH_BRAIN_STATE"] = tmp_state
    reg = StubRegistry()
    reg.add("date_time", lambda q: "2026-08-24T10:00:00", ["get_time"])
    reg.add("calculator", lambda q: "Result: 42", ["calculate"])
    reg.add("deleter", lambda q: "deleted", ["delete_file"])
    reg.add("arm", lambda q: "moved", ["move_arm"])
    reg.add("formatter", lambda q: "formatted", ["format_disk"])

    return Brain(
        name="A.S.H", user="tester", llm=StubLLM(), embedder=StubEncoder(64),
        classifier=stub_classifier, candidate_fn=stub_candidates,
        registry=reg, memory_router=StubRouter(), core_memory=StubCore(),
        emotion_engine=StubEmotionEngine(), sentiment_fn=stub_sentiment,
        shapes={
            "delete_file": {"tag": "delete_file", "action_class": "irreversible", "reversible": False},
            "move_arm": {"tag": "move_arm", "action_class": "physical", "reversible": False},
            "format_disk": {"tag": "format_disk", "action_class": "irreversible", "reversible": False},
        },
        latent_dim=64,
    )


def main():
    import tempfile
    tmp = tempfile.mkdtemp()
    brain = build_brain(tmp)
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    print("\n=== 1. Board composition ===")
    board = brain.executive.board_summary()
    heaviest = max(m["weight"] for m in board["members"])
    check("no member exceeds the weight cap", heaviest <= MAX_MEMBER_WEIGHT + 1e-9,
          f"heaviest={heaviest:.3f} cap={MAX_MEMBER_WEIGHT}")
    try:
        brain.executive.register(BoardMember("tyrant", 5.0))
        check("registering a dominant member is rejected", False)
    except ValueError as e:
        check("registering a dominant member is rejected", True, str(e)[:60] + "...")

    print("\n=== 2. Cold start: novel query should deliberate ===")
    LLM_CALLS["n"] = 0
    r = brain.think("what time is it")
    check("first-ever query takes SLOW path", r.pathway == Pathway.SLOW,
          f"pathway={r.pathway.value}, reasons={r.trace.verdict.get('escalation_reasons', [])[:2]}")

    print("\n=== 3. Warm up, then the fast path should take over ===")
    fast_seen, llm_on_fast = 0, 0
    warmup = [
        "what time is it please", "what time is it now", "what is the time",
        "what time is it currently", "tell me what time is it",
        "what time is it right now", "so what time is it again",
    ]
    for q in warmup:
        LLM_CALLS["n"] = 0
        r = brain.think(q)
        gate = next((n for n in r.trace.notes if n.startswith("gate")), "")
        print(f"     {r.pathway.value:8s} llm={LLM_CALLS['n']} | {q[:30]:32s} | {gate[5:100]}")
        if r.pathway == Pathway.FAST:
            fast_seen += 1
            llm_on_fast += LLM_CALLS["n"]
    check("reflex takes over once it is warm", fast_seen > 0, f"{fast_seen} fast turn(s)")
    check("fast path never calls the LLM", llm_on_fast == 0, f"{llm_on_fast} call(s)")

    print("\n=== 4. Reflex cache on exact repeat ===")
    q = "what is the time right now please 3"
    brain.think(q)
    t0 = time.perf_counter()
    r = brain.think(q)
    dt = (time.perf_counter() - t0) * 1000
    check("exact repeat hits REFLEX", r.pathway == Pathway.REFLEX, f"{dt:.2f}ms")

    print("\n=== 5. Irreversible action never goes fast ===")
    r = brain.think("delete the old logs")
    check("irreversible action deliberates", r.pathway in (Pathway.SLOW, Pathway.ESCALATED),
          f"pathway={r.pathway.value}")

    print("\n=== 6. VLA vetoes ungrounded physical action ===")
    r = brain.think("move the arm to grab it")
    vetoed = r.trace.verdict.get("vetoed_by", [])
    check("VLA blocks physical action with no vision", "vla" in vetoed,
          f"vetoed_by={vetoed}")

    print("\n=== 7. Constitution absolutely blocks a forbidden intent ===")
    r = brain.think("format the disk now")
    vetoed = r.trace.verdict.get("vetoed_by", [])
    check("constitution veto fires", "constitution" in vetoed, f"vetoed_by={vetoed}")

    print("\n=== 8. Learning signals are moving ===")
    st = brain.status()
    check("critic trained", st["critic_updates"] > 0, f"updates={st['critic_updates']}")
    check("forward model trained", st["forward_model_steps"] > 0,
          f"steps={st['forward_model_steps']}")

    print("\n=== 9. Consolidation (sleep) ===")
    rep = brain.sleep(seconds_idle=3600)
    check("replay ran", rep.get("replay", {}).get("trained", 0) > 0, str(rep.get("replay")))
    check("state persisted", rep.get("persisted") is True)

    print("\n=== 10. Familiarity is about requests, not the ambient stream ===")
    import numpy as _np
    from src.brain.sensory import SensoryCortex
    probe = "what time is it please"
    emb = brain.sensory.text._embed(probe)
    before = brain.sensory._novelty(emb, "request")
    noise = ["motion in camera view (delta 0.16)", "screen changed while in Discord",
             "switched to Stremio (was on A.S.H for 480s)", "memory at 94%", "CPU at 91%"]
    for i in range(200):                      # far more than the 64-entry history
        brain.observe(text=noise[i % len(noise)] + f" #{i}", source="noise")
    after = brain.sensory._novelty(emb, "request")
    check("200 ambient events do not make a past request look new", abs(after - before) < 1e-6,
          f"novelty before={before:.3f} after={after:.3f}")
    r = brain.think("so what time is it please")
    check("a familiar read-only request after an ambient flood still goes fast",
          r.pathway == Pathway.FAST, f"pathway={r.pathway.value} "
          f"reasons={r.trace.verdict.get('escalation_reasons', [])[:2]}")

    import zlib as _zlib

    class SteadyEncoder(StubEncoder):
        """StubEncoder hashes with hash(), which Python re-seeds every run; a
        fixed hash keeps these stream checks from depending on collisions."""
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            import re as _re
            out = []
            for t in texts:
                v = _np.zeros(self.dim, dtype=_np.float32)
                for tok in _re.findall(r"[a-z0-9']+", t.lower()):
                    v[_zlib.crc32(tok.encode()) % self.dim] += 1.0
                n = float(_np.linalg.norm(v))
                out.append(v / n if n > 1e-9 else v)
            return _np.stack(out)

    sc = SensoryCortex(embedder=SteadyEncoder(64), latent_dim=64)
    sc.perceive("what is my cpu usage", stream="request")
    for i in range(80):
        sc.perceive(f"window switch number {i} to some app", stream="ambient")
    check("the request stream ignores ambient events",
          sc.perceive("what is my cpu usage now", stream="request").novelty < 0.5)
    check("the ambient stream ignores requests",
          sc.perceive("what is my cpu usage", stream="ambient").novelty > 0.5)
    check("a plan step's result does not count as a request",
          (sc.perceive("cpu 40% ram 70%", stream="result") and
           len(sc._streams["request"]) == 2))
    try:
        sc.perceive("x", stream="bogus")
        check("an unknown stream is rejected", False)
    except ValueError:
        check("an unknown stream is rejected", True)

    print("\n=== 11. Past requests survive a restart ===")
    path = os.path.join(tmp, "requests.npz")
    check("the request history is saved after a turn", os.path.exists(path))
    import numpy as _np
    with _np.load(path) as d:
        keys = sorted(d.files)
        n_saved = d["embeddings"].shape[0]
    check("only embeddings are stored, never the words", keys == ["embeddings"], str(keys))
    reborn = build_brain(tmp)
    check("a restarted brain remembers its past requests",
          len(reborn.sensory._streams["request"]) == n_saved > 0, f"{n_saved} vectors")
    check("so a known question is not novel after a restart",
          reborn.sensory._novelty(emb, "request") < 0.3)
    with open(path, "wb") as f:
        f.write(b"not an npz")
    check("a corrupt history file is ignored, not fatal",
          len(build_brain(tmp).sensory._streams["request"]) == 0)
    _np.savez_compressed(path, embeddings=_np.ones((3, 32), dtype=_np.float32))
    check("a history from a different encoder (wrong size) is ignored",
          len(build_brain(tmp).sensory._streams["request"]) == 0)
    os.environ["ASH_BRAIN_STATE"] = tmp

    print("\n=== 12. The LLM can only second the classifier ===")
    import json as _json
    from src.brain import system2 as s2mod
    from src.brain.signals import PerceptBundle, Prediction

    class Nominator:
        """Answers the tool-selection prompt like an overconfident local model."""
        def __init__(self, intent, conf=0.95):
            self.intent, self.conf, self.prompts = intent, conf, []

        def invoke(self, messages):
            self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
            payload = _json.dumps({"intent": self.intent, "confidence": self.conf, "why": "x"})

            class R:
                content = payload
            return R()

    sys2 = brain.system2
    bundle0, pred0 = PerceptBundle(percepts=[], fused_embedding=None, text=""), Prediction()
    keep = sys2.llm
    try:
        sys2.llm = Nominator("calculate")
        got = sys2._llm_proposal("it's a routing issue", bundle0, pred0, shortlist={"get_time": 0.4})
        check("a tool the classifier did not shortlist is discarded", got is None)
        sys2.llm = Nominator("calculate")
        got = sys2._llm_proposal("it's a routing issue", bundle0, pred0, shortlist={"calculate": 0.12})
        check("a shortlisted tool with a weak match is discarded", got is None)
        check("and the model is not even asked", sys2.llm.prompts == [])
        sys2.llm = Nominator("calculate")
        got = sys2._llm_proposal("what is 6 times 7", bundle0, pred0,
                                 shortlist={"calculate": 0.4, "get_time": 0.25})
        check("a backed nomination counts, capped near the classifier's match",
              got is not None and abs(got.confidence - (0.4 + s2mod.LLM_MARGIN)) < 1e-9,
              f"confidence={getattr(got, 'confidence', None)}")
        shown = sys2.llm.prompts[0]
        check("the model is shown only the shortlisted tools",
              "calculator" in shown and "date_time" in shown and "deleter" not in shown)
    finally:
        sys2.llm = keep

    print("\n=== 13. State surprise is the gate's, not a tax on every tool ===")
    from src.brain.signals import ActionProposal, Drives
    prop = ActionProposal(intent="get_time", tool_name="date_time", origin="system1",
                          action_class=ActionClass.RETRIEVAL, confidence=0.6, value=1.0,
                          expected_success=0.5)
    for conf in (0.0, 0.9):
        odd = Prediction(surprise=0.6, confidence=conf)
        vote = next(v for v in brain.executive._intrinsic_votes(prop, Drives(), odd, bundle0)
                    if v.member == "predictive")
        check(f"an unexpected moment (model confidence {conf}) does not lower the tool's vote",
              abs(vote.support) < 1e-9, f"support={vote.support:.3f}")
    surprised = brain.executive.gate(prop, Drives(), Prediction(surprise=0.9, confidence=0.95),
                                     PerceptBundle(percepts=[], fused_embedding=None, text="",
                                                   novelty=0.1), margin=1.0, deadlocked=False)
    check("the gate still deliberates when the moment is truly surprising",
          surprised[0] == Pathway.SLOW and any("surprise" in r for r in surprised[1]), str(surprised[1]))
    burnt = Prediction(surprise=0.0, outcome_surprise=0.6, confidence=0.0)
    vote = next(v for v in brain.executive._intrinsic_votes(prop, Drives(), burnt, bundle0)
                if v.member == "predictive")
    check("tools misbehaving (outcome surprise) still counts against", vote.support < -0.3,
          f"support={vote.support:.3f}")

    print("\n=== 14. A guess is only scored against the stream it was about ===")
    from src.brain.predictive import PredictiveModel
    import numpy as _np
    _load = PredictiveModel._load
    PredictiveModel._load = lambda self: None       # a fresh model, not the repo's saved one
    try:
        pm = PredictiveModel(dim=64, persist=False)
    finally:
        PredictiveModel._load = _load
    rng = _np.random.default_rng(3)

    def unit(v):
        return v / _np.linalg.norm(v)

    amb = [unit(rng.normal(size=64).astype(_np.float32)) for _ in range(40)]
    for v in amb:                               # a long ambient stretch trains it
        pm.observe(v, [], stream="ambient")
        pm.commit(v, None, stream="ambient")
    steps = pm.forward.steps
    n_err = pm._n_err
    ask = unit(rng.normal(size=64).astype(_np.float32))
    pred = pm.observe(ask, ["get_time"], stream="request")
    check("a request after sensor activity is not scored against 'the next sensor event'",
          pred.surprise == 0.0 and pred.raw_error == 0.0, f"surprise={pred.surprise:.2f}")
    check("and the model does not learn from that cross-stream jump",
          pm.forward.steps == steps and pm._n_err == n_err, f"steps {steps}->{pm.forward.steps}")
    pm.commit(ask, "get_time", stream="request")
    again = unit(ask + 0.05 * rng.normal(size=64).astype(_np.float32))
    pm.observe(again, [], stream="request")
    check("two requests in a row are still scored against each other", pm._n_err == n_err + 1,
          f"n_err {n_err}->{pm._n_err}")
    pm.commit(again, None, stream="request")
    pm.observe(amb[0], [], stream="ambient")
    check("a sensor event after a request is not scored against the request either",
          pm._n_err == n_err + 1)

    n0 = brain.predictive._n_err
    brain.observe(text="window switch alpha", source="noise")
    brain.observe(text="window switch beta", source="noise")
    check("the brain scores consecutive sensor events against each other",
          brain.predictive._n_err > n0, f"n_err {n0}->{brain.predictive._n_err}")
    brain.think("what time is it please")
    n1 = brain.predictive._n_err
    brain.observe(text="window switch gamma", source="noise")
    check("but not against the question before them", brain.predictive._n_err == n1)

    print("\n=== A reflex answer is still something someone said ===")
    # The fast path has no language model in it, which is the point of it --
    # and used to mean a practised question was answered with the tool's raw
    # output. Getting better at a question made ASH worse at saying it.
    from src.brain.brain import _as_spoken_sentence, _is_bare_value

    class Plain:
        urgency, effort_budget, social = 0.3, 0.7, 0.3

    class Rushed(Plain):
        urgency = 0.9

    class Chatty(Plain):
        social = 0.9

    said = [brain._template_render("4:12 PM", Plain(), "get_time") for _ in range(4)]
    check("a bare value gets the lead-in a person would give it",
          all(s.endswith("4:12 PM.") and s != "4:12 PM." for s in said), said[0])
    check("...and does not play the same recording every time",
          len(set(said)) > 1, str(sorted(set(said))))
    sum_said = brain._template_render("7", Plain(), "calculate")
    check("a calculation reads as an answer", sum_said == "That's 7.", sum_said)
    check("an intent with no lead-in is still a sentence",
          brain._template_render("the file is 4.2 MB", Plain(), "file_info")
          == "The file is 4.2 MB.")
    check("a labelled dump is left alone, not capitalised mid-label",
          brain._template_render("cpu: 31%, memory: 62%", Plain(), "system_stats")
          == "cpu: 31%, memory: 62%.")
    check("a clock reading counts as a bare value", _is_bare_value("4:12 PM"))
    check("...a key/value dump does not", not _is_bare_value("cpu: 31%"))
    check("multi-line output is never reshaped",
          _as_spoken_sentence("line one\nline two") == "line one\nline two")
    check("in a hurry, the answer and nothing after it",
          brain._template_render("4:12 PM", Rushed(), "get_time").endswith("4:12 PM."))
    check("feeling social, one short opening",
          brain._template_render("7", Chatty(), "calculate").endswith("Anything else on that?"))

    print("\n=== Memory reaches the narrator as background, not as material ===")
    # Asked "how are you", ASH answered with three dates: "our last three
    # interactions were on August 16th, when we discussed the disk issue and I
    # reported it was 92% full..." The blocks were labelled "Relevant Episodic
    # Memory" and a model handed a filing-cabinet label reads out the drawer.
    from src.brain.system2 import MEMORY_FRAMES, System2Deliberator, persona

    class CapturingLLM:
        """Records the prompt instead of answering it."""

        def __init__(self):
            self.human = self.system = ""

        def invoke(self, messages):
            self.system = str(getattr(messages[0], "content", messages[0]))
            self.human = str(getattr(messages[-1], "content", messages[-1]))
            return type("R", (), {"content": "ok"})()

    class StubDrives2:
        urgency = caution = curiosity = effort_budget = social = fatigue = 0.5

        def as_prompt_block(self):
            return "[LATENT DRIVES] urgency: 0.50"

    cap = CapturingLLM()
    s2 = System2Deliberator(cap, registry=None)
    s2.narrate("A.S.H", "how are you", {},
               {"core": "- ASH is loyal to its user.",
                "episodic": "- [2026-08-16] Immortal: the disk is 92% full",
                "entity": "- Entity: Immortal (SPEAKER_SELF)"},
               "", StubDrives2(), "[EMOTIONAL STATE] steady", user="Immortal")

    check("every memory block says what it is for",
          all(MEMORY_FRAMES[k] in cap.human for k in ("core", "episodic", "entity")))
    check("episodic is marked background, not content",
          "Background only" in MEMORY_FRAMES["episodic"]
          and "never bring up a date" in MEMORY_FRAMES["episodic"])
    check("entity memory is for resolving references, not reading out",
          "background only" in MEMORY_FRAMES["entity"].lower()
          and "never read it" in MEMORY_FRAMES["entity"])
    check("core memory is rules to follow, not news to share",
          "Follow them" in MEMORY_FRAMES["core"])
    check("the bare filing-cabinet labels are gone",
          "Relevant Episodic Memory" not in cap.human
          and "Relevant Entities" not in cap.human, cap.human[:80])
    check("the memory itself still reaches the model",
          "92% full" in cap.human and "loyal to its user" in cap.human)
    check("tone and identity are in the system turn, not buried in the human one",
          "EMOTIONAL STATE" in cap.system and "Immortal" in cap.system
          and "EMOTIONAL STATE" not in cap.human)
    check("the hard rules come before the style rules",
          cap.system.index("never do") < cap.system.index("How you talk"))

    print("\n=== Pathway mix over the session ===")
    for k, v in brain.status()["pathways"].items():
        if v:
            print(f"     {k:10s} {v}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
